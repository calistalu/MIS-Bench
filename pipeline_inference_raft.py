import os
import torch
import json
import hashlib
import requests
import numpy as np
import pandas as pd
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from decord import VideoReader, cpu
from utils.vision_process import process_vision_info
import subprocess
import librosa
from tqdm import tqdm
from datasets import load_dataset

def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Downloaded {url} to {dest_path}")

def get_video_frames(video_path, num_frames=32, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)
    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()

    if video_path.startswith(('http://', 'https://')):
        cached = os.path.join(cache_dir, f"{video_hash}.mp4")
        if not os.path.exists(cached):
            download_video(video_path, cached)
        video_file_path = cached
    else:
        video_file_path = video_path

    frames_cache = os.path.join(cache_dir, f"{video_hash}_{num_frames}_frames.npy")
    ts_cache = os.path.join(cache_dir, f"{video_hash}_{num_frames}_timestamps.npy")
    if os.path.exists(frames_cache) and os.path.exists(ts_cache):
        return video_file_path, np.load(frames_cache), np.load(ts_cache)

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total = len(vr)
    idxs = np.linspace(0, total - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(idxs).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(i) for i in idxs])

    np.save(frames_cache, frames)
    np.save(ts_cache, timestamps)
    return video_file_path, frames, timestamps

def extract_audio(video_path, audio_path="temp.wav"):
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", 
           "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def format_example(example, user_template):
    # Fill the user prompt template
    # print(f"Extracted text for {example['video']}")
    # print(example["orig_score1_description"])

    user_prompt_filled = user_template.format(
        orig_instruction=example["orig_instruction"],
        orig_response=example["orig_response"],
        orig_criteria=example["orig_criteria"],
        orig_score1_description=example["orig_score1_description"],
        orig_score2_description=example["orig_score2_description"],
        orig_score3_description=example["orig_score3_description"],
        orig_score4_description=example["orig_score4_description"],
        orig_score5_description=example["orig_score5_description"],
    )

    # Compose the message structure
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": example["orig_instruction"]}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": example["video_path"],
                    "total_pixels": 20480*28*28,
                    "min_pixels": 16*28*28,
                },
                {"type": "text", "text": user_prompt_filled},
            ],
        }
    ]
    return {"messages": messages}

def prune_messages(raw_messages):
    """
    接收原始的 raw_messages（每个 message 是 dict，里头有很多冗余字段），
    返回只保留 role + content 中指定字段的新列表。
    """
    pruned = []
    for msg in raw_messages:
        new_msg = {"role": msg["role"], "content": []}
        for item in msg["content"]:
            # 公共字段：type, text, video, total_pixels, min_pixels
            entry = {"type": item["type"]}
            if item["type"] == "text":
                # 只保留 text
                entry["text"] = item["text"]
            elif item["type"] == "video":
                # 保留 video 路径和像素信息
                entry["video"] = item["video"]
                entry["total_pixels"] = item["total_pixels"]
                entry["min_pixels"] = item["min_pixels"]
            else:
                # 如果还有别的 type，你可以根据需要再加 elif
                continue
            new_msg["content"].append(entry)
        pruned.append(new_msg)
    return pruned



def create_collate_fn(processor):
    def collate_fn(examples):
        texts = []
        video_inputs = []
        fps_list = []

        # examples = [prune_messages(example["messages"]) for example in examples]  # 清理冗余字段
        for i, example in enumerate(examples):
            message_list = example
            # 构造对话模板文本
            text = processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            # 处理视频
            image_inputs, video_frames, video_kwargs = process_vision_info([message_list], return_video_kwargs=True)
           
            video_inputs.append(video_frames)               # 累积帧列表
            fps = video_kwargs.get('fps', None)             # 单个样本的 fps
            fps_list.append(fps[0])                            # 收集到列表里
            # print("len(texts):", len(texts))
            # print(i)
            # print("len(video_frames):", len(video_frames))
            # print("len(video_inputs):", len(video_inputs))
                            

        #print(len(texts), len(video_inputs), len(fps_list))
        # 统一调用 processor，注意 videos 和 fps 要对应 samples 数量
        batch = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            fps=fps_list,
            padding=True,
            return_tensors="pt"
        )
        # 构造 labels
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        # print(batch.keys())
        # print(batch["input_ids"])
        # print(batch["labels"])
        # print("=== Input IDs Decoded ===")
        # for text in processor.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False):
        #     print(text)
        #     print("-" * 50)

        # # 打印 labels 对应的文本（把 -100 先还原成 pad_token_id，不然 decode 不出来）
        # print("=== Labels Decoded ===")
        # labels_for_decode = batch["labels"].clone()
        # labels_for_decode[labels_for_decode == -100] = processor.tokenizer.pad_token_id
        # for text in processor.tokenizer.batch_decode(labels_for_decode, skip_special_tokens=False):
        #     print(text)
        #     print("-" * 50)
        return batch

    return collate_fn

import re

# 固定八个 FIS 列名（CSV 按这个顺序输出）
FIS_CATEGORIES = [
    "Verbal Fluency",
    "Hope & Positive Expectations",
    "Persuasiveness",
    "Emotional Expression",
    "Alliance Rupture-Repair Responsiveness",
    "Alliance Bond Capacity",
    "Warmth, Acceptance, & Understanding",
    "Empathy",
]

def _canonicalize_fis(name: str) -> str:
    """把各种写法映射到统一列名。"""
    key = re.sub(r'[^a-z0-9]+', ' ', name.lower()).strip()
    mapping = {
        "verbal fluency": "Verbal Fluency",
        "hope & positive expectations": "Hope & Positive Expectations",
        "persuasiveness": "Persuasiveness",
        "emotional expression": "Emotional Expression",
        "alliance rupture-repair responsiveness": "Alliance Rupture-Repair Responsiveness",
        "alliance bond capacity": "Alliance Bond Capacity",
        "warmth, acceptance, & understanding": "Warmth, Acceptance, & Understanding",
        "empathy": "Empathy",
    }
    return mapping.get(key, name.strip())

def extract_fis_category(user_text: str) -> str:
    """
    从 user_text 中抽取 FIS 类别名：
    取 '###Score Rubrics:' 后面方括号里的内容，并在第一个句号 '.' 前截断。
    """
    m = re.search(r"###Score Rubrics:\s*\[\s*(.*?)\s*\]", user_text, re.DOTALL)
    if not m:
        return "unknown"
    inside = m.group(1).strip()
    # 取第一个句号前面的短语作为类别名
    name = inside.split('.', 1)[0].strip()
    # 清理首尾杂字符
    name = re.sub(r'^[\W_]+|[\W_]+$', '', name)
    return _canonicalize_fis(name)


def main(folder='data/FIS996', output_csv='finetune_cot_basedon_raft.csv'):
    device = "cuda:0"

    cache_vlm = './.cache/qwen2.5'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_vlm,
        torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_vlm)
    model.load_adapter("./output-cot/checkpoint-100")

    data_collator = create_collate_fn(processor)
    raw_dataset = load_dataset("CalistaLu/FIS-Full-Dataset", split="train")
    print(f"Raw dataset loaded with {len(raw_dataset)} examples.")

   
    # Load templates
    with open("prompt/user_template_cot.txt", "r") as f:
        user_template = f.read().strip()

    # Format each example into message format
    formatted_dataset = raw_dataset.map(
        lambda example: format_example(example,user_template)
    )
    # Remove extra columns and keep only "messages"
    dataset = formatted_dataset.remove_columns([
        col for col in formatted_dataset.column_names if col != "messages"
    ])
    #打印第一个message
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = splits["train"]
    temp_ds = splits["test"].train_test_split(test_size=0.5, seed=42)
    eval_ds = temp_ds["train"]
    test_ds = temp_ds["test"]
    
    train_ds = [prune_messages(data["messages"]) for data in dataset]
    eval_ds = [prune_messages(data["messages"]) for data in eval_ds]
    test_ds = [prune_messages(data["messages"]) for data in test_ds]
    print(test_ds[0])

    # inference on each sample in the test dataset, and save the response to a csv file
    # row: video, column: orig_criteria
        # inference on each sample in the test dataset, and save the response to a csv file
    results = {}  # {video_id(无扩展名): {FIS列名: 模型输出文本}}

    for example in tqdm(test_ds, desc="Processing test dataset"):
        messages = example

        # 取视频路径并去掉 .mp4（或任意扩展名）
        video_path = None
        for item in messages[1]["content"]:
            if item["type"] == "video":
                video_path = item["video"]
                break
        if video_path is None:
            continue
        
        # 去掉扩展名
        import os
        video_id = os.path.splitext(video_path)[0]

        # 抽取 user_text 并解析出 FIS 类别名
        user_text = next(c["text"] for c in messages[1]["content"] if c["type"] == "text")
        orig_criteria = extract_fis_category(user_text)  # 期望得到 8 个之一

        # ===== 推理 =====
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
        fps = video_kwargs["fps"]

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            fps=fps,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        outs = model.generate(**inputs, max_new_tokens=512)
        gen = outs[0][inputs.input_ids.shape[-1]:]
        result = processor.batch_decode([gen], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        print(f"Video: {video_id}, Criteria: {orig_criteria}, Result: {result}")

        #（可选）只取数字，如需的话解注释
        # m_score = re.search(r"(-?\d+(?:\.\d)?)", result)
        # if m_score:
        #     result = m_score.group(1)

        # ===== 存储 =====
        if video_id not in results:
            results[video_id] = {}
        results[video_id][orig_criteria] = result

    rows = []
    for video_id, preds in results.items():
        row = {"video": video_id}
        for cat in FIS_CATEGORIES:
            row[cat] = preds.get(cat, "")
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"Saved results to {output_csv}")


if __name__ == '__main__':
    main()
