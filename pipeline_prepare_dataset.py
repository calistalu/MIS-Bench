import os
import torch
import json
import hashlib
import requests
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.model_selection import train_test_split
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from decord import VideoReader, cpu
from vision_process import process_vision_info
import subprocess
import librosa
from tqdm import tqdm

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

# 利用 Whisper 处理长音频

def transcribe_long_audio(audio_path,
                          model,
                          processor,
                          chunk_length_s=30,
                          chunk_overlap_s=2):
    # 读取整段音频
    speech, sr = librosa.load(audio_path, sr=16000, mono=True)
    total_samples = speech.shape[0]
    chunk_size = chunk_length_s * sr
    overlap = chunk_overlap_s * sr

    transcripts = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_size, total_samples)
        chunk = speech[start:end]

        # 处理 chunk
        inputs = processor(chunk,
                           sampling_rate=sr,
                           return_tensors="pt")
        # cast to fp16 if model was loaded in fp16
        inputs = {k: v.half().cuda() for k, v in inputs.items()}

        # 你也可以加上 language="en", task="translate" 等参数
        ids = model.generate(**inputs)
        text = processor.batch_decode(ids,
                                      skip_special_tokens=True)[0]
        transcripts.append(text.strip())
        del inputs, ids
        torch.cuda.empty_cache()

        # 滑动窗口：下一个 chunk 开始点
        start += chunk_size - overlap

    # 拼接所有片段，去掉重复
    full_transcript = " ".join(transcripts)
    return full_transcript

def format_data(coding_row, system_prompt, user_prompt, video_path, total_pixels, min_pixels):
    path, frames, timestamps = get_video_frames(video_path)
    dimensions = [
        'verbal_fluency', 'hope_and_pe', 'persuasiveness',
        'emotional_expression', 'wau', 'empathy', 'abc', 'arrr'
    ]
    # coding_row 只有一行 → 直接取出并转成 float
    scores_dict = {dim: float(coding_row.iloc[0][dim]) for dim in dimensions}

    # 用 json.dumps 而不是 str()，保证合法 JSON 并去掉 np.float64
    scores_text = json.dumps({"scores": scores_dict}, ensure_ascii=False)

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": path, 
                    "total_pixels": total_pixels, 
                    "min_pixels": min_pixels},
                {
                    "type": "text",
                    "text": user_prompt,
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": scores_text}],
        }
    ]
def text_generator(sample_data):
    sample_data = sample_data["messages"]
    text = processor.apply_chat_template(
        sample_data[0:2], tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps = video_kwargs['fps']
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=fps,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    outs = model.generate(**inputs, max_new_tokens=MAX_SEQ_LEN)
    gen = outs[0][inputs.input_ids.shape[-1]:]
    result = processor.batch_decode([gen], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    actual_answer = sample_data[2]["content"][0]["text"]
    del image_inputs, video_inputs, inputs, outs, gen
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return result, actual_answer

# def create_collate_fn(processor):
#     def collate_fn(examples):
#         texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
#         image_inputs = [example[1]["content"][0]["image"] for example in examples]

#         batch = processor(
#             text=texts, images=image_inputs, return_tensors="pt", padding=True
#         )
#         labels = batch["input_ids"].clone()
#         labels[labels == processor.tokenizer.pad_token_id] = -100
#         batch["labels"] = labels  # Use modified labels

#         return batch
#     return collate_fn

def create_collate_fn(processor):
    def collate_fn(examples):
        texts = []
        video_inputs = []
        for example in examples:
            try:
                example = example["messages"]
                # Apply chat template to the example
                text = processor.apply_chat_template(example, tokenize=False)
                texts.append(text)
                # Access video from user message
                video = example[1]["content"][0].get("video", None)
                video_inputs.append(video)
            except (IndexError, KeyError) as e:
                print(f"Error processing example: {example}")
                raise Exception(f"Data structure error: {str(e)}")

        batch = processor(
            text=texts, videos=video_inputs, return_tensors="pt", padding=True
        )
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
    return collate_fn

# 配置参数集中管理（全部大写常量）
DEVICE = "cuda:0"
CACHE_WHISPER = "./.cache/whisper"
WHISPER_MODEL_NAME = "openai/whisper-medium"
CACHE_VLM = "./.cache/qwen2.5"
VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
CODING_CSV = "coding.csv"
OUTPUT_CSV = "fisclipt_scores.csv"
SYSTEM_PROMPT = "system_prompt.txt"
USER_PROMPT = "user_prompt.txt"
VIDEO_FOLDER = "FISclipped"
TOTAL_PIXELS = 20480*28*28
MIN_PIXELS = 16*28*28
MAX_SEQ_LEN = 2048
CHUNK_LENGTH_S = 30
CHUNK_OVERLAP_S = 2

EPOCHS = 10
BATCH_SIZE = 16
GRADIENT_CHECKPOINTING = True,  # Tradeoff between memory efficiency and computation time.
USE_REENTRANT = False,
OPTIM = "paged_adamw_32bit"
LEARNING_RATE = 2e-5
LOGGING_STEPS = 50
EVAL_STEPS = 50
SAVE_STEPS = 50
EVAL_STRATEGY = "steps"
SAVE_STRATEGY = "steps"
METRIC_FOR_BEST_MODEL="eval_loss"
LOAD_BEST_MODEL_AT_END=True
MAX_GRAD_NORM = 1
WARMUP_STEPS = 0
DATASET_KWARGS={"skip_prepare_dataset": True} # We have to put for VLMs
REMOVE_UNUSED_COLUMNS = False # VLM thing

def main():
    
    # 使用常量参数
    device = DEVICE
    cache_whisper = CACHE_WHISPER
    whisper_model_name = WHISPER_MODEL_NAME
    cache_vlm = CACHE_VLM
    vlm_model_name = VLM_MODEL_NAME

    # # 允许外部传参，否则用常量默认
    # folder = folder or VIDEO_FOLDER
    # output_csv = output_csv or OUTPUT_CSV
    # system_prompt = system_prompt or SYSTEM_PROMPT
    # user_prompt = user_prompt or USER_PROMPT

    folder = VIDEO_FOLDER
    coding_csv = CODING_CSV
    output_csv = OUTPUT_CSV
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT

    # 1) 加载并缓存 Whisper 模型一次
    whisper_processor = WhisperProcessor.from_pretrained(
        whisper_model_name, cache_dir=cache_whisper)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        whisper_model_name, cache_dir=cache_whisper,
        torch_dtype=torch.float16, device_map={"": 0}).to(device)

    # 2) 加载并缓存 VLM 模型一次
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_model_name, cache_dir=cache_vlm,
        torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(
        vlm_model_name, cache_dir=cache_vlm)
    # processor = Qwen2VLProcessor.from_pretrained(vlm_model_name)
    # processor.tokenizer.padding_side = "right"
    data_collator = create_collate_fn(processor)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
 

    training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    eval_steps=EVAL_STEPS,
    eval_strategy=EVAL_STRATEGY,
    save_strategy=SAVE_STRATEGY,
    save_steps=SAVE_STEPS,
    metric_for_best_model=METRIC_FOR_BEST_MODEL,
    load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
    max_grad_norm=MAX_GRAD_NORM,
    warmup_steps=WARMUP_STEPS,
    dataset_kwargs=DATASET_KWARGS,
    max_seq_length=MAX_SEQ_LEN,
    remove_unused_columns = REMOVE_UNUSED_COLUMNS,
    optim=OPTIM,
)


    # 3) 读取 prompt 模板
    with open(system_prompt, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()
    with open(user_prompt, 'r', encoding='utf-8') as f:
        prompt_template = f.read().strip()

    
    coding_df = pd.read_csv(coding_csv)
    coding_df['video'] = (coding_df['file_name'].astype(str).str.replace('Time', 'T').str.replace('_FIS', ''))
    
    video_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.mp4')]
    
    messages = []
    #for idx, vid in enumerate(tqdm(sorted(video_paths)[260:]), start=261):
    for idx, vid in enumerate(tqdm(sorted(video_paths)), start=1):
        print(f"Processing {vid}...")
        # 3.1 转录音频
        audio_file = extract_audio(vid)
        transcript = transcribe_long_audio(
            audio_file, whisper_model, whisper_processor,
            chunk_length_s=CHUNK_LENGTH_S, chunk_overlap_s=CHUNK_OVERLAP_S)
        # print(f"Transcript: {transcript[:60]}...")

        # 3.2 格式化 prompt 并调用多模态推理
        video_name = os.path.splitext(os.path.basename(vid))[0]
        matching_row = coding_df[coding_df['video'] == video_name]
        if matching_row.empty:
            print(f"No matching coding row for video {video_name}. Skipping...")
            continue
        # prompt = prompt_template.format(video=video_name, transcript=transcript)
        user_prompt = prompt_template.replace("{transcript}", transcript)
        message = format_data(
            coding_row=matching_row,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            video_path=vid,
            total_pixels=TOTAL_PIXELS,
            min_pixels=MIN_PIXELS
        )
        print(f"Message {idx}: {message}")
        messages.append( message)
    dataset_dict = {"messages": messages}
    dataset = Dataset.from_dict(dataset_dict)
    print("Dataset example:", dataset[0]["messages"])
    dataset.save_to_disk("my_dataset")

#     dataset = load_from_disk("my_dataset")
#     print("Dataset example:", dataset[0])

#     # Step 4: Split the dataset into training, validation, and test sets
#     splits = dataset.train_test_split(test_size=0.2, seed=42)
#     train_ds = splits["train"]
#     temp_ds = splits["test"].train_test_split(test_size=0.5, seed=42)
#     eval_ds = temp_ds["train"]
#     test_ds = temp_ds["test"]

#     # Step 5: Pass these datasets to the trainer
#     trainer = SFTTrainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_ds,
#         eval_dataset=eval_ds,
#         data_collator=data_collator,
#         peft_config=peft_config,
#         processing_class=processor.tokenizer,
#     )


#     print("-"*30)
#     print("Initial Evaluation")
#     metric = trainer.evaluate()
#     print(metric)
#     print("-"*30)

#     print("Training")
#     trainer.train()
#     print("-"*30)

#     trainer.save_model(training_args.output_dir)
        
if __name__ == '__main__':
    main()
