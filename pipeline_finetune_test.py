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
from utils.vision_process import process_vision_info
import subprocess
import librosa
from tqdm import tqdm
from accelerate import Accelerator
from transformers import TrainerCallback, TrainerState, TrainerControl

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

def text_generator(sample_data):
    massages = sample_data
    text = processor.apply_chat_template(
        sample_data[0:2], tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info([massages], return_video_kwargs=True)
  
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

def create_collate_fn(processor):
    def collate_fn(examples):
        texts = []
        video_inputs = []
        fps_list = []
        examples = [prune_messages(example["messages"]) for example in examples]  # 清理冗余字段
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
                         
            # if video_inputs is not None:
            #     t, c, h, w = video_inputs[0].shape
            #     print(f"Video tensor shape: T={t}, C={c}, H={h}, W={w}")

            #     # 查看模型 patch size
            #     ps = processor.feature_extractor.patch_size
            #     print(f"Processor patch_size: {ps}")

            #     # 计算网格
            #     p_h, p_w = (ps, ps) if isinstance(ps, int) else ps
            #     grid_h, grid_w = h // p_h, w // p_w
            #     print(f"Grid: {grid_h}×{grid_w} patches per frame, total_visual_tokens = T × {grid_h*grid_w}")
                        

        # print(len(texts), len(video_inputs), len(fps_list))
        # 统一调用 processor，注意 videos 和 fps 要对应 samples 数量
        batch = processor(
            text=texts,
            videos=video_inputs,
            fps=fps_list,
            padding=True,
            return_tensors="pt"
        )
        # 构造 labels
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

    return collate_fn

# 配置参数集中管理（全部大写常量）
DEVICE = "cuda"
CACHE_WHISPER = "./.cache/whisper"
WHISPER_MODEL_NAME = "openai/whisper-medium"
CACHE_VLM = "./.cache/qwen2.5"
VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
CODING_CSV = "data/coding.csv"
OUTPUT_CSV = "data/fisclipt_scores.csv"
SYSTEM_PROMPT = "prompt/system_prompt.txt"
USER_PROMPT = "prompt/user_prompt.txt"
VIDEO_FOLDER = "data/FIS996"
TOTAL_PIXELS = 20480*28*28
MIN_PIXELS = 16*28*28
MAX_SEQ_LEN = 512
CHUNK_LENGTH_S = 30
CHUNK_OVERLAP_S = 2

EPOCHS = 60
BATCH_SIZE = 2
GRADIENT_CHECKPOINTING = True  # Tradeoff between memory efficiency and computation time.
USE_REENTRANT = False
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

from transformers import TrainerCallback, TrainerState, TrainerControl

class LossPrinterCallback(TrainerCallback):
    """
    打印训练和验证损失：
      • on_log   —— 训练过程中 logging_steps 时触发，打印 train_loss 和（若有）eval_loss
      • on_evaluate —— 每次 evaluate() 完成后触发，打印 eval_loss
    """

    @staticmethod
    def _is_main(args):
        return getattr(args, "local_rank", -1) in (-1, 0)

    # 训练阶段：Trainer 每到 logging_steps 都会进这里
    def on_log(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        if not self._is_main(args) or logs is None:
            return

        step = state.global_step

        if "loss" in logs:        # 训练损失
            print(f"[step {step:>6}] train_loss = {logs['loss']:.4f}")

        if "eval_loss" in logs:   # 有些设置下 eval_loss 也会出现在 logs
            print(f"[step {step:>6}]  eval_loss = {logs['eval_loss']:.4f}")
    def on_evaluate(
            self,
            args,
            state: TrainerState,
            control: TrainerControl,
            metrics=None,
            **kwargs,
        ):
            if not self._is_main(args) or metrics is None:
                return

            if "eval_loss" in metrics:
                step = state.global_step
                print(f"[step {step:>6}]  eval_loss = {metrics['eval_loss']:.4f}")

                
def main():
    # 分布式训练建议用 accelerate
    accelerator = Accelerator()

    device = accelerator.device
    cache_whisper = CACHE_WHISPER
    whisper_model_name = WHISPER_MODEL_NAME
    cache_vlm = CACHE_VLM
    vlm_model_name = VLM_MODEL_NAME
    folder = VIDEO_FOLDER
    coding_csv = CODING_CSV
    output_csv = OUTPUT_CSV
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Load VLM model and processor
    local_rank = accelerator.local_process_index
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vlm_model_name,
        cache_dir=cache_vlm,
        quantization_config=bnb_config,
        device_map={"": local_rank},
    )
    processor = AutoProcessor.from_pretrained(vlm_model_name, cache_dir=cache_vlm, use_fast=True)
    processor.save_pretrained(cache_vlm)

    data_collator = create_collate_fn(processor)

    dataset = load_from_disk("my_dataset")
    splits = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds = splits["train"]
    temp_ds = splits["test"].train_test_split(test_size=0.5, seed=42)
    eval_ds = temp_ds["train"]
    test_ds = temp_ds["test"]
    # 打印test set的第一个example
    print(test_ds[0])

    # PEFT 配置
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    # 1) 额外引入 EarlyStoppingCallback
    from transformers import EarlyStoppingCallback

    # 2) 创建 EarlyStoppingCallback 实例
    early_stop_cb = EarlyStoppingCallback(
        early_stopping_patience=10,   # 连续 10 次评估无提升就停
        early_stopping_threshold=0.0  # 只要有任何下降就认为“提升”
    )

    # SFT 训练参数
    training_args = SFTConfig(
        output_dir="./output",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        logging_first_step=True,
        log_level="info",
        report_to="none",
        eval_steps=EVAL_STEPS,
        eval_strategy=EVAL_STRATEGY,
        save_strategy=SAVE_STRATEGY,
        save_steps=SAVE_STEPS,
        metric_for_best_model=METRIC_FOR_BEST_MODEL,
        greater_is_better=False,  # eval_loss 越小越好
        load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
        max_grad_norm=MAX_GRAD_NORM,
        warmup_steps=WARMUP_STEPS,
        dataset_kwargs=DATASET_KWARGS,
        max_seq_length=MAX_SEQ_LEN,
        remove_unused_columns=REMOVE_UNUSED_COLUMNS,
        optim=OPTIM,
        label_names=["labels"],
        ddp_find_unused_parameters=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        peft_config=peft_config,
        processing_class=processor.tokenizer,
        callbacks=[LossPrinterCallback, early_stop_cb], 
    )
    print("-"*30)
    print("Initial Evaluation")
    metric = trainer.evaluate()
    print(metric)
    print("-"*30)

    print("Starting training...")
    trainer.train()
    print("Training finished.")

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()