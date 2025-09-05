import os
import torch
import json
import hashlib
import requests
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from sklearn.model_selection import train_test_split
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from decord import VideoReader, cpu
from utils.vision_process import process_vision_info
from utils.custom_loss_raft import compute_raft_loss
from utils.custom_trainer import CustomSFTTrainer
import subprocess
import librosa
from tqdm import tqdm
from accelerate import Accelerator
from transformers import TrainerCallback, TrainerState, TrainerControl
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
                    "total_pixels": TOTAL_PIXELS,
                    "min_pixels": MIN_PIXELS,
                },
                {"type": "text", "text": user_prompt_filled},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"The score prediction is: {example["score"]:.1f}"}],
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

# def create_collate_fn(processor):
#     def collate_fn(examples):
#         texts = []
#         video_inputs = []
#         fps_list = []
#         # examples = [prune_messages(example["messages"]) for example in examples]  # 清理冗余字段
#         for i, example in enumerate(examples):
#             message_list = example
#             # 构造对话模板文本
#             text = processor.apply_chat_template(message_list, tokenize=False, add_generation_prompt=False)
#             texts.append(text)
#             # 处理视频
#             image_inputs, video_frames, video_kwargs = process_vision_info([message_list], return_video_kwargs=True)
           
#             video_inputs.append(video_frames)               # 累积帧列表
#             fps = video_kwargs.get('fps', None)             # 单个样本的 fps
#             fps_list.append(fps[0])                            # 收集到列表里
#             # print("len(texts):", len(texts))
#             print(i)
#             print("len(video_frames):", len(video_frames))
#             print("len(video_inputs):", len(video_inputs))
                            

#         print(len(texts), len(video_inputs), len(fps_list))
#         # 统一调用 processor，注意 videos 和 fps 要对应 samples 数量
#         batch = processor(
#             text=texts,
#             images=image_inputs,
#             videos=video_inputs,
#             fps=fps_list,
#             padding=True,
#             return_tensors="pt"
#         )
#         # 构造 labels
#         labels = batch["input_ids"].clone()
#         labels[labels == processor.tokenizer.pad_token_id] = -100
#         batch["labels"] = labels
#         print(batch.keys())
#         print(batch["input_ids"])
#         print(batch["labels"])
#         return batch

#     return collate_fn

# 配置参数集中管理（全部大写常量）
DEVICE = "cuda"
CACHE_WHISPER = "./.cache/whisper"
WHISPER_MODEL_NAME = "openai/whisper-medium"
CACHE_VLM = "./.cache/qwen2.5"
VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
CODING_CSV = "data/coding.csv"
OUTPUT_CSV = "fisclipt_scores.csv"
SYSTEM_PROMPT = "prompt/system_template.txt"
USER_PROMPT = "prompt/user_template.txt"
VIDEO_FOLDER = "data/FISclipped"
TOTAL_PIXELS = 20480*28*28
MIN_PIXELS = 16*28*28
MAX_SEQ_LEN = 512
CHUNK_LENGTH_S = 30
CHUNK_OVERLAP_S = 2

EPOCHS = 2
BATCH_SIZE = 2
GRADIENT_CHECKPOINTING = True  # Tradeoff between memory efficiency and computation time.
USE_REENTRANT = False
OPTIM = "paged_adamw_32bit"
LEARNING_RATE = 2e-5
LOGGING_STEPS = 50
EVAL_STEPS = 100
SAVE_STEPS = 100
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
    )
    processor = AutoProcessor.from_pretrained(vlm_model_name, cache_dir=cache_vlm, use_fast=True)
    processor.save_pretrained(cache_vlm)

    data_collator = create_collate_fn(processor)

    # dataset = load_from_disk("my_dataset")
    # splits = dataset.train_test_split(test_size=0.2, seed=42)
    # train_ds = splits["train"]
    # temp_ds = splits["test"].train_test_split(test_size=0.5, seed=42)
    # eval_ds = temp_ds["train"]
    # test_ds = temp_ds["test"]

    raw_dataset = load_dataset("CalistaLu/FIS-Full-Dataset", split="train")
    print(f"Raw dataset loaded with {len(raw_dataset)} examples.")

   
    # Load templates
    with open(USER_PROMPT, "r") as f:
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
    
    train_ds = [prune_messages(data["messages"]) for data in train_ds]
    eval_ds = [prune_messages(data["messages"]) for data in eval_ds]
    test_ds = [prune_messages(data["messages"]) for data in test_ds]
    print("First message in dataset:", train_ds[0])

    



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
        greater_is_better=None,  # eval_loss 越小越好
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

    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        peft_config=peft_config,
        compute_loss_func=compute_raft_loss,
        processing_class=processor.tokenizer,
        callbacks=[LossPrinterCallback, early_stop_cb], 
    )


    print("Starting training...")
    trainer.train()
    print("Training finished.")

    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    main()