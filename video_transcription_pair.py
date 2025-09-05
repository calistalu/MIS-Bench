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
import csv

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
                          chunk_overlap_s=1):
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
                                      skip_special_tokens=True, normalize=False)[0]
        print(text)
        transcripts.append(text.strip())
        del inputs, ids
        torch.cuda.empty_cache()

        # 滑动窗口：下一个 chunk 开始点
        start += chunk_size - overlap

    # 拼接所有片段，去掉重复
    full_transcript = " ".join(transcripts)
    return full_transcript

# 配置参数集中管理（全部大写常量）
DEVICE = "cuda:0"
CACHE_WHISPER = "./.cache/whisper"
WHISPER_MODEL_NAME = "openai/whisper-large-v2"
CACHE_VLM = "./.cache/qwen2.5"
VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
CODING_CSV = "coding.csv"
OUTPUT_CSV = "fis_transcript.csv"
SYSTEM_PROMPT = "system_prompt.txt"
USER_PROMPT = "user_prompt.txt"
VIDEO_FOLDER = "data/FIS996"
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
    whisper_model.config.forced_decoder_ids = whisper_processor.get_decoder_prompt_ids(language="en", task="transcribe")

    video_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.mp4')]
        
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['video', 'transcript'])  # 写入表头

        total = len(video_paths)
        print(f"Total videos to process: {total}")
        # 遍历视频文件并生成转录内容
        for idx, vid in enumerate(tqdm(sorted(video_paths)), start=1):
            print(f"Processing {vid}...")
            # 3.1 转录音频
            audio_file = extract_audio(vid)
            transcript = transcribe_long_audio(
                audio_file, whisper_model, whisper_processor,
                chunk_length_s=CHUNK_LENGTH_S, chunk_overlap_s=CHUNK_OVERLAP_S)
            
            # 获取视频文件名，不带 mp4 后缀
            video_name = os.path.splitext(os.path.basename(vid))[0]

            # 写入视频名和转录内容
            writer.writerow([video_name, transcript])

    print(f"Transcriptions saved to {output_csv}")

if __name__ == '__main__':
    main()
