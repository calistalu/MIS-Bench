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


def inference_once(model, processor, video_file, prompt, total_pixels, min_pixels):
    path, frames, timestamps = get_video_frames(video_file)
    path = path.replace('FISclipped', 'data/FIS996')
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "video", "video": path, "total_pixels": total_pixels, "min_pixels": min_pixels}
        ]}
    ]
    print(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps = video_kwargs['fps']

    print("len(video_frames):", len(video_inputs))
    print("len(video_inputs):", len(video_inputs))
    # print("len(fps_list):", len(fps_list))

    
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
                    

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=fps,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    outs = model.generate(**inputs, max_new_tokens=2048)
    gen = outs[0][inputs.input_ids.shape[-1]:]
    result = processor.batch_decode([gen], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

    # 只删除本函数用到的临时 Tensor
    del image_inputs, video_inputs, inputs, outs, gen
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return result


def main(folder='FISclipped', output_csv='data/fisclipt_scores.csv', prompt_file='prompt/prompt.txt'):
    device = "cuda:0"
    # 1) 加载并缓存 Whisper 模型一次
    cache_whisper = './.cache/whisper'
    whisper_processor = WhisperProcessor.from_pretrained(
        "openai/whisper-medium", cache_dir=cache_whisper)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        "openai/whisper-medium", cache_dir=cache_whisper,
        torch_dtype=torch.float16, device_map={"": 0}).to(device)

    # 2) 加载并缓存 VLM 模型一次
    cache_vlm = './.cache/qwen2.5'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_vlm,
        torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", cache_dir=cache_vlm)

    # 3) 读取 prompt 模板
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_template = f.read().strip()

    rows = []
    video_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.mp4')]

    # for idx, vid in enumerate(tqdm(sorted(video_paths)[200:]), start=201):
    for idx, vid in enumerate(tqdm(sorted(video_paths)), start=1):
        print(f"Processing {vid}...")
        # 3.1 转录音频
        audio_file = extract_audio(vid)
        transcript = transcribe_long_audio(
            audio_file, whisper_model, whisper_processor,
            chunk_length_s=30, chunk_overlap_s=2)
        # print(f"Transcript: {transcript[:60]}...")

        # 3.2 格式化 prompt 并调用多模态推理
        video_name = os.path.basename(vid)
        # prompt = prompt_template.format(video=video_name, transcript=transcript)
        prompt = prompt_template.replace("{video}", video_name).replace("{transcript}", transcript)
        resp = inference_once(
            model, processor, vid, prompt,
            total_pixels=20480*28*28, min_pixels=16*28*28)

        try:
            # 1. 先清洗 Markdown 代码块标记
            clean = resp.strip()
            if clean.startswith("```"):
                import re
                clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean)

            # 2. 再解析 JSON
            data = json.loads(clean)


            # 3. 构造 entry
            entry = {'video': data.get('video', video_name)}
            entry.update(data.get('scores', {}))
            entry['explanation'] = data.get('explanation', "")
            entry['suggestion'] = data.get('suggestion', "")
            rows.append(entry)

        except json.JSONDecodeError:
            print(f"Failed to parse JSON for {vid}. Cleaned string was:\n{clean}\nOriginal resp was:\n{resp}")

        del resp
    
        if idx % 50 == 0:
            df_partial = pd.DataFrame(rows)
            df_partial.to_csv(output_csv, index=False)
            print(f"Saved interim results after {idx} videos to {output_csv}")

        import gc
        gc.collect()
        torch.cuda.empty_cache()


    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == '__main__':
    main()
