import argparse
import json
import os
import re

import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from utils.vision_process import process_vision_info


CACHE_VLM = "./.cache/qwen2.5"
VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
USER_PROMPT = "prompt/user_template.txt"
ADAPTER_PATH = "./output-finetune"
OUTPUT_CSV = "finetune.csv"
DATASET_NAME = "CalistaLu/FIS-Full-Dataset"
DATASET_SPLIT = "test"
TOTAL_PIXELS = 20480 * 28 * 28
MIN_PIXELS = 16 * 28 * 28
MAX_NEW_TOKENS = 16


def format_messages(example, user_template):
    user_prompt = user_template.format(
        orig_instruction=example["orig_instruction"],
        orig_response=example["orig_response"],
        orig_criteria=example["orig_criteria"],
        orig_score1_description=example["orig_score1_description"],
        orig_score2_description=example["orig_score2_description"],
        orig_score3_description=example["orig_score3_description"],
        orig_score4_description=example["orig_score4_description"],
        orig_score5_description=example["orig_score5_description"],
    )

    return [
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
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def parse_score(text):
    clean = text.strip()
    try:
        data = json.loads(clean)
        if isinstance(data, dict):
            for key in ("score", "Score"):
                if key in data:
                    return float(data[key])
    except json.JSONDecodeError:
        pass

    match = re.search(r"[-+]?\d+(?:\.\d+)?", clean)
    if not match:
        return None

    score = float(match.group(0))
    if score < 1 or score > 5:
        return None
    return round(score, 1)


def normalize_video_name(video_path):
    root, ext = os.path.splitext(video_path)
    return root if ext.lower() == ".mp4" else video_path


@torch.inference_mode()
def inference_once(model, processor, messages):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages],
        return_video_kwargs=True,
    )
    fps = video_kwargs["fps"]

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        fps=fps,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    generated = outputs[0][inputs.input_ids.shape[-1] :]
    result = processor.batch_decode(
        [generated],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    del image_inputs, video_inputs, inputs, outputs, generated
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=DATASET_NAME)
    parser.add_argument("--split", default=DATASET_SPLIT)
    parser.add_argument("--adapter_path", default=ADAPTER_PATH)
    parser.add_argument("--output_csv", default=OUTPUT_CSV)
    parser.add_argument("--user_prompt", default=USER_PROMPT)
    args = parser.parse_args()

    if not os.path.exists(args.adapter_path):
        raise FileNotFoundError(
            f"Adapter path not found: {args.adapter_path}. "
            "Run pipeline_finetune.py first or pass --adapter_path."
        )

    with open(args.user_prompt, "r", encoding="utf-8") as f:
        user_template = f.read().strip()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLM_MODEL_NAME,
        cache_dir=CACHE_VLM,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        VLM_MODEL_NAME,
        cache_dir=CACHE_VLM,
        use_fast=True,
    )

    dataset = load_dataset(args.dataset_name, split=args.split)
    rows_by_video = {}
    raw_rows = []

    for example in tqdm(dataset, desc=f"Inferencing {args.split} split"):
        messages = format_messages(example, user_template)
        response = inference_once(model, processor, messages)
        score = parse_score(response)

        video = normalize_video_name(example["video_path"])
        criterion = example["orig_criteria"]
        rows_by_video.setdefault(video, {"video": video})[criterion] = score
        raw_rows.append(
            {
                "video": video,
                "criterion": criterion,
                "prediction": score,
                "raw_response": response,
            }
        )

    df = pd.DataFrame(rows_by_video.values())
    df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    raw_output = os.path.splitext(args.output_csv)[0] + "_raw.csv"
    pd.DataFrame(raw_rows).to_csv(raw_output, index=False, encoding="utf-8-sig")
    print(f"Saved scores to {args.output_csv}")
    print(f"Saved raw responses to {raw_output}")


if __name__ == "__main__":
    main()
