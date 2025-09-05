# from vision_process import process_vision_info, IMAGE_FACTOR
# from datasets import Dataset, DatasetDict, load_from_disk
# # 假设你已经构造了一个 conversations，其中包含一个视频 ele 字典
# dataset = load_from_disk("my_dataset")
# print(len(dataset))
# conversations = dataset[0]["messages"]
# images, videos = process_vision_info(conversations)
# video = videos[0]              # 取第一个视频，类型是 torch.Tensor
# T, C, H, W = video.shape
# scale = 2
# patch_size = IMAGE_FACTOR // scale  # 14

# num_patches_per_row = H // patch_size
# num_patches_per_col = W // patch_size
# total_patches = num_patches_per_row * num_patches_per_col

# print(f"Video tensor shape: {video.shape}")
# print(f"Patch size: {patch_size}×{patch_size}")
# print(f"Patches per frame: {num_patches_per_row}×{num_patches_per_col} = {total_patches}")

from utils.vision_process import process_vision_info, IMAGE_FACTOR
from datasets import load_from_disk
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


# 载入数据集并取出第一条对话
dataset = load_from_disk("my_dataset")
conversations = prune_messages(dataset[11]["messages"])
# 处理视觉信息，拿到视频张量
images, videos = process_vision_info(conversations)
video = videos[0]              # torch.Tensor, shape (T, C, H, W)

# 解包维度
T, C, H, W = video.shape
print(T, C, H, W)
# 补丁大小：IMAGE_FACTOR × IMAGE_FACTOR
patch_size = IMAGE_FACTOR       # 28
print(f"Patch size: {patch_size}×{patch_size}")
# 每帧的 patch 行列数
num_patches_per_row = H // patch_size
num_patches_per_col = W // patch_size
total_patches = num_patches_per_row * num_patches_per_col

print(f"Video tensor shape: {video.shape}")
print(f"Patch size: {patch_size}×{patch_size}")
print(f"Patches per frame: {num_patches_per_row}×{num_patches_per_col} = {total_patches}")

