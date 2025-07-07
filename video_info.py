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

from vision_process import process_vision_info, IMAGE_FACTOR
from datasets import load_from_disk

# 载入数据集并取出第一条对话
dataset = load_from_disk("my_dataset")
print(len(dataset))
conversations = dataset[0]["messages"]

# 处理视觉信息，拿到视频张量
images, videos = process_vision_info(conversations)
video = videos[0]              # torch.Tensor, shape (T, C, H, W)

# 解包维度
T, C, H, W = video.shape

# 补丁大小：IMAGE_FACTOR × IMAGE_FACTOR
patch_size = IMAGE_FACTOR       # 28

# 每帧的 patch 行列数
num_patches_per_row = H // patch_size
num_patches_per_col = W // patch_size
total_patches = num_patches_per_row * num_patches_per_col

print(f"Video tensor shape: {video.shape}")
print(f"Patch size: {patch_size}×{patch_size}")
print(f"Patches per frame: {num_patches_per_row}×{num_patches_per_col} = {total_patches}")

