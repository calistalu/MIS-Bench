from transformers import AutoModel
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# 1️⃣ 加载基础模型（不用量化方便测试）
base_model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_name, cache_dir="./.cache/qwen2.5")

# 2️⃣ 加载已有 LoRA adapter
pretrained_lora_path = "output/checkpoint-1992"  # 包含 adapter_model.safetensors 和 config.json
model = PeftModel.from_pretrained(model, pretrained_lora_path)
# 手动激活 LoRA 参数
for name, param in model.named_parameters():
    # 只激活 LoRA 参数（通常名字包含 lora）
    if "lora" in name:
        param.requires_grad = True
# 3️⃣ 确保模型切换到训练模式
model.train()

# 4️⃣ 打印可训练参数统计
def print_lora_trainable_params(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100*trainable_params/all_params:.6f}")

print_lora_trainable_params(model)

# 5️⃣ 可选：打印 LoRA target modules 中的参数名，确认加载正确
print("LoRA trainable parameter names:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
