from transformers import AutoProcessor

# 加载 Qwen2.5-VL 处理器
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct"
)

# 获取 tokenizer
tokenizer = processor.tokenizer

# 获取特定文本的 token id
# text = "2.1"
# token_ids = tokenizer.encode(text, add_special_tokens=True)

token_id1 = 374
token_id2 = 220  # 假设 13 是小数点的 token ID
token_id3 = 18
text1 = tokenizer.decode(token_id1)
text2 = tokenizer.decode(token_id2)
text3 = tokenizer.decode(token_id3)

# 打印 token 和对应的 ID
print(f"Text: {text1}, Token ID: {token_id1}")
print(f"Text: {text2}, Token ID: {token_id2}")
print(f"Text: {text3}, Token ID: {token_id3}")
