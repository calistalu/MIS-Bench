import torch
from absl.testing import absltest, parameterized
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # 假设已安装 qwen_llm 包
from PIL import Image

class QwenEmbeddingTest(absltest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
        cls.cache_dir = "./.cache/qwen2.5"
        # 加载模型和 Processor
        cls.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cls.model_name,
            cache_dir=cls.cache_dir,
            torch_dtype=torch.float16
        ).to("cuda")
        cls.processor = AutoProcessor.from_pretrained(
            cls.model_name,
            cache_dir=cls.cache_dir
        )

    def test_text_embedding(self):
        texts = ["今天天气很好。", "测试多模态 embedding。"]
        # 仅文本输入
        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to("cuda")

        # 前向并拿到 encoder 输出隐藏态
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        # encoder_last_hidden_state: [B, T, D]
        enc_states = outputs.encoder_last_hidden_state  # torch.Tensor
        # 简单 pooling（平均）得到 [B, D]
        embeddings = enc_states.mean(dim=1)

        # 检查类型和形状
        self.assertIsInstance(embeddings, torch.Tensor)
        batch_size, hidden_size = embeddings.shape
        self.assertEqual(batch_size, len(texts))
        self.assertEqual(hidden_size, self.model.config.hidden_size)
        self.assertEqual(embeddings.dtype, torch.float16)

    def test_multimodal_embedding(self):
        texts = ["带图像的示例。"]
        # 构造一张 224x224 白色图片
        img = Image.new("RGB", (224, 224), color="white")

        inputs = self.processor(
            text=texts,
            images=[img],
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to("cuda")

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
        enc_states = outputs.encoder_last_hidden_state  # [B, T, D]
        embeddings = enc_states.mean(dim=1)

        # 检查类型和形状
        self.assertIsInstance(embeddings, torch.Tensor)
        batch_size, hidden_size = embeddings.shape
        self.assertEqual(batch_size, 1)
        self.assertEqual(hidden_size, self.model.config.hidden_size)
        self.assertEqual(embeddings.dtype, torch.float16)

if __name__ == '__main__':
    absltest.main()
