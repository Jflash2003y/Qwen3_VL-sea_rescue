"""
将 LoRA 权重合并到基础模型，导出完整模型供 vLLM 使用
"""
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

MODEL_PATH = '/root/autodl-tmp/qwen_vl/models/Qwen/Qwen3-VL-8B-Instruct'
LORA_PATH = '/root/autodl-tmp/qwen_vl/lora_output_v2/best_model'
OUTPUT_PATH = '/root/autodl-tmp/qwen_vl/models/Qwen3-VL-8B-Instruct-merged-v2'

print("1. 加载基础模型...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cpu"
)

print("2. 加载 LoRA 权重...")
model = PeftModel.from_pretrained(model, LORA_PATH)

print("3. 合并 LoRA...")
model = model.merge_and_unload()

print("4. 保存合并后的模型...")
model.save_pretrained(OUTPUT_PATH)

print("5. 复制 processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)
processor.save_pretrained(OUTPUT_PATH)

print(f"\n✅ 合并完成！保存到: {OUTPUT_PATH}")