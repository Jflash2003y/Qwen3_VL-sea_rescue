"""
==========================================================================
零样本测试：看 Qwen3-VL-8B 原模型面对我们的 prompt 会输出什么格式
==========================================================================
用途：确认原模型输出格式，再决定训练数据是否需要调整
"""

import json
import os
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ╔══════════════════════════════════════════════════════════════╗
# ║  ⬇⬇⬇ 检查这两个路径 ⬇⬇⬇                                   ║
# ╚══════════════════════════════════════════════════════════════╝

MODEL_PATH = "/root/autodl-tmp/qwen_vl/models/Qwen/Qwen3-VL-8B-Instruct"
DATA_BASE  = "/root/autodl-tmp/qwen_vl/finetune_data_v2"

# ╔══════════════════════════════════════════════════════════════╗
# ║  ⬆⬆⬆ 以下不需要改 ⬆⬆⬆                                      ║
# ╚══════════════════════════════════════════════════════════════╝

# 和训练数据完全一致的 system prompt 和 user prompt
SYSTEM_PROMPT = (
    "你是一个海上搜救无人机视觉分析系统。"
    "你的任务是分析无人机航拍图像，"
    "检测水面上的所有目标（水中人员、船只、水上摩托、救生设备、浮标），"
    "并以JSON格式返回每个目标的类别和归一化边界框坐标。"
    "坐标格式为 [x1, y1, x2, y2]，数值范围 0-1000。"
)

USER_PROMPT = "请检测这张无人机航拍图中水面上的所有目标，返回每个目标的类别和边界框坐标。"

NUM_TEST = 5  # 测试几张图


def main():
    print("=" * 60)
    print("  零样本测试：查看原模型输出格式")
    print("=" * 60)

    # 加载模型
    print(f"\n加载模型: {MODEL_PATH}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    print("模型加载完成！\n")

    # 从 test_split.json 取几张图（模型完全没见过的）
    test_json = os.path.join(DATA_BASE, "test_split.json")
    with open(test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 取不同图片
    tested_images = set()
    test_samples = []
    for sample in test_data:
        img_path = sample["messages"][1]["content"][0]["image"]
        if img_path not in tested_images and len(test_samples) < NUM_TEST:
            tested_images.add(img_path)
            test_samples.append(sample)

    # 逐张测试
    for i, sample in enumerate(test_samples):
        img_relative = sample["messages"][1]["content"][0]["image"]
        img_absolute = os.path.join(DATA_BASE, img_relative)
        ground_truth = sample["messages"][2]["content"]  # JSON字符串

        print(f"{'='*60}")
        print(f"测试 {i+1}/{NUM_TEST}：{img_relative}")
        print(f"{'='*60}")

        # ---- 测试1：带 system prompt（和训练数据一致的 prompt）----
        messages_with_system = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_absolute}"},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]

        response_1 = run_inference(model, processor, messages_with_system)
        print(f"\n【带 system prompt 的回答】：")
        print(response_1)

        # ---- 测试2：不带 system prompt（纯裸问）----
        messages_bare = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_absolute}"},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]

        response_2 = run_inference(model, processor, messages_bare)
        print(f"\n【不带 system prompt 的回答】：")
        print(response_2)

        # ---- GT ----
        try:
            gt_targets = json.loads(ground_truth)
            print(f"\n【正确答案 (GT)】：{len(gt_targets)} 个目标")
            for t in gt_targets[:5]:
                print(f"  {t['label']}: {t['bbox_2d']}")
            if len(gt_targets) > 5:
                print(f"  ... 还有 {len(gt_targets)-5} 个")
        except json.JSONDecodeError:
            print(f"\n【正确答案】：{ground_truth[:200]}")

        print()

    print("=" * 60)
    print("零样本测试完成！")
    print("请观察模型输出的格式，然后告诉我结果。")
    print("我会根据原模型的输出格式来调整训练数据。")
    print("=" * 60)


def run_inference(model, processor, messages):
    """对一组 messages 做推理，返回模型回复文本"""
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=1024)

    output_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return response


if __name__ == "__main__":
    main()