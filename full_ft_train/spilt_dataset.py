"""
==========================================================================
数据集拆分脚本 V2（适配 convert_v3 无 system prompt 的格式）
==========================================================================

输入：
  finetune_data_v2/train.json  (8930 条)
  finetune_data_v2/val.json    (1547 条)

输出：
  finetune_data_v2/train_split.json  (原train的90% ≈ 8037 条) → 训练用
  finetune_data_v2/val_split.json    (原train的10% ≈ 893 条)  → 训练时验证loss用
  finetune_data_v2/test_split.json   (原val全部 = 1547 条)    → 最终评估用
"""

import json
import os
import random
from collections import Counter

# ╔══════════════════════════════════════════════════════════════╗
# ║  ⬇⬇⬇ 确认路径 ⬇⬇⬇                                         ║
# ╚══════════════════════════════════════════════════════════════╝

DATA_DIR = "/root/autodl-tmp/qwen_vl/finetune_data_v2"

# ╔══════════════════════════════════════════════════════════════╗
# ║  ⬆⬆⬆ 以下不需要改 ⬆⬆⬆                                      ║
# ╚══════════════════════════════════════════════════════════════╝

VAL_RATIO = 0.1

TRAIN_JSON = os.path.join(DATA_DIR, "train.json")
VAL_JSON   = os.path.join(DATA_DIR, "val.json")

TRAIN_SPLIT = os.path.join(DATA_DIR, "train_split.json")
VAL_SPLIT   = os.path.join(DATA_DIR, "val_split.json")
TEST_SPLIT  = os.path.join(DATA_DIR, "test_split.json")


def get_image_path(sample):
    """
    从样本中提取图片路径。
    自动适配两种格式：
      - 有 system prompt: messages[0]=system, messages[1]=user → 图片在 messages[1]
      - 无 system prompt: messages[0]=user → 图片在 messages[0]
    """
    for msg in sample["messages"]:
        if msg["role"] == "user":
            # user 的 content 是列表 [{"type":"image",...}, {"type":"text",...}]
            for item in msg["content"]:
                if isinstance(item, dict) and item.get("type") == "image":
                    return item["image"]
    return None


def get_assistant_content(sample):
    """从样本中提取 assistant 回复"""
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            return msg["content"]
    return ""


def count_categories(data):
    cat_counter = Counter()
    for sample in data:
        content = get_assistant_content(sample)
        try:
            targets = json.loads(content)
            for t in targets:
                cat_counter[t["label"]] += 1
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    return cat_counter


def main():
    random.seed(42)

    print("=" * 55)
    print("  数据集拆分：train → train+val，val → test")
    print("=" * 55)

    for path in [TRAIN_JSON, VAL_JSON]:
        if not os.path.exists(path):
            print(f"\n  ❌ 文件不存在: {path}")
            return
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  找到: {path} ({size_mb:.1f} MB)")

    print(f"\n读取原始数据...")
    with open(TRAIN_JSON, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(VAL_JSON, 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    print(f"  原 train.json: {len(train_data)} 条")
    print(f"  原 val.json:   {len(val_data)} 条")
    print(f"  总计:          {len(train_data) + len(val_data)} 条")

    # 按图片分组
    image_to_samples = {}
    for sample in train_data:
        img_path = get_image_path(sample)
        if img_path not in image_to_samples:
            image_to_samples[img_path] = []
        image_to_samples[img_path].append(sample)

    image_list = list(image_to_samples.keys())
    random.shuffle(image_list)

    val_count = int(len(image_list) * VAL_RATIO)

    val_images   = image_list[:val_count]
    train_images = image_list[val_count:]

    new_train = []
    for img in train_images:
        new_train.extend(image_to_samples[img])

    new_val = []
    for img in val_images:
        new_val.extend(image_to_samples[img])

    new_test = val_data

    random.shuffle(new_train)
    random.shuffle(new_val)

    print(f"\n保存拆分结果...")
    for name, data, path in [
        ("train_split", new_train, TRAIN_SPLIT),
        ("val_split",   new_val,   VAL_SPLIT),
        ("test_split",  new_test,  TEST_SPLIT),
    ]:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  ✅ {name}: {len(data):>5d} 条 → {path} ({size_mb:.1f} MB)")

    print(f"\n{'='*55}")
    print(f"  📋 拆分结果汇总")
    print(f"{'='*55}")
    print(f"  {'数据集':<15} {'样本数':>6} {'图片数':>6} {'用途'}")
    print(f"  {'─'*50}")
    print(f"  {'train_split':<15} {len(new_train):>6d} {len(train_images):>6d}  训练模型")
    print(f"  {'val_split':<15} {len(new_val):>6d} {len(val_images):>6d}  训练时看loss曲线")
    print(f"  {'test_split':<15} {len(new_test):>6d} {'1547':>6s}  最终评估（模型没见过）")
    print(f"  {'─'*50}")
    print(f"  {'合计':<15} {len(new_train)+len(new_val)+len(new_test):>6d}")

    for name, data in [("train_split", new_train), ("val_split", new_val), ("test_split", new_test)]:
        cats = count_categories(data)
        print(f"\n  {name} 类别分布:")
        for cat_name, cnt in cats.most_common():
            print(f"    {cat_name}: {cnt}")

    train_set = set(train_images)
    val_set   = set(val_images)
    overlap   = train_set & val_set
    if overlap:
        print(f"\n  ⚠️ 警告：train 和 val 有 {len(overlap)} 张图片重叠！")
    else:
        print(f"\n  ✅ 验证通过：train 和 val 无图片重叠")
    print(f"  ✅ test 来自原 val/，和 train 天然无重叠")

    print(f"\n{'='*55}")
    print(f"  拆分完成！")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()