"""
==========================================================================
SeaDroneSee COCO → Qwen3-VL 全参数微调数据格式转换 V3
==========================================================================

V3 相比 V2 的改动（基于零样本测试结果）：
  1. 去掉 system prompt（原模型不带 system 时输出格式更稳定）
  2. JSON 键名顺序改为 bbox_2d 在前（和原模型输出一致）

零样本测试发现原模型输出格式：
  [{"bbox_2d": [x1,y1,x2,y2], "label": "..."}, ...]
  - 坐标已经是 0-1000 归一化 ✅
  - 不带 system prompt 时格式最稳定 ✅
  - 键名 bbox_2d 在前，label 在后 ✅
"""

import json
import os
import random
from collections import defaultdict, Counter

# ╔══════════════════════════════════════════════════════════════╗
# ║  ⬇⬇⬇ 你需要修改的只有这 2 行路径 ⬇⬇⬇                      ║
# ╚══════════════════════════════════════════════════════════════╝

BASE_DIR = r'E:\Graduation Project\dataset\seadronessee Objection V2\Compressed Version\Compressed Version'
OUTPUT_DIR = r'E:\Graduation Project\dataset\switch\output_v3'

# ╔══════════════════════════════════════════════════════════════╗
# ║  ⬆⬆⬆ 以下不需要改 ⬆⬆⬆                                      ║
# ╚══════════════════════════════════════════════════════════════╝

TRAIN_JSON = os.path.join(BASE_DIR, 'annotations', 'instances_train.json')
VAL_JSON   = os.path.join(BASE_DIR, 'annotations', 'instances_val.json')

os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, 'train.json')
VAL_OUTPUT   = os.path.join(OUTPUT_DIR, 'val.json')

# 类别映射
CATEGORY_NAMES = {
    1: '水中人员',
    2: '船只',
    3: '水上摩托',
    4: '救生设备',
    5: '浮标',
}

# 统一的用户提问（和零样本测试用同一句，方便对比）
USER_PROMPT = "请检测这张无人机航拍图中水面上的所有目标，返回每个目标的类别和边界框坐标。"


def bbox_coco_to_normalized(bbox, img_width, img_height):
    x, y, w, h = bbox
    x1 = max(0, min(1000, round(x / img_width  * 1000)))
    y1 = max(0, min(1000, round(y / img_height * 1000)))
    x2 = max(0, min(1000, round((x + w) / img_width  * 1000)))
    y2 = max(0, min(1000, round((y + h) / img_height * 1000)))
    return [x1, y1, x2, y2]


def generate_response(annotations, img_width, img_height):
    """
    生成 JSON 回复，键名顺序和原模型输出一致：
    [{"bbox_2d": [x1,y1,x2,y2], "label": "水中人员"}, ...]
    """
    targets = []
    for ann in annotations:
        cat_name = CATEGORY_NAMES.get(ann['category_id'])
        if cat_name is None:
            continue
        bbox_norm = bbox_coco_to_normalized(ann['bbox'], img_width, img_height)
        # ★ bbox_2d 在前，label 在后（和原模型输出顺序一致）
        targets.append({
            "bbox_2d": bbox_norm,
            "label": cat_name,
        })

    targets.sort(key=lambda t: (t["bbox_2d"][1], t["bbox_2d"][0]))
    return json.dumps(targets, ensure_ascii=False, separators=(',', ':'))


def create_sample(image_info, annotations, image_folder):
    """
    生成一条训练样本。
    ★ V3 改动：不再包含 system message，只有 user + assistant
    """
    img_width  = image_info['width']
    img_height = image_info['height']
    file_name  = image_info['file_name']
    image_path = f"{image_folder}/{file_name}"

    response = generate_response(annotations, img_width, img_height)

    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text",  "text": USER_PROMPT},
                ],
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]
    }


def convert_coco_to_qwen(coco_json_path, image_folder, output_path):
    print(f"\n{'─'*55}")
    print(f"读取: {coco_json_path}")

    with open(coco_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    images_dict = {img['id']: img for img in data['images']}

    annotations_by_image = defaultdict(list)
    total_ann = 0
    for ann in data['annotations']:
        if ann['category_id'] in CATEGORY_NAMES:
            annotations_by_image[ann['image_id']].append(ann)
            total_ann += 1

    print(f"  图片总数:       {len(images_dict)}")
    print(f"  有效标注总数:   {total_ann}")
    print(f"  有标注的图片数: {len(annotations_by_image)}")

    cat_counter = Counter()
    for anns in annotations_by_image.values():
        for ann in anns:
            cat_counter[CATEGORY_NAMES[ann['category_id']]] += 1
    print(f"  类别分布:")
    for name, count in cat_counter.most_common():
        print(f"    {name}: {count}")

    all_samples = []
    skipped = 0

    for image_id, img_info in images_dict.items():
        anns = annotations_by_image.get(image_id, [])
        if len(anns) == 0:
            skipped += 1
            continue
        all_samples.append(create_sample(img_info, anns, image_folder))

    random.shuffle(all_samples)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"  跳过无标注图片: {skipped}")
    print(f"  生成样本数:     {len(all_samples)}")
    print(f"  输出文件:       {output_path}")
    print(f"  文件大小:       {file_size:.1f} MB")

    return len(all_samples)


def verify_output(output_path, n=3):
    with open(output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n{'='*55}")
    print(f"验证: {os.path.basename(output_path)} (共 {len(data)} 条，展示 {n} 条)")
    print(f"{'='*55}")

    for i, sample in enumerate(data[:n]):
        print(f"\n--- 样本 {i+1} ---")
        for msg in sample['messages']:
            role = msg['role']
            if role == 'user':
                for item in msg['content']:
                    if item.get('type') == 'image':
                        print(f"  [User]      图片: {item['image']}")
                    else:
                        print(f"  [User]      提问: {item['text']}")
            elif role == 'assistant':
                try:
                    targets = json.loads(msg['content'])
                    print(f"  [Assistant] {len(targets)} 个目标:")
                    for t in targets[:3]:
                        print(f"              bbox_2d={t['bbox_2d']}  label={t['label']}")
                    if len(targets) > 3:
                        print(f"              ... 还有 {len(targets)-3} 个")
                except json.JSONDecodeError:
                    print(f"  [Assistant] {msg['content'][:80]}")


if __name__ == '__main__':
    random.seed(42)

    print("=" * 55)
    print("  SeaDroneSee → Qwen3-VL 微调数据转换 V3")
    print("  (无 system prompt，bbox_2d 在前)")
    print("=" * 55)

    for path in [TRAIN_JSON, VAL_JSON]:
        if not os.path.exists(path):
            print(f"\n  ❌ 文件不存在: {path}")
            exit(1)

    train_count = convert_coco_to_qwen(TRAIN_JSON, 'train', TRAIN_OUTPUT)
    val_count   = convert_coco_to_qwen(VAL_JSON,   'val',   VAL_OUTPUT)

    print(f"\n{'='*55}")
    print(f"  ✅ 全部完成！")
    print(f"  训练集: {train_count} 条 → {TRAIN_OUTPUT}")
    print(f"  验证集: {val_count} 条  → {VAL_OUTPUT}")
    print(f"{'='*55}")

    verify_output(TRAIN_OUTPUT, n=3)
    verify_output(VAL_OUTPUT,   n=2)