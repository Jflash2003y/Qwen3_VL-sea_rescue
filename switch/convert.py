import json
import os
import random
from collections import defaultdict

# ============================================================
# 路径配置（根据你的本地目录结构）
# ============================================================
BASE_DIR = r'E:\Graduation Project\dataset\seadronessee Objection V2\Compressed Version\Compressed Version'
OUTPUT_DIR = r'E:\Graduation Project\dataset\switch\output'

# 输入
TRAIN_JSON = os.path.join(BASE_DIR, 'annotations', 'instances_train.json')
VAL_JSON = os.path.join(BASE_DIR, 'annotations', 'instances_val.json')

# 输出
os.makedirs(OUTPUT_DIR, exist_ok=True)
TRAIN_OUTPUT = os.path.join(OUTPUT_DIR, 'train.json')
VAL_OUTPUT = os.path.join(OUTPUT_DIR, 'val.json')

# ============================================================
# 类别名称映射（中文）
# ============================================================
CATEGORY_NAMES = {
    0: '忽略',
    1: '水中人员',
    2: '船只',
    3: '水上摩托',
    4: '救生设备',
    5: '浮标'
}

# ============================================================
# 提问模板（随机选择，增加数据多样性）
# ============================================================
DETECTION_PROMPTS = [
    "请检测这张无人机航拍图中水面上的所有目标，返回每个目标的类别和位置坐标。",
    "这是一张海上搜救无人机拍摄的图片，请找出画面中所有可见目标并标出位置。",
    "请仔细观察这张航拍图片，识别出水面上的人员、船只和其他物体，并给出它们的坐标位置。",
    "作为海上搜救系统，请分析这张无人机图片，检测所有目标并报告类别与坐标。",
    "请对这张无人机航拍的海面图片进行目标检测，列出所有发现的目标及其位置。",
]

RESCUE_PROMPTS = [
    "请判断这张无人机航拍图中是否有人处于危险状态？请检测所有人员位置并分析情况。",
    "作为搜救专家，请分析这张图片，找出水中是否有需要救援的人员，并标出位置。",
    "请观察这张海上航拍图片，重点关注是否有落水人员，给出人员位置和你的判断。",
]

COUNT_PROMPTS = [
    "请统计这张无人机航拍图中有多少人、多少艘船？请分别标出它们的位置。",
    "请清点这张海面航拍图中的所有目标数量，并给出每个目标的类别和坐标。",
]


# ============================================================
# 坐标转换函数
# ============================================================
def bbox_to_normalized(bbox, img_width, img_height):
    """
    将 COCO bbox [x, y, w, h] 转换为 Qwen3-VL 归一化坐标 (x1, y1, x2, y2)
    归一化到 0-1000 范围
    """
    x, y, w, h = bbox
    x1 = round(x / img_width * 1000)
    y1 = round(y / img_height * 1000)
    x2 = round((x + w) / img_width * 1000)
    y2 = round((y + h) / img_height * 1000)

    # 确保坐标在 0-1000 范围内
    x1 = max(0, min(1000, x1))
    y1 = max(0, min(1000, y1))
    x2 = max(0, min(1000, x2))
    y2 = max(0, min(1000, y2))

    return (x1, y1, x2, y2)


# ============================================================
# 生成助手回复
# ============================================================
def generate_detection_response(annotations, img_width, img_height):
    """生成目标检测类的回复"""
    targets = []
    for ann in annotations:
        cat_name = CATEGORY_NAMES.get(ann['category_id'], '未知')
        coords = bbox_to_normalized(ann['bbox'], img_width, img_height)
        targets.append((cat_name, coords))

    # 构建回复文本
    lines = ["检测到以下目标："]
    for i, (name, coords) in enumerate(targets, 1):
        lines.append(f"{i}. {name}：({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})")

    # 添加统计摘要
    from collections import Counter
    cat_counter = Counter(name for name, _ in targets)
    summary_parts = [f"{count}{'名' if name == '水中人员' else '艘' if name in ['船只', '水上摩托'] else '个'}{name}"
                     for name, count in cat_counter.items()]
    lines.append(f"\n共发现{'，'.join(summary_parts)}。")

    return "\n".join(lines)


def generate_rescue_response(annotations, img_width, img_height):
    """生成搜救判断类的回复"""
    swimmers = [ann for ann in annotations if ann['category_id'] == 1]
    all_targets = []
    for ann in annotations:
        cat_name = CATEGORY_NAMES.get(ann['category_id'], '未知')
        coords = bbox_to_normalized(ann['bbox'], img_width, img_height)
        all_targets.append((cat_name, coords))

    if swimmers:
        lines = [f"⚠️ 发现{len(swimmers)}名水中人员，可能需要救援！\n"]
        lines.append("人员位置：")
        for i, ann in enumerate(swimmers, 1):
            coords = bbox_to_normalized(ann['bbox'], img_width, img_height)
            lines.append(f"  人员{i}：({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})")

        lines.append(f"\n画面中所有目标：")
        for i, (name, coords) in enumerate(all_targets, 1):
            lines.append(f"{i}. {name}：({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})")

        lines.append(f"\n建议立即展开救援行动。")
    else:
        lines = ["画面中未发现水中人员。\n"]
        lines.append("检测到的其他目标：")
        for i, (name, coords) in enumerate(all_targets, 1):
            lines.append(f"{i}. {name}：({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})")
        lines.append("\n当前区域暂无人员需要救援。")

    return "\n".join(lines)


def generate_count_response(annotations, img_width, img_height):
    """生成计数类的回复"""
    from collections import Counter
    cat_counter = Counter(CATEGORY_NAMES.get(ann['category_id'], '未知') for ann in annotations)

    lines = ["目标统计："]
    for name, count in cat_counter.items():
        unit = '名' if name == '水中人员' else '艘' if name in ['船只', '水上摩托'] else '个'
        lines.append(f"- {name}：{count}{unit}")

    lines.append("\n各目标位置：")
    for i, ann in enumerate(annotations, 1):
        cat_name = CATEGORY_NAMES.get(ann['category_id'], '未知')
        coords = bbox_to_normalized(ann['bbox'], img_width, img_height)
        lines.append(f"{i}. {name}：({coords[0]}, {coords[1]}, {coords[2]}, {coords[3]})")

    return "\n".join(lines)


# ============================================================
# 生成单条训练样本
# ============================================================
def create_sample(image_info, annotations, image_folder, prompt_type='detection'):
    """
    根据一张图片和其标注，生成一条 Qwen3-VL 格式的训练样本
    """
    img_width = image_info['width']
    img_height = image_info['height']
    file_name = image_info['file_name']
    image_path = f"{image_folder}/{file_name}"

    # 选择提问和回复
    if prompt_type == 'detection':
        prompt = random.choice(DETECTION_PROMPTS)
        response = generate_detection_response(annotations, img_width, img_height)
    elif prompt_type == 'rescue':
        prompt = random.choice(RESCUE_PROMPTS)
        response = generate_rescue_response(annotations, img_width, img_height)
    elif prompt_type == 'count':
        prompt = random.choice(COUNT_PROMPTS)
        response = generate_count_response(annotations, img_width, img_height)

    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response}
                ]
            }
        ]
    }
    return sample


# ============================================================
# 转换主函数
# ============================================================
def convert_coco_to_qwen(coco_json_path, image_folder, output_path):
    """
    将 COCO JSON 转换为 Qwen3-VL 微调格式
    """
    print(f"\n正在读取: {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        data = json.load(f)

    # 构建图片信息字典
    images_dict = {img['id']: img for img in data['images']}

    # 按 image_id 分组标注（过滤掉 category_id=0 的忽略项）
    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        if ann['category_id'] != 0:  # 跳过 ignored 类别
            annotations_by_image[ann['image_id']].append(ann)

    print(f"  图片总数: {len(images_dict)}")
    print(f"  有标注的图片数: {len(annotations_by_image)}")

    # 生成训练样本
    all_samples = []
    skipped = 0

    for image_id, img_info in images_dict.items():
        anns = annotations_by_image.get(image_id, [])

        if len(anns) == 0:
            skipped += 1
            continue

        # 每张图生成 1 条检测样本（必选）
        sample1 = create_sample(img_info, anns, image_folder, 'detection')
        all_samples.append(sample1)

        # 如果图中有 swimmer，额外生成 1 条搜救判断样本
        has_swimmer = any(a['category_id'] == 1 for a in anns)
        if has_swimmer:
            sample2 = create_sample(img_info, anns, image_folder, 'rescue')
            all_samples.append(sample2)

        # 30% 概率额外生成 1 条计数样本
        if random.random() < 0.3:
            sample3 = create_sample(img_info, anns, image_folder, 'count')
            all_samples.append(sample3)

    # 打乱顺序
    random.shuffle(all_samples)

    # 保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, ensure_ascii=False, indent=2)

    print(f"  跳过无标注图片: {skipped} 张")
    print(f"  生成训练样本: {len(all_samples)} 条")
    print(f"  保存到: {output_path}")

    return len(all_samples)


# ============================================================
# 执行转换
# ============================================================
if __name__ == '__main__':
    random.seed(42)  # 固定随机种子，保证可复现

    print("=" * 60)
    print("SeaDroneSee → Qwen3-VL 微调数据格式转换")
    print("=" * 60)

    # 转换训练集
    train_count = convert_coco_to_qwen(TRAIN_JSON, 'train', TRAIN_OUTPUT)

    # 转换验证集
    val_count = convert_coco_to_qwen(VAL_JSON, 'val', VAL_OUTPUT)

    print("\n" + "=" * 60)
    print("转换完成！")
    print(f"  训练集: {train_count} 条 → {TRAIN_OUTPUT}")
    print(f"  验证集: {val_count} 条 → {VAL_OUTPUT}")
    print("=" * 60)