"""
vLLM 版 V2 微调模型测试
使用合并后的模型 + val 集评估

修复内容：
1. OMP_NUM_THREADS 环境变量修复（必须在 import torch/vllm 之前设置）
2. vLLM multi_modal_data 需要传入 PIL.Image 对象，而非文件路径字符串
3. 修复报告标题乱码
"""

# ★ 修复1：必须在 import torch/vllm 之前设置，否则 vLLM 内部调用
#   torch.set_num_threads() 时会因为无效的 OMP_NUM_THREADS 值而崩溃
import os
os.environ["OMP_NUM_THREADS"] = str(max(1, os.cpu_count() or 1))

import json
import re
import time
import torch
from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

# ============================================================
# 配置
# ============================================================
MODEL_PATH = '/root/autodl-tmp/qwen_vl/models/Qwen3-VL-8B-Instruct-merged-v2'
DATA_DIR = '/root/autodl-tmp/qwen_vl/finetune_data'
VAL_JSON = os.path.join(DATA_DIR, 'val.json')
VAL_IMG_DIR = os.path.join(DATA_DIR, 'val')

MAX_TEST = 50  # 测试图片数量（val 有 1547 张，先跑 50 张）

PROMPT = """请检测这张水域监控图像中的所有目标，包括：水中人员、船只、浮标、救生设备、水上摩托等。
对每个目标，给出类别和边界框坐标 [x1, y1, x2, y2]（像素坐标）。
请以JSON格式输出，格式如下：
```json
[
    {"bbox_2d": [x1, y1, x2, y2], "label": "类别名称"},
    ...
]
```"""

IOU_THRESHOLD = 0.3

LABEL_MAP = {
    '人': '水中人员', '人员': '水中人员', '游泳的人': '水中人员',
    '溺水者': '水中人员', '游泳者': '水中人员', '水中人员': '水中人员',
    '船': '船只', '船只': '船只', '快艇': '船只', '渔船': '船只',
    '浮标': '浮标', '浮球': '浮标', '漂浮物': '浮标',
    '救生设备': '救生设备', '救生圈': '救生设备', '救生衣': '救生设备',
    '水上摩托': '水上摩托', '摩托艇': '水上摩托',
}

VALID_LABELS = {'水中人员', '船只', '浮标', '救生设备', '水上摩托'}


# ============================================================
# 从 val.json 解析真实标签
# ============================================================
def load_val_labels(val_json):
    """
    从 val.json 解析出每张图的真实标注
    返回: {图片文件名: [{'label': xx, 'bbox': [x1,y1,x2,y2]}, ...]}
    """
    with open(val_json, 'r') as f:
        data = json.load(f)

    labels = {}
    for item in data:
        messages = item['messages']
        # 找图片名
        img_name = None
        for msg in messages:
            if msg['role'] == 'user':
                for c in msg['content']:
                    if c.get('type') == 'image':
                        # "val/16735.jpg" -> "16735.jpg"
                        img_name = os.path.basename(c['image'])
                break

        if not img_name:
            continue

        # 找助手回复中的标注
        assistant_text = ''
        for msg in messages:
            if msg['role'] == 'assistant':
                for c in msg['content']:
                    if c.get('type') == 'text':
                        assistant_text = c['text']
                break

        # 解析坐标: "类别：(x1, y1, x2, y2)" 格���
        pattern = r'([\u4e00-\u9fff]+)[\s：:]*[\(（]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[\)）]'
        matches = re.findall(pattern, assistant_text)

        gts = []
        for label, x1, y1, x2, y2 in matches:
            label = LABEL_MAP.get(label, label)
            if label in VALID_LABELS:
                gts.append({'label': label, 'bbox': [int(x1), int(y1), int(x2), int(y2)]})

        if img_name not in labels:
            labels[img_name] = gts
        else:
            # 同一张图多条标注，合并（去重）
            existing_bboxes = [str(g['bbox']) for g in labels[img_name]]
            for gt in gts:
                if str(gt['bbox']) not in existing_bboxes:
                    labels[img_name].append(gt)

    return labels


# ============================================================
# 解析模型预测
# ============================================================
def parse_predictions(text):
    preds = []

    # JSON 格式
    json_match = re.search(r'\[.*\]', text, re.DOTALL)
    if json_match:
        try:
            items = json.loads(json_match.group())
            for item in items:
                if isinstance(item, dict):
                    bbox = item.get('bbox_2d') or item.get('坐标') or item.get('bbox')
                    label = item.get('label') or item.get('类别') or item.get('目标')
                    if bbox and label:
                        label = LABEL_MAP.get(label, label)
                        if label in VALID_LABELS:
                            preds.append({'label': label, 'bbox': bbox})
            if preds:
                return preds
        except json.JSONDecodeError:
            pass

    # 文本格式
    pattern = r'([\u4e00-\u9fff]+)[\s：:]*[\(（]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[,，]\s*(\d+)\s*[\)）]'
    matches = re.findall(pattern, text)
    for label, x1, y1, x2, y2 in matches:
        label = LABEL_MAP.get(label, label)
        if label in VALID_LABELS:
            preds.append({'label': label, 'bbox': [int(x1), int(y1), int(x2), int(y2)]})

    return preds


def calc_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def match_predictions(gts, preds, iou_threshold=IOU_THRESHOLD):
    tp, fp, fn = 0, 0, 0
    matched_gt = set()
    iou_list = []

    for pred in preds:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(gts):
            if i in matched_gt:
                continue
            iou = calc_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
            iou_list.append(best_iou)
        else:
            fp += 1

    fn = len(gts) - len(matched_gt)
    return tp, fp, fn, iou_list


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("  🚀 vLLM 版 V2 微调模型测试 (val 集)")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    try:
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        vram = 0
    print(f"  显存: {vram:.0f} GB")
    print("=" * 60)

    # 加载 val 标签
    print("\n  解析 val.json 标签...")
    labels = load_val_labels(VAL_JSON)

    # 只选有标签且图片存在的
    valid_images = []
    for img_name, gts in labels.items():
        if gts and os.path.exists(os.path.join(VAL_IMG_DIR, img_name)):
            valid_images.append(img_name)

    valid_images = sorted(valid_images)[:MAX_TEST]
    print(f"  val 集有标签的图片: {len(labels)}")
    print(f"  本次测试图片数: {len(valid_images)}")

    # 统计标签分布
    total_gt = sum(len(labels[img]) for img in valid_images)
    print(f"  本次真实目标总数: {total_gt}")

    # 加载 vLLM 模型
    print("\n" + "─" * 60)
    print("  加载 vLLM 模型中...")
    t0 = time.time()

    llm = LLM(
        model=MODEL_PATH,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": 256 * 28 * 28,
            "max_pixels": 768 * 28 * 28,
        },
    )

    load_time = time.time() - t0
    mem_used = torch.cuda.memory_allocated() / 1024**3
    print(f"  ✅ 模型加载完成 | 耗时: {load_time:.1f}s | 显存: {mem_used:.1f}GB")

    # 加载 processor（用于 chat template）
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=1024,
        stop=["<|im_end|>"],
    )

    # 开始测试
    print("\n" + "=" * 60)
    print("  开始逐张测试")
    print("=" * 60)

    total_tp, total_fp, total_fn = 0, 0, 0
    all_ious = []
    total_correct_cls = 0
    total_matched = 0
    category_stats = {}
    infer_times = []
    start_time = time.time()

    for idx, img_name in enumerate(valid_images):
        img_path = os.path.join(VAL_IMG_DIR, img_name)
        gts = labels[img_name]

        print(f"\n  [{idx+1:2d}/{len(valid_images)}] 📷 {img_name}")
        print("  " + "─" * 50)
        print(f"  真实目标 ({len(gts)}个):")
        for g in gts:
            print(f"    ▫ {g['label']}: ({g['bbox'][0]}, {g['bbox'][1]}, {g['bbox'][2]}, {g['bbox'][3]})")

        # ★ 修复2：vLLM 的 multi_modal_data 需要 PIL.Image 对象，不能传文件路径字符串
        #   原代码传的是 "file:///root/..." 字符串，vLLM 无法解析会报错
        pil_image = Image.open(img_path).convert("RGB")

        # 构建 vLLM 输入（chat template 中的 image 占位符仍用 file:// 格式）
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": PROMPT},
            ]}
        ]

        prompt_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        t1 = time.time()
        outputs = llm.generate(
            {
                "prompt": prompt_text,
                "multi_modal_data": {
                    "image": pil_image,  # ★ 传入 PIL.Image 对象
                },
            },
            sampling_params=sampling_params,
        )
        infer_time = time.time() - t1
        infer_times.append(infer_time)

        response = outputs[0].outputs[0].text
        preds = parse_predictions(response)

        print(f"  模型预测 ({len(preds)}个):")
        if not preds:
            print("    ⚠️ 未检测到任何目标")
        for p in preds:
            print(f"    ▸ {p['label']}: ({p['bbox'][0]}, {p['bbox'][1]}, {p['bbox'][2]}, {p['bbox'][3]})")

        tp, fp, fn, ious = match_predictions(gts, preds)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_ious.extend(ious)

        # 分类准确率
        for pred in preds:
            for gt in gts:
                iou = calc_iou(pred['bbox'], gt['bbox'])
                if iou >= IOU_THRESHOLD:
                    total_matched += 1
                    if pred['label'] == gt['label']:
                        total_correct_cls += 1
                    break

        # 按类别统计
        for gt in gts:
            cat = gt['label']
            if cat not in category_stats:
                category_stats[cat] = {'total': 0, 'detected': 0}
            category_stats[cat]['total'] += 1

        matched_gt_indices = set()
        for pred in preds:
            for i, gt in enumerate(gts):
                if i in matched_gt_indices:
                    continue
                if calc_iou(pred['bbox'], gt['bbox']) >= IOU_THRESHOLD:
                    category_stats[gt['label']]['detected'] += 1
                    matched_gt_indices.add(i)
                    break

        # 打印进度
        cum_p = total_tp / (total_tp + total_fp) * 100 if (total_tp + total_fp) > 0 else 0
        cum_r = total_tp / (total_tp + total_fn) * 100 if (total_tp + total_fn) > 0 else 0
        elapsed = time.time() - start_time
        remaining = elapsed / (idx + 1) * (len(valid_images) - idx - 1)

        print(f"  匹配: TP={tp} FP={fp} FN={fn}", end="")
        if ious:
            print(f" | 本张IoU: {sum(ious)/len(ious):.4f}", end="")
        print()
        print(f"  ⏱️  本张: {infer_time:.1f}s | 累计 P:{cum_p:.1f}% R:{cum_r:.1f}% | 剩���≈{remaining/60:.1f}分钟")

        # 显示模型回复（截取）
        resp_short = response[:150].replace('\n', ' ')
        print(f"  回复: \"{resp_short}...\"" if len(response) > 150 else f"  回复: \"{resp_short}\"")

    # ============================================================
    # 汇总报告
    # ============================================================
    precision = total_tp / (total_tp + total_fp) * 100 if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) * 100 if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_iou = sum(all_ious) / len(all_ious) if all_ious else 0
    cls_acc = total_correct_cls / total_matched * 100 if total_matched > 0 else 0
    total_time = time.time() - start_time
    avg_infer = sum(infer_times) / len(infer_times) if infer_times else 0

    # ★ 修复3：原代码这里的 emoji 是乱码 "����"，替换为正确的 📊
    print("\n\n" + "=" * 60)
    print("  📊 vLLM 版 V2 微调模型 测试报告 (val 集)")
    print("=" * 60)
    print(f"  测试图片数:     {len(valid_images)}")
    print(f"  真实目标总数:   {total_tp + total_fn}")
    print(f"  预测目标总数:   {total_tp + total_fp}")
    print(f"  TP / FP / FN:   {total_tp} / {total_fp} / {total_fn}")
    print("─" * 60)
    print(f"  精确率 (P):     {precision:.1f}%")
    print(f"  召回率 (R):     {recall:.1f}%")
    print(f"  F1 分数:        {f1:.1f}%")
    print(f"  平均 IoU:       {avg_iou:.4f}")
    print(f"  分类准确率:     {cls_acc:.1f}%")
    print("─" * 60)
    print(f"  ⏱️  总耗时:         {total_time/60:.1f} 分钟")
    print(f"  ⏱️  平均每张:       {avg_infer:.1f} 秒")
    print(f"  ⏱️  推理速度:       {len(valid_images)/total_time:.2f} 张/秒")
    print(f"  ⏱️  首张耗时:       {infer_times[0]:.1f} 秒（含预热）")
    print(f"  ⏱️  后续平均:       {sum(infer_times[1:])/max(len(infer_times)-1,1):.1f} 秒")

    print(f"\n  📋 按类别召回率:")
    for cat, stats in sorted(category_stats.items()):
        det = stats['detected']
        tot = stats['total']
        rate = det / tot * 100 if tot > 0 else 0
        bar = '█' * int(rate / 5) + '░' * (20 - int(rate / 5))
        print(f"    {cat:8s}  {det:3d}/{tot:<3d}  {bar} {rate:.1f}%")

    print("\n" + "=" * 60)
    print("  ✅ 测��完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
