"""
==========================================================================
零样本 vs LoRA微调 全面对比评估脚本
在验证集上计算量化指标，生成对比报告
==========================================================================
"""

import json
import os
import re
import torch
import time
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# ============================================================
# 配置
# ============================================================
MODEL_PATH = '/root/autodl-tmp/qwen_vl/models/Qwen/Qwen3-VL-8B-Instruct'
LORA_PATH = '/root/autodl-tmp/qwen_vl/lora_output_v2/best_model'
BASE = '/root/autodl-tmp/qwen_vl/finetune_data'
NUM_TEST = 20
IOU_THRESHOLD = 0.3

PROMPT = "请检测这张无人机航拍图中水面上的所有目标，返回每个目标的类别和位置坐标。"

# ============================================================
# 工具函数
# ============================================================

def parse_ground_truth(text):
    targets = []
    pattern = r'(水中人员|船只|水上摩托|救生设备|浮标)：\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
    matches = re.findall(pattern, text)
    for m in matches:
        targets.append({
            'category': m[0],
            'bbox': [int(m[1]), int(m[2]), int(m[3]), int(m[4])]
        })
    return targets


def parse_model_response(text):
    targets = []

    # 格式1：中文格式
    pattern1 = r'(水中人员|船只|水上摩托|救生设备|浮标)：\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
    matches1 = re.findall(pattern1, text)
    for m in matches1:
        targets.append({
            'category': m[0],
            'bbox': [int(m[1]), int(m[2]), int(m[3]), int(m[4])]
        })

    if targets:
        return targets

    # 格式2：JSON格式
    pattern2 = r'"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\].*?"label"\s*:\s*"([^"]+)"'
    matches2 = re.findall(pattern2, text)

    label_map = {
        '人': '水中人员', 'person': '水中人员', '水中人员': '水中人员',
        'boat': '船只', '船': '船只', '船只': '船只',
        '水上摩托': '水上摩托', 'jetski': '水上摩托',
        '救生设备': '救生设备', '浮标': '浮标',
    }

    for m in matches2:
        label = label_map.get(m[4], m[4])
        targets.append({
            'category': label,
            'bbox': [int(m[0]), int(m[1]), int(m[2]), int(m[3])]
        })

    return targets


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    if union == 0:
        return 0.0
    return inter / union


def match_predictions(preds, gts, iou_threshold=0.3):
    tp = 0
    fp = 0
    matched_ious = []
    category_correct = 0

    gt_matched = [False] * len(gts)

    for pred in preds:
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gts):
            if gt_matched[gt_idx]:
                continue
            iou = compute_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            gt_matched[best_gt_idx] = True
            matched_ious.append(best_iou)
            if pred['category'] == gts[best_gt_idx]['category']:
                category_correct += 1
        else:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)

    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'matched_ious': matched_ious,
        'category_correct': category_correct,
    }


# ============================================================
# 推理函数
# ============================================================

def run_inference(model, processor, img_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=256)

    output_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return response


# ============================================================
# 评估单个模型（增加详细输出）
# ============================================================

def evaluate_model(model, processor, test_samples, model_name):
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name}")
    print(f"共 {len(test_samples)} 张图片待评估")
    print(f"{'='*60}")

    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_ious = []
    all_cat_correct = 0
    all_cat_total = 0
    category_stats = {}

    eval_start = time.time()

    for i, sample in enumerate(test_samples):
        img_path_raw = sample['messages'][0]['content'][0]['image']
        img_path_full = os.path.join(BASE, img_path_raw)
        ground_truth_text = sample['messages'][1]['content'][0]['text']

        # 解析 ground truth
        gt_targets = parse_ground_truth(ground_truth_text)
        if len(gt_targets) == 0:
            print(f"  [{i+1}/{len(test_samples)}] {os.path.basename(img_path_raw)} - 跳过（无标注）")
            continue

        # 推理
        img_start = time.time()
        response = run_inference(model, processor, img_path_full, PROMPT)
        img_time = time.time() - img_start
        pred_targets = parse_model_response(response)

        # 匹配
        result = match_predictions(pred_targets, gt_targets, IOU_THRESHOLD)

        all_tp += result['tp']
        all_fp += result['fp']
        all_fn += result['fn']
        all_ious.extend(result['matched_ious'])
        all_cat_correct += result['category_correct']
        all_cat_total += result['tp']

        # 按类别统计 gt
        for gt in gt_targets:
            cat = gt['category']
            if cat not in category_stats:
                category_stats[cat] = {'gt': 0, 'detected': 0}
            category_stats[cat]['gt'] += 1

        # 按类别统计检测到的
        for pred in pred_targets:
            cat = pred['category']
            for gt in gt_targets:
                if compute_iou(pred['bbox'], gt['bbox']) >= IOU_THRESHOLD and gt['category'] == cat:
                    if cat not in category_stats:
                        category_stats[cat] = {'gt': 0, 'detected': 0}
                    category_stats[cat]['detected'] += 1
                    break

        # ===== 每张图都输出进度 =====
        elapsed = time.time() - eval_start
        avg_per_img = elapsed / (i + 1)
        remaining = avg_per_img * (len(test_samples) - i - 1)

        gt_count = len(gt_targets)
        pred_count = len(pred_targets)
        hit = result['tp']

        # 当前累计指标
        cur_precision = all_tp / max(all_tp + all_fp, 1)
        cur_recall = all_tp / max(all_tp + all_fn, 1)

        print(f"  [{i+1:>2}/{len(test_samples)}] {os.path.basename(img_path_raw):<16} "
              f"| 真实:{gt_count} 预测:{pred_count} 命中:{hit} "
              f"| 耗时:{img_time:.1f}s "
              f"| 累计P:{cur_precision*100:.1f}% R:{cur_recall*100:.1f}% "
              f"| 剩余≈{remaining/60:.1f}分钟")

    # 计算最终指标
    total_time = time.time() - eval_start
    precision = all_tp / max(all_tp + all_fp, 1)
    recall = all_tp / max(all_tp + all_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    avg_iou = sum(all_ious) / max(len(all_ious), 1)
    cat_accuracy = all_cat_correct / max(all_cat_total, 1)

    print(f"\n  ✅ {model_name} 评估完成！耗时 {total_time/60:.1f} 分钟")

    report = {
        'model_name': model_name,
        'num_images': len(test_samples),
        'total_gt': all_tp + all_fn,
        'total_pred': all_tp + all_fp,
        'tp': all_tp,
        'fp': all_fp,
        'fn': all_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_iou': avg_iou,
        'category_accuracy': cat_accuracy,
        'category_stats': category_stats,
        'eval_time': total_time,
    }

    return report


def print_report(report):
    print(f"\n{'─'*50}")
    print(f"  模型: {report['model_name']}")
    print(f"{'─'*50}")
    print(f"  测试图片数:    {report['num_images']}")
    print(f"  真实目标总数:  {report['total_gt']}")
    print(f"  预测目标总数:  {report['total_pred']}")
    print(f"  TP / FP / FN:  {report['tp']} / {report['fp']} / {report['fn']}")
    print(f"  精确率:        {report['precision']:.4f} ({report['precision']*100:.1f}%)")
    print(f"  召回率:        {report['recall']:.4f} ({report['recall']*100:.1f}%)")
    print(f"  F1 分数:       {report['f1']:.4f} ({report['f1']*100:.1f}%)")
    print(f"  平均 IoU:      {report['avg_iou']:.4f}")
    print(f"  分类准确率:    {report['category_accuracy']:.4f} ({report['category_accuracy']*100:.1f}%)")
    print(f"  评估耗时:      {report['eval_time']/60:.1f} 分钟")
    print(f"\n  按类别召回率:")
    for cat, stats in report['category_stats'].items():
        r = stats['detected'] / max(stats['gt'], 1)
        print(f"    {cat}: {stats['detected']}/{stats['gt']} = {r*100:.1f}%")


# ============================================================
# 主流程
# ============================================================

def main():
    total_start = time.time()

    print("=" * 60)
    print("  零样本 vs LoRA微调 全面对比评估")
    print("=" * 60)

    # 读取验证集
    with open(os.path.join(BASE, 'val.json'), 'r') as f:
        val_data = json.load(f)

    tested_images = set()
    test_samples = []
    for sample in val_data:
        img_path = sample['messages'][0]['content'][0]['image']
        if img_path not in tested_images and len(test_samples) < NUM_TEST:
            tested_images.add(img_path)
            test_samples.append(sample)

    print(f"评估图片数: {len(test_samples)}")

    # ========== 评估1：零样本 ==========
    print("\n\n" + "=" * 60)
    print("  [1/2] 加载零样本模型（原始 Qwen3-VL-8B）...")
    print("=" * 60)

    load_start = time.time()
    model_base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model_base.eval()
    print(f"  模型加载完成，耗时 {time.time()-load_start:.1f} 秒")

    report_zero = evaluate_model(model_base, processor, test_samples, "零样本（原始模型）")

    # 释放显存
    print("\n  释放零样本模型显存...")
    del model_base
    torch.cuda.empty_cache()
    print("  显存已释放 ✅")

    # ========== 评估2：LoRA 微调 ==========
    print("\n\n" + "=" * 60)
    print("  [2/2] 加载 LoRA 微调后模型...")
    print("=" * 60)

    load_start = time.time()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model_lora = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model_lora = PeftModel.from_pretrained(model_lora, LORA_PATH)
    model_lora.eval()
    print(f"  LoRA 模型加载完成，耗时 {time.time()-load_start:.1f} 秒")

    report_lora = evaluate_model(model_lora, processor, test_samples, "LoRA微调后模型")

    # ========== 输出对比报告 ==========
    print("\n\n")
    print("=" * 60)
    print("               对 比 报 告")
    print("=" * 60)

    print_report(report_zero)
    print_report(report_lora)

    # 对比表格
    print(f"\n\n{'='*60}")
    print("                 指标对比汇总")
    print(f"{'='*60}")
    print(f"{'指标':<16} {'零样本':>12} {'LoRA微调':>12} {'提升':>12}")
    print(f"{'─'*52}")

    metrics = [
        ('精确率', 'precision'),
        ('召回率', 'recall'),
        ('F1 分数', 'f1'),
        ('平均 IoU', 'avg_iou'),
        ('分类准确率', 'category_accuracy'),
    ]

    for name, key in metrics:
        v0 = report_zero[key]
        v1 = report_lora[key]
        diff = v1 - v0
        sign = '+' if diff >= 0 else ''
        print(f"{name:<16} {v0*100:>10.1f}% {v1*100:>10.1f}% {sign}{diff*100:>10.1f}%")

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  评估全部完成！总耗时 {total_time/60:.1f} 分钟")
    print(f"  以上数据可直接用于毕业论文。")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()