"""
==========================================================================
V2 微调模型 快速测试脚本
RTX PRO 6000 (96GB) - bfloat16 全精度 + LoRA 合并
==========================================================================
在 20 张验证集图片上快速验证微调效果
每张图输出：原图名 | 真实目标 | 模型预测 | 命中情况 | 模型原始回复
最后输出统计报告
==========================================================================
"""

import json
import os
import re
import time
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# ============================================================
# 配置
# ============================================================
MODEL_PATH = '/root/autodl-tmp/qwen_vl/models/Qwen/Qwen3-VL-8B-Instruct'
LORA_PATH = '/root/autodl-tmp/qwen_vl/lora_output_v2/best_model'
BASE = '/root/autodl-tmp/qwen_vl/finetune_data'
NUM_TEST = 20
MAX_NEW_TOKENS = 256
IOU_THRESHOLD = 0.3

PROMPT = "请检测这张无人机航拍图中水面上的所有目标，返回每个目标的类别和位置坐标。"


# ============================================================
# 工具函数
# ============================================================

def parse_ground_truth(text):
    targets = []
    pattern = r'(水中人员|船只|水上摩托|救生设备|浮标)：\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
    for m in re.findall(pattern, text):
        targets.append({'category': m[0], 'bbox': [int(m[1]), int(m[2]), int(m[3]), int(m[4])]})
    return targets


def parse_model_response(text):
    targets = []
    pattern1 = r'(水中人员|船只|水上摩托|救生设备|浮标)：\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
    for m in re.findall(pattern1, text):
        targets.append({'category': m[0], 'bbox': [int(m[1]), int(m[2]), int(m[3]), int(m[4])]})
    if targets:
        return targets

    pattern2 = r'"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\].*?"label"\s*:\s*"([^"]+)"'
    label_map = {
        '人': '水中人员', 'person': '水中人员', '水中人员': '水中人员',
        'boat': '船只', '船': '船只', '船只': '船只',
        '水上摩托': '水上摩托', 'jetski': '水上摩托',
        '救生设备': '救生设备', '浮标': '浮标',
    }
    for m in re.findall(pattern2, text):
        label = label_map.get(m[4], m[4])
        targets.append({'category': label, 'bbox': [int(m[0]), int(m[1]), int(m[2]), int(m[3])]})
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
    return inter / union if union > 0 else 0.0


def match_predictions(preds, gts, iou_threshold=0.3):
    tp, fp, category_correct = 0, 0, 0
    matched_ious = []
    gt_matched = [False] * len(gts)

    for pred in preds:
        best_iou, best_gt_idx = 0, -1
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
    return {'tp': tp, 'fp': fp, 'fn': fn, 'matched_ious': matched_ious, 'category_correct': category_correct}


# ============================================================
# 推理
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
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    output_ids = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(output_ids, skip_special_tokens=True)[0]


# ============================================================
# 主流程
# ============================================================

def main():
    total_start = time.time()

    print("=" * 60)
    print("  🔍 V2 微调模型 快速测试")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB")
    print(f"  精度: bfloat16 全精度（无量化）")
    print(f"  LoRA: 合并后推理（merge_and_unload）")
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

    print(f"\n  测试图片数: {len(test_samples)}")

    # 加载模型
    print(f"\n{'─'*60}")
    print("  加载模型中...")
    load_start = time.time()

    processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=256*28*28, max_pixels=768*28*28)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model = model.merge_and_unload()
    model.eval()

    mem = torch.cuda.memory_allocated() / 1024**3
    print(f"  ✅ 加载完成 | 耗时: {time.time()-load_start:.1f}s | 显存占用: {mem:.1f}GB")

    # 开始测试
    print(f"\n{'='*60}")
    print("  开始逐张测试")
    print(f"{'='*60}")

    all_tp, all_fp, all_fn = 0, 0, 0
    all_ious = []
    all_cat_correct, all_cat_total = 0, 0
    category_stats = {}
    eval_start = time.time()

    for i, sample in enumerate(test_samples):
        img_path_raw = sample['messages'][0]['content'][0]['image']
        img_path_full = os.path.join(BASE, img_path_raw)
        gt_text = sample['messages'][1]['content'][0]['text']
        gt_targets = parse_ground_truth(gt_text)

        if len(gt_targets) == 0:
            print(f"\n  [{i+1:>2}/{NUM_TEST}] {os.path.basename(img_path_raw)} — 跳过（无标注）")
            continue

        # 推理
        img_start = time.time()
        response = run_inference(model, processor, img_path_full, PROMPT)
        img_time = time.time() - img_start

        pred_targets = parse_model_response(response)
        result = match_predictions(pred_targets, gt_targets, IOU_THRESHOLD)

        all_tp += result['tp']
        all_fp += result['fp']
        all_fn += result['fn']
        all_ious.extend(result['matched_ious'])
        all_cat_correct += result['category_correct']
        all_cat_total += result['tp']

        # 按类别统计
        for gt in gt_targets:
            cat = gt['category']
            if cat not in category_stats:
                category_stats[cat] = {'gt': 0, 'detected': 0}
            category_stats[cat]['gt'] += 1
        for pred in pred_targets:
            for gt in gt_targets:
                if compute_iou(pred['bbox'], gt['bbox']) >= IOU_THRESHOLD and gt['category'] == pred['category']:
                    if pred['category'] not in category_stats:
                        category_stats[pred['category']] = {'gt': 0, 'detected': 0}
                    category_stats[pred['category']]['detected'] += 1
                    break

        # 累计指标
        cur_p = all_tp / max(all_tp + all_fp, 1)
        cur_r = all_tp / max(all_tp + all_fn, 1)
        elapsed = time.time() - eval_start
        remaining = elapsed / (i + 1) * (NUM_TEST - i - 1)

        # 打印每张图的详细结果
        print(f"\n  [{i+1:>2}/{NUM_TEST}] 📷 {os.path.basename(img_path_raw)}")
        print(f"  {'─'*50}")
        print(f"  真实目标 ({len(gt_targets)}个):")
        for gt in gt_targets:
            print(f"    ▫ {gt['category']}: ({gt['bbox'][0]}, {gt['bbox'][1]}, {gt['bbox'][2]}, {gt['bbox'][3]})")
        print(f"  模型预测 ({len(pred_targets)}个):")
        if pred_targets:
            for pred in pred_targets:
                print(f"    ▸ {pred['category']}: ({pred['bbox'][0]}, {pred['bbox'][1]}, {pred['bbox'][2]}, {pred['bbox'][3]})")
        else:
            print(f"    ⚠️ 模型未检测到任何目标")
        print(f"  匹配结果: TP={result['tp']} FP={result['fp']} FN={result['fn']}")
        if result['matched_ious']:
            avg_iou_this = sum(result['matched_ious']) / len(result['matched_ious'])
            print(f"  本张IoU: {avg_iou_this:.4f}")
        print(f"  耗时: {img_time:.1f}s | 累计 P:{cur_p*100:.1f}% R:{cur_r*100:.1f}% | 剩余≈{remaining/60:.1f}分钟")

        # 打印模型原始回复（方便调试）
        print(f"  模型原始回复:")
        print(f"    \"{response[:200]}{'...' if len(response) > 200 else ''}\"")

    # ========== 最终报告 ==========
    total_time = time.time() - eval_start
    precision = all_tp / max(all_tp + all_fp, 1)
    recall = all_tp / max(all_tp + all_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    avg_iou = sum(all_ious) / max(len(all_ious), 1)
    cat_accuracy = all_cat_correct / max(all_cat_total, 1)

    print(f"\n\n{'='*60}")
    print(f"  📊 V2 微调模型 测试报告")
    print(f"{'='*60}")
    print(f"  测试图片数:     {NUM_TEST}")
    print(f"  真实目标总数:   {all_tp + all_fn}")
    print(f"  预测目标总数:   {all_tp + all_fp}")
    print(f"  TP / FP / FN:   {all_tp} / {all_fp} / {all_fn}")
    print(f"{'─'*60}")
    print(f"  精确率 (P):     {precision*100:.1f}%")
    print(f"  召回率 (R):     {recall*100:.1f}%")
    print(f"  F1 分数:        {f1*100:.1f}%")
    print(f"  平均 IoU:       {avg_iou:.4f}")
    print(f"  分类准确率:     {cat_accuracy*100:.1f}%")
    print(f"{'─'*60}")
    print(f"  总耗时:         {total_time/60:.1f} 分钟")
    print(f"  平均每张:       {total_time/NUM_TEST:.1f} 秒")
    print(f"  推理速度:       {NUM_TEST/total_time:.2f} 张/秒")

    print(f"\n  📋 按类别召回率:")
    for cat, stats in sorted(category_stats.items(), key=lambda x: x[1]['gt'], reverse=True):
        r = stats['detected'] / max(stats['gt'], 1)
        bar = '█' * int(r * 20) + '░' * (20 - int(r * 20))
        print(f"    {cat:<8} {stats['detected']:>3}/{stats['gt']:<3} {bar} {r*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"  ✅ 测试完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()