"""
==========================================================================
LoRA 微调模型 全性能测试脚本
RTX PRO 6000 (96GB) 专用 - 全精度加载 + 批量推理
==========================================================================
功能：
1. 全精度 bfloat16 加载（不量化，96GB 显存完全够）
2. 批量推理（batch inference）多张图同时处理
3. 在验证集上全面评估，输出详细报告
4. 保存检测结果可视化图片（画框 + 标签）
5. 零样本 vs 微调 对比
==========================================================================
"""

import json
import os
import re
import time
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# ============================================================
# 配置
# ============================================================
MODEL_PATH = '/root/autodl-tmp/qwen_vl/models/Qwen/Qwen3-VL-8B-Instruct'
LORA_PATH = '/root/autodl-tmp/qwen_vl/lora_output_v2/best_model'
BASE = '/root/autodl-tmp/qwen_vl/finetune_data'
OUTPUT_DIR = '/root/autodl-tmp/qwen_vl/test_results'

NUM_TEST = 100          # 评估图片数（96GB 显存 + 全精度，跑 100 张没问题）
BATCH_SIZE = 4          # 批量大小（一次同时推理 4 张图）
MAX_NEW_TOKENS = 256
IOU_THRESHOLD = 0.3

PROMPT = "请检测这张无人机航拍图中水面上的所有目标，返回每个目标的类别和位置坐标。"

# 类别颜色（画框用）
CATEGORY_COLORS = {
    '水中人员': '#FF0000',   # 红色
    '船只':    '#00FF00',    # 绿色
    '水上摩托': '#0000FF',   # 蓝色
    '救生设备': '#FFD700',   # 金色
    '浮标':    '#FF00FF',    # 紫色
}

# ============================================================
# 工具函数
# ============================================================

def parse_ground_truth(text):
    targets = []
    pattern = r'(水中人员|船只|水上摩托|救生设备|浮标)：\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
    for m in re.findall(pattern, text):
        targets.append({
            'category': m[0],
            'bbox': [int(m[1]), int(m[2]), int(m[3]), int(m[4])]
        })
    return targets


def parse_model_response(text):
    targets = []

    # 格式1：中文格式
    pattern1 = r'(水中人员|船只|水上摩托|救生设备|浮标)：\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)'
    for m in re.findall(pattern1, text):
        targets.append({
            'category': m[0],
            'bbox': [int(m[1]), int(m[2]), int(m[3]), int(m[4])]
        })
    if targets:
        return targets

    # 格式2：JSON 格式
    pattern2 = r'"bbox_2d"\s*:\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\].*?"label"\s*:\s*"([^"]+)"'
    label_map = {
        '人': '水中人员', 'person': '水中人员', '水中人员': '水中人员',
        'boat': '船只', '船': '船只', '船只': '船只',
        '水上摩托': '水上摩托', 'jetski': '水上摩托',
        '救生设备': '救生设备', '浮标': '浮标',
    }
    for m in re.findall(pattern2, text):
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
# 可视化：在图片上画检测框
# ============================================================

def draw_detections(img_path, pred_targets, gt_targets, save_path):
    """在图片上画预测框（实线）和真实框（虚线），保存对比图"""
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    # 尝试加载中文字体，失败就用默认
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    # 画真实框（细线 + 白色）
    for gt in gt_targets:
        x1, y1, x2, y2 = gt['bbox']
        draw.rectangle([x1, y1, x2, y2], outline='white', width=1)

    # 画预测框（粗线 + 彩色 + 标签）
    for pred in pred_targets:
        x1, y1, x2, y2 = pred['bbox']
        color = CATEGORY_COLORS.get(pred['category'], '#FFFFFF')
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = pred['category']
        draw.text((x1, max(y1 - 18, 0)), label, fill=color, font=font)

    img.save(save_path)


# ============================================================
# 单张推理
# ============================================================

def run_single_inference(model, processor, img_path, prompt):
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
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    output_ids = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return response


# ============================================================
# 批量推理（核心加速）
# ============================================================

def run_batch_inference(model, processor, img_paths, prompt):
    """同时推理多张图片，充分利用 GPU"""
    all_texts = []
    all_image_inputs = []

    for img_path in img_paths:
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
        image_inputs, _ = process_vision_info(messages)
        all_texts.append(text)
        all_image_inputs.extend(image_inputs)

    inputs = processor(
        text=all_texts,
        images=all_image_inputs,
        videos=None,
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    # 解码每个样本的输出
    responses = []
    for i in range(len(img_paths)):
        # 截取生成的部分
        input_len = (inputs.input_ids[i] != processor.tokenizer.pad_token_id).sum().item()
        output = output_ids[i][input_len:]
        resp = processor.decode(output, skip_special_tokens=True)
        responses.append(resp)

    return responses


# ============================================================
# 评估函数
# ============================================================

def evaluate_model(model, processor, test_samples, model_name, save_vis=False):
    print(f"\n{'='*60}")
    print(f"  评估模型: {model_name}")
    print(f"  图片数: {len(test_samples)} | 批量大小: {BATCH_SIZE}")
    print(f"  推理精度: bfloat16（全精度，无量化）")
    print(f"{'='*60}")

    all_tp, all_fp, all_fn = 0, 0, 0
    all_ious = []
    all_cat_correct, all_cat_total = 0, 0
    category_stats = {}
    total_tokens = 0

    eval_start = time.time()

    # 准备所有图片路径和 ground truth
    img_paths = []
    gt_list = []
    valid_indices = []

    for i, sample in enumerate(test_samples):
        img_path_raw = sample['messages'][0]['content'][0]['image']
        img_path_full = os.path.join(BASE, img_path_raw)
        gt_text = sample['messages'][1]['content'][0]['text']
        gt_targets = parse_ground_truth(gt_text)

        if len(gt_targets) == 0:
            continue

        img_paths.append(img_path_full)
        gt_list.append(gt_targets)
        valid_indices.append(i)

    print(f"  有效图片: {len(img_paths)} 张（过滤无标注后）")
    print(f"  批次数: {(len(img_paths) + BATCH_SIZE - 1) // BATCH_SIZE}")
    print(f"{'─'*60}")

    # 批量推理
    all_responses = []
    batch_count = 0

    for batch_start in range(0, len(img_paths), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(img_paths))
        batch_paths = img_paths[batch_start:batch_end]
        batch_count += 1

        batch_time_start = time.time()

        try:
            responses = run_batch_inference(model, processor, batch_paths, PROMPT)
        except Exception as e:
            # 如果批量推理失败（OOM 等），降级为单张推理
            print(f"  ⚠️ 批量推理失败，降级为单张推理: {e}")
            responses = []
            for path in batch_paths:
                resp = run_single_inference(model, processor, path, PROMPT)
                responses.append(resp)

        batch_time = time.time() - batch_time_start
        all_responses.extend(responses)

        # 打印批次进度
        elapsed = time.time() - eval_start
        processed = batch_end
        speed = processed / elapsed
        remaining = (len(img_paths) - processed) / max(speed, 0.01)

        print(f"  批次 {batch_count:>3} | "
              f"图片 {batch_start+1}-{batch_end}/{len(img_paths)} | "
              f"批次耗时: {batch_time:.1f}s | "
              f"每张: {batch_time/len(batch_paths):.1f}s | "
              f"速度: {speed:.2f} 张/秒 | "
              f"剩余: {remaining/60:.1f}分钟")

    # 统计指标
    print(f"\n{'─'*60}")
    print(f"  推理完成，统计指标中...")

    for idx in range(len(img_paths)):
        pred_targets = parse_model_response(all_responses[idx])
        gt_targets = gt_list[idx]

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
                category_stats[cat] = {'gt': 0, 'detected': 0, 'correct_cls': 0}
            category_stats[cat]['gt'] += 1

        for pred in pred_targets:
            cat = pred['category']
            for gt in gt_targets:
                if compute_iou(pred['bbox'], gt['bbox']) >= IOU_THRESHOLD and gt['category'] == cat:
                    if cat not in category_stats:
                        category_stats[cat] = {'gt': 0, 'detected': 0, 'correct_cls': 0}
                    category_stats[cat]['detected'] += 1
                    break

        # 保存可视化（前 20 张）
        if save_vis and idx < 20:
            vis_dir = os.path.join(OUTPUT_DIR, model_name.replace(' ', '_').replace('（', '').replace('）', ''))
            os.makedirs(vis_dir, exist_ok=True)
            img_name = os.path.basename(img_paths[idx])
            save_path = os.path.join(vis_dir, f"det_{img_name}")
            draw_detections(img_paths[idx], pred_targets, gt_targets, save_path)

    # 计算指标
    total_time = time.time() - eval_start
    precision = all_tp / max(all_tp + all_fp, 1)
    recall = all_tp / max(all_tp + all_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    avg_iou = sum(all_ious) / max(len(all_ious), 1)
    cat_accuracy = all_cat_correct / max(all_cat_total, 1)

    print(f"\n  ✅ {model_name} 评估完成！")
    print(f"  总耗时: {total_time/60:.1f} 分钟 | 平均: {total_time/len(img_paths):.1f} 秒/张")

    return {
        'model_name': model_name,
        'num_images': len(img_paths),
        'total_gt': all_tp + all_fn,
        'total_pred': all_tp + all_fp,
        'tp': all_tp, 'fp': all_fp, 'fn': all_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'avg_iou': avg_iou,
        'category_accuracy': cat_accuracy,
        'category_stats': category_stats,
        'eval_time': total_time,
        'speed': len(img_paths) / total_time,
    }


def print_report(report):
    print(f"\n{'━'*55}")
    print(f"  📊 {report['model_name']}")
    print(f"{'━'*55}")
    print(f"  测试图片数:     {report['num_images']}")
    print(f"  真实目标总数:   {report['total_gt']}")
    print(f"  预测目标总数:   {report['total_pred']}")
    print(f"  TP / FP / FN:   {report['tp']} / {report['fp']} / {report['fn']}")
    print(f"  ──────────────────────────────────")
    print(f"  精确率 (P):     {report['precision']:.4f}  ({report['precision']*100:.1f}%)")
    print(f"  召回率 (R):     {report['recall']:.4f}  ({report['recall']*100:.1f}%)")
    print(f"  F1 分数:        {report['f1']:.4f}  ({report['f1']*100:.1f}%)")
    print(f"  平均 IoU:       {report['avg_iou']:.4f}")
    print(f"  分类准确率:     {report['category_accuracy']:.4f}  ({report['category_accuracy']*100:.1f}%)")
    print(f"  ──────────────────────────────────")
    print(f"  推理速度:       {report['speed']:.2f} 张/秒")
    print(f"  评估耗时:       {report['eval_time']/60:.1f} 分钟")
    print(f"\n  📋 按类别召回率:")
    for cat, stats in sorted(report['category_stats'].items(), key=lambda x: x[1]['gt'], reverse=True):
        r = stats['detected'] / max(stats['gt'], 1)
        bar = '█' * int(r * 20) + '░' * (20 - int(r * 20))
        print(f"    {cat:<8} {stats['detected']:>3}/{stats['gt']:<3} {bar} {r*100:.1f}%")


# ============================================================
# 主流程
# ============================================================

def main():
    total_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  🚀 LoRA 微调模型 全性能测试")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.0f} GB")
    print(f"  精度: bfloat16（全精度，无量化）")
    print(f"  批量大小: {BATCH_SIZE}")
    print("=" * 60)

    # 读取验证集
    with open(os.path.join(BASE, 'val.json'), 'r') as f:
        val_data = json.load(f)

    # 去重（每张图片只取一条）
    tested_images = set()
    test_samples = []
    for sample in val_data:
        img_path = sample['messages'][0]['content'][0]['image']
        if img_path not in tested_images and len(test_samples) < NUM_TEST:
            tested_images.add(img_path)
            test_samples.append(sample)

    print(f"\n  评估图片数: {len(test_samples)}")

    # ========== 加载处理器 ==========
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # ========== 评估 1: 零样本 ==========
    print(f"\n\n{'='*60}")
    print("  [1/2] 加载零样本模型（bfloat16 全精度）")
    print("=" * 60)

    load_start = time.time()
    model_base = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model_base.eval()
    mem_used = torch.cuda.memory_allocated() / 1024**3
    print(f"  加载完成 | 耗时: {time.time()-load_start:.1f}s | 显存: {mem_used:.1f}GB")

    report_zero = evaluate_model(model_base, processor, test_samples, "零样本（原始模型）", save_vis=True)

    del model_base
    torch.cuda.empty_cache()
    print("\n  显存已释放 ✅")

    # ========== 评估 2: LoRA 微调 ==========
    print(f"\n\n{'='*60}")
    print("  [2/2] 加载 LoRA 微调模型（bfloat16 全精度，无量化！）")
    print("=" * 60)

    load_start = time.time()
    model_lora = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model_lora = PeftModel.from_pretrained(model_lora, LORA_PATH)
    model_lora = model_lora.merge_and_unload()  # 合并 LoRA 权重，推理更快！
    model_lora.eval()
    mem_used = torch.cuda.memory_allocated() / 1024**3
    print(f"  加载完成（LoRA 已合并）| 耗时: {time.time()-load_start:.1f}s | 显存: {mem_used:.1f}GB")

    report_lora = evaluate_model(model_lora, processor, test_samples, "LoRA微调（全精度）", save_vis=True)

    # ========== 对比报告 ==========
    print(f"\n\n{'='*60}")
    print("  📊 对 比 报 告")
    print("=" * 60)

    print_report(report_zero)
    print_report(report_lora)

    # 对比表格
    print(f"\n\n{'='*60}")
    print("  📈 指标对比汇总")
    print(f"{'='*60}")
    print(f"  {'指标':<12} {'零样本':>10} {'LoRA微调':>10} {'提升':>10}")
    print(f"  {'─'*42}")

    metrics = [
        ('精确率', 'precision'),
        ('召回率', 'recall'),
        ('F1 分数', 'f1'),
        ('平均 IoU', 'avg_iou'),
        ('分类准确率', 'category_accuracy'),
        ('推理速度', 'speed'),
    ]

    for name, key in metrics:
        v0 = report_zero[key]
        v1 = report_lora[key]
        diff = v1 - v0
        sign = '+' if diff >= 0 else ''
        if key == 'speed':
            print(f"  {name:<12} {v0:>8.2f}张/s {v1:>8.2f}张/s {sign}{diff:>8.2f}张/s")
        else:
            print(f"  {name:<12} {v0*100:>9.1f}% {v1*100:>9.1f}% {sign}{diff*100:>9.1f}%")

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  ✅ 全部完成！总耗时: {total_time/60:.1f} 分钟")
    print(f"  可视化结果保存在: {OUTPUT_DIR}/")
    print(f"  以上数据可直接用于毕业论文。")
    print(f"{'='*60}")

    # 保存报告到文件
    report_path = os.path.join(OUTPUT_DIR, 'evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        # category_stats 直接保存
        json.dump({
            'zero_shot': {k: v for k, v in report_zero.items() if k != 'category_stats'},
            'lora_finetuned': {k: v for k, v in report_lora.items() if k != 'category_stats'},
        }, f, ensure_ascii=False, indent=2)
    print(f"  报告已保存: {report_path}")


if __name__ == "__main__":
    main()