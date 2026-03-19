"""
==========================================================================
Qwen3-VL-8B 全参数微调训练脚本（最终版）
==========================================================================
数据格式：V3（无 system prompt，bbox_2d 在前）
数据集：  train_split / val_split / test_split
硬件：    NVIDIA RTX Pro 6000 (96GB VRAM)
目标显存：~77GB（80%）

输出：
  best_model/           → 验证 loss 最低的模型（vLLM 直接加载推理）
  final_model/          → 最后一步的模型
  training_log.csv      → 每步的 loss/lr/显存记录
  training_curves.png   → loss 和 lr 曲线图
  train_config.json     → 本次训练超参数备份
"""

import os
import sys
import json
import csv
import math
import time
import torch
import random
import datetime
import numpy as np
from collections import deque
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    get_cosine_schedule_with_warmup,
)
from qwen_vl_utils import process_vision_info

# ╔══════════════════════════════════════════════════════════════╗
# ║  ⬇⬇⬇ 你需要检查/修改的配置 ⬇⬇⬇                            ║
# ╚══════════════════════════════════════════════════════════════╝

CONFIG = {
    # --- 路径 ---
    "model_path":  "/root/autodl-tmp/qwen_vl/models/Qwen/Qwen3-VL-8B-Instruct",
    "data_base":   "/root/autodl-tmp/qwen_vl/finetune_data_v2",
    "output_dir":  "/root/autodl-tmp/qwen_vl/full_ft_train/full_ft_output",

    # --- 数据文件名 ---
    "train_file":  "train_split.json",
    "val_file":    "val_split.json",

    # --- 训练超参数 ---
    "learning_rate": 1e-5,
    "num_epochs":    3,
    "batch_size":    2,
    "gradient_accumulation_steps": 8,
    "warmup_ratio":  0.05,
    "weight_decay":  0.01,
    "max_grad_norm": 1.0,

    # --- 图片/序列长度 ---
    "max_pixels":  672 * 672,
    "min_pixels":  336 * 336,
    "max_length":  2560,

    # --- 日志与保存 ---
    "log_every":   10,
    "eval_every":  250,
    "save_every":  999999,
    "eval_max_batches": 100,
}

# ╔══════════════════════════════════════════════════════════════╗
# ║  ⬆⬆⬆ 以下代码不需要改动 ⬆⬆⬆                                ║
# ╚══════════════════════════════════════════════════════════════╝


# ============================================================
# 工具函数
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    else:
        return f"{s}s"


def get_gpu_memory_gb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**3
    return 0.0


# ============================================================
# 数据集
# ============================================================
class SeaDroneDataset(Dataset):
    """
    V3 数据格式（无 system prompt）：
    messages[0] = user（图片+提问）
    messages[1] = assistant（JSON回复）
    """

    def __init__(self, json_path, base_dir, processor, max_length):
        print(f"  加载数据: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.base_dir = base_dir
        self.processor = processor
        self.max_length = max_length
        print(f"  样本数: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        messages = sample["messages"]

        user_msg      = messages[0]
        assistant_msg = messages[1]

        img_relative = user_msg["content"][0]["image"]
        img_absolute = os.path.join(self.base_dir, img_relative)

        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_absolute}"},
                    {"type": "text", "text": user_msg["content"][1]["text"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_msg["content"]},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(chat_messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()

        assistant_text = assistant_msg["content"]
        assistant_tokens = self.processor.tokenizer.encode(
            assistant_text, add_special_tokens=False
        )
        assistant_len = len(assistant_tokens)

        non_pad_mask = input_ids != self.processor.tokenizer.pad_token_id
        non_pad_indices = torch.where(non_pad_mask)[0]

        if len(non_pad_indices) > 0:
            labels[:] = -100
            total_valid = len(non_pad_indices)
            if assistant_len < total_valid:
                start_idx = non_pad_indices[total_valid - assistant_len]
                end_idx = non_pad_indices[-1] + 1
                labels[start_idx:end_idx] = input_ids[start_idx:end_idx]

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "labels": labels,
        }


def collate_fn(batch):
    result = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }
    if batch[0]["pixel_values"] is not None:
        result["pixel_values"] = torch.cat(
            [b["pixel_values"] for b in batch], dim=0
        )
    if batch[0]["image_grid_thw"] is not None:
        result["image_grid_thw"] = torch.cat(
            [b["image_grid_thw"] for b in batch], dim=0
        )
    return result


# ============================================================
# 日志记录器
# ============================================================
class TrainingLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, "training_log.csv")
        self.records = []

        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "epoch", "train_loss", "val_loss",
                "learning_rate", "gpu_memory_gb", "elapsed_sec", "timestamp"
            ])

    def log_step(self, step, epoch, train_loss, lr, elapsed):
        gpu_mem = get_gpu_memory_gb()
        record = {
            "step": step, "epoch": epoch,
            "train_loss": round(train_loss, 6), "val_loss": None,
            "learning_rate": lr, "gpu_memory_gb": round(gpu_mem, 2),
            "elapsed_sec": round(elapsed, 1),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.records.append(record)
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                record["step"], record["epoch"], record["train_loss"],
                "", record["learning_rate"], record["gpu_memory_gb"],
                record["elapsed_sec"], record["timestamp"],
            ])

    def log_eval(self, step, epoch, val_loss, elapsed):
        gpu_mem = get_gpu_memory_gb()
        record = {
            "step": step, "epoch": epoch,
            "train_loss": None, "val_loss": round(val_loss, 6),
            "learning_rate": None, "gpu_memory_gb": round(gpu_mem, 2),
            "elapsed_sec": round(elapsed, 1),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.records.append(record)
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                record["step"], record["epoch"], "",
                record["val_loss"], "", record["gpu_memory_gb"],
                record["elapsed_sec"], record["timestamp"],
            ])

    def plot_curves(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("  ⚠️ matplotlib 未安装，跳过画图。pip install matplotlib")
            return

        steps_train, losses_train = [], []
        steps_val, losses_val = [], []
        steps_lr, lrs = [], []

        for r in self.records:
            if r["train_loss"] is not None:
                steps_train.append(r["step"])
                losses_train.append(r["train_loss"])
            if r["val_loss"] is not None:
                steps_val.append(r["step"])
                losses_val.append(r["val_loss"])
            if r["learning_rate"] is not None:
                steps_lr.append(r["step"])
                lrs.append(r["learning_rate"])

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax1 = axes[0]
        ax1.plot(steps_train, losses_train, color='blue', alpha=0.3, linewidth=0.5, label='raw')
        if len(losses_train) > 20:
            window = min(50, len(losses_train) // 5)
            smoothed = np.convolve(losses_train, np.ones(window)/window, mode='valid')
            ax1.plot(steps_train[window-1:], smoothed, color='blue', linewidth=2, label='smoothed')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Train Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        if losses_val:
            ax2.plot(steps_val, losses_val, 'ro-', linewidth=2, markersize=6)
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Loss')
        ax2.set_title('Validation Loss')
        ax2.grid(True, alpha=0.3)

        ax3 = axes[2]
        if lrs:
            ax3.plot(steps_lr, lrs, color='green', linewidth=1.5)
            ax3.set_xlabel('Step')
            ax3.set_ylabel('LR')
        ax3.set_title('Learning Rate')
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.output_dir, "training_curves.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  📊 曲线图已保存: {save_path}")


# ============================================================
# 验证函数
# ============================================================
@torch.no_grad()
def evaluate(model, val_loader, device, max_batches):
    model.eval()
    total_loss = 0.0
    count = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
        try:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            if batch.get("pixel_values") is not None:
                kwargs["pixel_values"] = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            if batch.get("image_grid_thw") is not None:
                kwargs["image_grid_thw"] = batch["image_grid_thw"].to(device)

            outputs = model(**kwargs)
            loss_val = outputs.loss.item()
            if not (math.isnan(loss_val) or math.isinf(loss_val)):
                total_loss += loss_val
                count += 1
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                continue
            raise

    model.train()
    return total_loss / max(count, 1) if count > 0 else float("nan")


# ============================================================
# 主训练流程
# ============================================================
def main():
    set_seed(42)
    cfg = CONFIG

    print("=" * 60)
    print("  Qwen3-VL-8B 全参数微调")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("❌ 没有检测到 GPU！")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  GPU:        {gpu_name}")
    print(f"  显存:       {gpu_vram:.0f} GB")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  CUDA:       {torch.version.cuda}")

    if gpu_vram < 48:
        print(f"  ⚠️ 显存只有 {gpu_vram:.0f}GB，全参数微调建议 ≥48GB")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    config_path = os.path.join(cfg["output_dir"], "train_config.json")
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  配置已保存: {config_path}")

    # ---- 加载模型 ----
    print(f"\n[1/5] 加载模型: {cfg['model_path']}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数:     {total_params / 1e9:.2f}B")
    print(f"  可训练参数: {trainable_params / 1e9:.2f}B ({trainable_params/total_params*100:.1f}%)")
    print(f"  模型显存:   {get_gpu_memory_gb():.1f} GB")

    # ---- 加载处理器 ----
    print(f"\n[2/5] 加载处理器")
    processor = AutoProcessor.from_pretrained(
        cfg["model_path"],
        max_pixels=cfg["max_pixels"],
        min_pixels=cfg["min_pixels"],
    )

    # ---- 加载数据集 ----
    print(f"\n[3/5] 加载数据集")
    train_dataset = SeaDroneDataset(
        json_path=os.path.join(cfg["data_base"], cfg["train_file"]),
        base_dir=cfg["data_base"],
        processor=processor,
        max_length=cfg["max_length"],
    )
    val_dataset = SeaDroneDataset(
        json_path=os.path.join(cfg["data_base"], cfg["val_file"]),
        base_dir=cfg["data_base"],
        processor=processor,
        max_length=cfg["max_length"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ---- 优化器 & 调度器 ----
    print(f"\n[4/5] 配置优化器")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.95),
    )

    steps_per_epoch = math.ceil(len(train_loader) / cfg["gradient_accumulation_steps"])
    total_steps = steps_per_epoch * cfg["num_epochs"]
    warmup_steps = int(total_steps * cfg["warmup_ratio"])

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"  训练集样本数:    {len(train_dataset)}")
    print(f"  验证集样本数:    {len(val_dataset)}")
    print(f"  每轮步数:        {steps_per_epoch}")
    print(f"  总训练步数:      {total_steps}")
    print(f"  Warmup 步数:     {warmup_steps}")
    print(f"  等效 batch size: {cfg['batch_size'] * cfg['gradient_accumulation_steps']}")
    print(f"  学习率:          {cfg['learning_rate']}")

    # ---- 开始训练 ----
    print(f"\n[5/5] 开始训练！")
    print(f"{'='*60}")

    logger = TrainingLogger(cfg["output_dir"])

    best_val_loss = float("inf")
    global_step = 0
    start_time = time.time()
    nan_total = 0
    recent_losses = deque(maxlen=100)

    model.train()
    device = next(model.parameters()).device

    for epoch in range(cfg["num_epochs"]):
        print(f"\n{'━'*60}")
        print(f"  Epoch {epoch+1}/{cfg['num_epochs']}")
        print(f"{'━'*60}")

        epoch_loss = 0.0
        epoch_count = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            try:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
                if batch.get("pixel_values") is not None:
                    kwargs["pixel_values"] = batch["pixel_values"].to(
                        device, dtype=torch.bfloat16
                    )
                if batch.get("image_grid_thw") is not None:
                    kwargs["image_grid_thw"] = batch["image_grid_thw"].to(device)

                outputs = model(**kwargs)
                loss_value = outputs.loss.item()

                if math.isnan(loss_value) or math.isinf(loss_value):
                    nan_total += 1
                    if nan_total % 10 == 1:
                        print(f"  ⚠️ NaN/Inf loss (第{nan_total}次), 跳过")
                    optimizer.zero_grad()
                    continue

                loss = outputs.loss / cfg["gradient_accumulation_steps"]
                loss.backward()

                epoch_loss += loss_value
                epoch_count += 1
                recent_losses.append(loss_value)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ⚠️ OOM at batch {batch_idx}, 跳过")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                raise

            if (batch_idx + 1) % cfg["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg["max_grad_norm"]
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time

                if global_step % cfg["log_every"] == 0:
                    recent_avg = sum(recent_losses) / max(len(recent_losses), 1)
                    speed = elapsed / global_step
                    remaining = speed * (total_steps - global_step)
                    gpu_mem = get_gpu_memory_gb()

                    print(
                        f"  [Step {global_step:>5d}/{total_steps}] "
                        f"loss: {recent_avg:.4f} | "
                        f"lr: {current_lr:.2e} | "
                        f"显存: {gpu_mem:.1f}GB | "
                        f"速度: {speed:.1f}s/step | "
                        f"剩余≈{format_time(remaining)}"
                    )
                    logger.log_step(global_step, epoch+1, recent_avg, current_lr, elapsed)

                if global_step % cfg["eval_every"] == 0:
                    print(f"\n  ── 验证中 (最多 {cfg['eval_max_batches']} batch) ──")
                    val_loss = evaluate(model, val_loader, device, cfg["eval_max_batches"])
                    elapsed = time.time() - start_time
                    print(f"  验证 loss: {val_loss:.4f}")
                    logger.log_eval(global_step, epoch+1, val_loss, elapsed)

                    if not math.isnan(val_loss) and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_path = os.path.join(cfg["output_dir"], "best_model")
                        print(f"  ✅ 新的最佳！保存到: {save_path}")
                        model.save_pretrained(save_path, safe_serialization=True)
                        processor.save_pretrained(save_path)
                    else:
                        print(f"  未改善 (最佳: {best_val_loss:.4f})")

                    model.train()
                    print(f"  ── 验证结束，继续训练 ──\n")

                if global_step % cfg["save_every"] == 0 and global_step % cfg["eval_every"] != 0:
                    ckpt_path = os.path.join(cfg["output_dir"], f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_path, safe_serialization=True)
                    processor.save_pretrained(ckpt_path)
                    print(f"  💾 Checkpoint: {ckpt_path}")

        avg_epoch_loss = epoch_loss / max(epoch_count, 1)
        print(f"\n  Epoch {epoch+1} 完成 | 平均 loss: {avg_epoch_loss:.4f} | NaN: {nan_total}")

    # ============================================================
    # 训练完成
    # ============================================================
    total_time = time.time() - start_time

    final_path = os.path.join(cfg["output_dir"], "final_model")
    print(f"\n  保存最终模型: {final_path}")
    model.save_pretrained(final_path, safe_serialization=True)
    processor.save_pretrained(final_path)

    print(f"\n  生成训练曲线图...")
    logger.plot_curves()

    print(f"\n{'='*60}")
    print(f"  📋 训练完成摘要")
    print(f"{'='*60}")
    print(f"  总用时:           {format_time(total_time)}")
    print(f"  总训练步数:       {global_step}")
    print(f"  最佳 val loss:    {best_val_loss:.4f}")
    print(f"  NaN 总计:         {nan_total} 次")
    print(f"  峰值显存:         {get_gpu_memory_gb():.1f} GB")
    print(f"")
    print(f"  输出文件:")
    print(f"    最佳模型:   {cfg['output_dir']}/best_model/")
    print(f"    最终模型:   {cfg['output_dir']}/final_model/")
    print(f"    训练日志:   {cfg['output_dir']}/training_log.csv")
    print(f"    曲线图:     {cfg['output_dir']}/training_curves.png")
    print(f"{'='*60}")
    print(f"  ✅ 完成！用 vLLM 加载 best_model/ 推理即可。")
    print(f"{'='*60}")


if __name__ == "__main__":
    import traceback

    try:
        main()
        print("\n✅ 训练正常完成！")
    except Exception as e:
        print(f"\n❌ 训练异常退出：{e}")
        traceback.print_exc()

    # 训练完（不管成功还是崩溃）都关机
    print("\n30秒后自动关机，取消请按 Ctrl+C")
    import time
    time.sleep(30)
    os.system("shutdown -h now")