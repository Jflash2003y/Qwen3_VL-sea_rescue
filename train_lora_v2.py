"""
==========================================================================
Qwen3-VL-8B QLoRA 微调训练脚本 V2（修复 NaN 问题）
任务：海上无人机航拍图像中的目标检测（水中人员、船只、救生设备等）
==========================================================================

V2 修复内容：
1. compute_dtype: float16 → bfloat16（RTX 5090 支持，不容易溢出）
2. 学习率: 2e-5 → 1e-5（更稳定）
3. 梯度裁剪: max_norm 1.0 → 0.5（更保守）
4. 新增 NaN 检测机制（遇到 NaN loss 自动跳过该 batch）
5. 新增 NaN 连续计数，超过阈值自动停止训练（防止白跑）
6. 新增 loss 滑动窗口日志（更直观看趋势）
"""

import os
import json
import torch
import math
import time
from PIL import Image
from collections import deque
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

# ==========================================================================
# 一、训练超参数配置
# ==========================================================================

CONFIG = {
    # --- 路径 ---
    "model_path": "/root/autodl-tmp/qwen_vl/models/Qwen/Qwen3-VL-8B-Instruct",
    "data_base": "/root/autodl-tmp/qwen_vl/finetune_data",
    "output_dir": "/root/autodl-tmp/qwen_vl/lora_output_v2",

    # --- QLoRA 量化参数 ---
    "use_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",   # ★ 修复1：float16 → bfloat16
    "use_double_quant": True,

    # --- LoRA 参数 ---
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,

    # --- 训练参数 ---
    "learning_rate": 1e-5,                   # ★ 修复2：2e-5 → 1e-5
    "num_epochs": 3,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "warmup_steps": 100,
    "max_grad_norm": 0.5,                    # ★ 修复3：1.0 → 0.5
    "max_pixels": 384 * 384,
    "min_pixels": 196 * 196,
    "max_length": 1024,

    # --- NaN 保护 ---
    "max_consecutive_nan": 50,               # ★ 修复4：连续50次NaN自动停止

    # --- 日志与保存 ---
    "log_every": 20,
    "eval_every": 500,
    "save_every": 500,
}


# ==========================================================================
# 二、自定义数据集类
# ==========================================================================

class SeaDroneDataset(Dataset):
    def __init__(self, json_path, base_dir, processor, max_pixels, min_pixels, max_length):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.base_dir = base_dir
        self.processor = processor
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        messages = sample["messages"]

        img_relative_path = messages[0]["content"][0]["image"]
        img_absolute_path = os.path.join(self.base_dir, img_relative_path)

        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{img_absolute_path}"},
                    {"type": "text", "text": messages[0]["content"][1]["text"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": messages[1]["content"][0]["text"]},
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

        assistant_text = messages[1]["content"][0]["text"]
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
                labels[start_idx:non_pad_indices[-1] + 1] = input_ids[start_idx:non_pad_indices[-1] + 1]

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs.get("pixel_values"),
            "image_grid_thw": inputs.get("image_grid_thw"),
            "labels": labels,
        }


def custom_collate_fn(batch):
    result = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }
    if batch[0]["pixel_values"] is not None:
        result["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], dim=0)
    if batch[0]["image_grid_thw"] is not None:
        result["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], dim=0)
    return result


# ==========================================================================
# 三、验证函数
# ==========================================================================

@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0
    count = 0

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break

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

        # ★ 验证时也跳过 NaN
        if not math.isnan(loss_val) and not math.isinf(loss_val):
            total_loss += loss_val
            count += 1

    model.train()
    if count == 0:
        return float("nan")
    return total_loss / count


# ==========================================================================
# 四、主训练流程
# ==========================================================================

def main():
    print("=" * 60)
    print("Qwen3-VL-8B QLoRA 微调训练 V2（NaN修复版）")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 4.1 用 4bit 量化加载模型
    # ------------------------------------------------------------------
    print("\n[1/6] 用 4bit 量化加载模型...")

    # ★ 修复核心：使用 bfloat16 计算，避免 float16 溢出
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=CONFIG["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=torch.bfloat16,      # ★ 关键修改
        bnb_4bit_use_double_quant=CONFIG["use_double_quant"],
    )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CONFIG["model_path"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # ------------------------------------------------------------------
    # 4.2 加载处理器
    # ------------------------------------------------------------------
    print("\n[2/6] 加载处理器...")
    processor = AutoProcessor.from_pretrained(
        CONFIG["model_path"],
        max_pixels=CONFIG["max_pixels"],
        min_pixels=CONFIG["min_pixels"],
    )

    # ------------------------------------------------------------------
    # 4.3 配置 LoRA
    # ------------------------------------------------------------------
    print("\n[3/6] 配置 LoRA...")
    lora_config = LoraConfig(
        r=CONFIG["lora_rank"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 4.4 准备数据集
    # ------------------------------------------------------------------
    print("\n[4/6] 加载数据集...")

    train_dataset = SeaDroneDataset(
        json_path=os.path.join(CONFIG["data_base"], "train.json"),
        base_dir=CONFIG["data_base"],
        processor=processor,
        max_pixels=CONFIG["max_pixels"],
        min_pixels=CONFIG["min_pixels"],
        max_length=CONFIG["max_length"],
    )

    val_dataset = SeaDroneDataset(
        json_path=os.path.join(CONFIG["data_base"], "val.json"),
        base_dir=CONFIG["data_base"],
        processor=processor,
        max_pixels=CONFIG["max_pixels"],
        min_pixels=CONFIG["min_pixels"],
        max_length=CONFIG["max_length"],
    )

    print(f"  训练集: {len(train_dataset)} 条样本")
    print(f"  验证集: {len(val_dataset)} 条样本")

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # 4.5 配置优化器和学习率调度器
    # ------------------------------------------------------------------
    print("\n[5/6] 配置优化器...")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=0.01,
    )

    steps_per_epoch = math.ceil(len(train_loader) / CONFIG["gradient_accumulation_steps"])
    total_steps = steps_per_epoch * CONFIG["num_epochs"]
    print(f"  每轮步数: {steps_per_epoch}")
    print(f"  总训练步数: {total_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG["warmup_steps"],
        num_training_steps=total_steps,
    )

    # ------------------------------------------------------------------
    # 4.6 开始训练
    # ------------------------------------------------------------------
    print("\n[6/6] 开始训练！")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  量化方式: 4bit NF4 + bfloat16 计算（QLoRA）")
    print(f"  等效 batch size: {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"  学习率: {CONFIG['learning_rate']}")
    print(f"  梯度裁剪: {CONFIG['max_grad_norm']}")
    print(f"  LoRA rank: {CONFIG['lora_rank']}")
    print(f"  训练轮数: {CONFIG['num_epochs']}")
    print("=" * 60)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0
    start_time = time.time()
    nan_consecutive = 0          # ★ NaN 连续计数器
    nan_total = 0                # ★ NaN 总计数
    recent_losses = deque(maxlen=100)  # ★ 最近100步loss滑动窗口

    model.train()
    device = next(model.parameters()).device

    training_aborted = False

    for epoch in range(CONFIG["num_epochs"]):
        if training_aborted:
            break

        print(f"\n{'='*60}")
        print(f"第 {epoch + 1}/{CONFIG['num_epochs']} 轮")
        print(f"{'='*60}")

        epoch_loss = 0
        epoch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            if training_aborted:
                break

            try:
                # --- 前向传播 ---
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
                loss_value = outputs.loss.item()

                # ★ 修复5：NaN / Inf 检测，跳过坏 batch
                if math.isnan(loss_value) or math.isinf(loss_value):
                    nan_consecutive += 1
                    nan_total += 1
                    print(f"  ⚠️ 步 {batch_idx}: loss={loss_value}, 跳过 (连续NaN: {nan_consecutive}, 总计: {nan_total})")
                    optimizer.zero_grad()

                    # ★ 修复6：连续NaN过多，自动停止
                    if nan_consecutive >= CONFIG["max_consecutive_nan"]:
                        print(f"\n  ❌ 连续 {nan_consecutive} 次 NaN，训练自动终止！")
                        print(f"  请检查数据或降低学习率。")
                        training_aborted = True
                        break
                    continue

                # NaN 检测通过，重置连续计数
                nan_consecutive = 0

                loss = outputs.loss / CONFIG["gradient_accumulation_steps"]

                # --- 反向传播 ---
                loss.backward()

                epoch_loss += loss_value
                epoch_count += 1
                recent_losses.append(loss_value)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ⚠️ 跳过样本 {batch_idx}（显存不足）")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue
                else:
                    raise e

            # --- 梯度累积后更新参数 ---
            if (batch_idx + 1) % CONFIG["gradient_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=CONFIG["max_grad_norm"]    # ★ 使用更保守的裁剪
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_step += 1

                # --- 打印训练日志 ---
                if global_step % CONFIG["log_every"] == 0:
                    avg_loss = epoch_loss / max(epoch_count, 1)
                    recent_avg = sum(recent_losses) / max(len(recent_losses), 1)
                    elapsed = time.time() - start_time
                    lr = scheduler.get_last_lr()[0]
                    gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
                    print(
                        f"  步数 {global_step}/{total_steps} | "
                        f"loss: {avg_loss:.4f} | "
                        f"近期loss: {recent_avg:.4f} | "
                        f"lr: {lr:.2e} | "
                        f"显存: {gpu_mem:.1f}GB | "
                        f"NaN总计: {nan_total} | "
                        f"时间: {elapsed/60:.1f}分钟"
                    )

                # --- 验证 + 保存 ---
                if global_step % CONFIG["eval_every"] == 0:
                    print(f"\n  >>> 验证中...")
                    val_loss = evaluate(model, val_loader, device)
                    print(f"  >>> 验证 loss: {val_loss:.4f}")

                    if not math.isnan(val_loss) and val_loss < best_val_loss:
                        best_val_loss = val_loss
                        save_path = os.path.join(CONFIG["output_dir"], "best_model")
                        model.save_pretrained(save_path)
                        processor.save_pretrained(save_path)
                        print(f"  >>> ✅ 最佳模型已保存: {save_path}")
                        print(f"  >>> 最佳验证 loss: {best_val_loss:.4f}")
                    else:
                        print(f"  >>> 验证 loss 没有改善 (最佳: {best_val_loss:.4f})")

                    model.train()

                # --- 定期保存 checkpoint ---
                if global_step % CONFIG["save_every"] == 0:
                    ckpt_path = os.path.join(CONFIG["output_dir"], f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_path)
                    print(f"  >>> Checkpoint 已保存: {ckpt_path}")

        avg_epoch_loss = epoch_loss / max(epoch_count, 1)
        print(f"\n第 {epoch+1} 轮完成 | 平均 loss: {avg_epoch_loss:.4f} | NaN总计: {nan_total}")

    # ------------------------------------------------------------------
    # 训练完成
    # ------------------------------------------------------------------
    total_time = (time.time() - start_time) / 60
    print(f"\n{'='*60}")
    if training_aborted:
        print("训练被提前终止（NaN过多）")
    else:
        print("训练完成！")
    print(f"总用时: {total_time:.1f} 分钟")
    print(f"最佳验证 loss: {best_val_loss:.4f}")
    print(f"NaN 总计: {nan_total} 次")
    print(f"模型保存在: {CONFIG['output_dir']}")
    print(f"{'='*60}")

    final_path = os.path.join(CONFIG["output_dir"], "final_model")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"最终模型已保存到: {final_path}")


if __name__ == "__main__":
    main()