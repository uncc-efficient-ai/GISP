#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""LoRA adaptation helper for the main-paper Table 8 experiment."""

import os
import math
import importlib
import argparse
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

import torch
from torch.utils.data import Dataset
import wandb
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed,
)

from peft import LoraConfig, get_peft_model, TaskType


# ---------- utils ----------

def _is_linear_like(m: torch.nn.Module) -> bool:
    # robust enough for most HF LLaMA variants / pruned variants
    if isinstance(m, torch.nn.Linear):
        return True
    name = m.__class__.__name__.lower()
    if "linear" in name and hasattr(m, "weight"):
        try:
            return isinstance(m.weight, torch.Tensor) and m.weight.dim() == 2
        except Exception:
            return False
    return False


def infer_lora_target_modules(model, requested: str = "auto") -> List[str]:
    """
    For structured-pruned models, some projections might be missing.
    If requested == "auto", we only select module-name suffixes that actually exist.
    """
    if requested and requested != "auto":
        return [x.strip() for x in requested.split(",") if x.strip()]

    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    exist = set()
    for name, module in model.named_modules():
        if not _is_linear_like(module):
            continue
        suffix = name.split(".")[-1]
        if suffix in candidates:
            exist.add(suffix)

    chosen = [m for m in candidates if m in exist]
    if not chosen:
        raise RuntimeError(
            "infer_lora_target_modules(auto) 找不到任何候选模块。\n"
            "请打印 model.named_modules() 看一下你的结构化剪枝模型线性层名字，"
            "然后用 --lora_target_modules 手动指定（例如: proj1,proj2,...）。"
        )
    return chosen


def trim_trailing_pad(input_ids: torch.Tensor, labels: torch.Tensor, pad_id: int) -> (torch.Tensor, torch.Tensor):
    """
    只裁掉末尾连续 pad_id（不改中间内容）。
    你的 get_cmqa_no_pad 里会把 pad 放在末尾，所以这样能显著省算力。
    """
    if pad_id is None:
        return input_ids, labels
    if input_ids.numel() == 0:
        return input_ids, labels
    # find last non-pad
    i = input_ids.numel()
    while i > 0 and int(input_ids[i - 1]) == int(pad_id):
        i -= 1
    return input_ids[:i], labels[:i]


# ---------- dataset / collator ----------

class PositiveOnlyDataset(Dataset):
    """
    Wraps the list from get_cmqa_no_pad and only keeps is_positive=True examples.
    Each item returns dict(input_ids, labels). attention_mask is created in collator.
    """
    def __init__(
        self,
        triples: List[Any],
        pad_token_id: int,
        trim_pad: bool = True,
        max_samples: int = -1,
    ):
        self.data: List[Dict[str, torch.Tensor]] = []
        kept = 0
        for item in triples:
            # expected: (inp, labels, is_positive)
            if len(item) != 3:
                raise ValueError("get_cmqa_no_pad 的 trainloader 元素应为 (inp, labels, is_positive) 三元组。")
            inp, lab, is_pos = item
            if not bool(is_pos):
                continue

            # inp/lab are [1, L] in your code -> squeeze to [L]
            if inp.dim() == 2 and inp.size(0) == 1:
                inp = inp.squeeze(0)
            if lab.dim() == 2 and lab.size(0) == 1:
                lab = lab.squeeze(0)

            if trim_pad:
                inp, lab = trim_trailing_pad(inp, lab, pad_token_id)

            self.data.append({"input_ids": inp.to(torch.long), "labels": lab.to(torch.long)})
            kept += 1
            if max_samples > 0 and kept >= max_samples:
                break

        if len(self.data) == 0:
            raise RuntimeError(
                "正样本数据为 0：可能 total_budget 太小，或你的 loader 没有 is_positive=True。"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]


@dataclass
class CausalLMCollator:
    pad_token_id: int
    pad_to_multiple_of: Optional[int] = 8

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # features: [{"input_ids": [L], "labels":[L]}, ...]
        bs = len(features)
        lengths = [int(f["input_ids"].numel()) for f in features]
        max_len = max(lengths)

        if self.pad_to_multiple_of:
            max_len = int(math.ceil(max_len / self.pad_to_multiple_of) * self.pad_to_multiple_of)

        input_ids = torch.full((bs, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((bs, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((bs, max_len), dtype=torch.long)

        for i, f in enumerate(features):
            x = f["input_ids"]
            y = f["labels"]
            L = int(x.numel())
            input_ids[i, :L] = x
            labels[i, :L] = y
            attention_mask[i, :L] = 1

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


# ---------- main ----------

def main(model, tokenizer, save_path, argv):
    parser = argparse.ArgumentParser()

    # model / tokenizer
    # your data module
    parser.add_argument("--data_module", type=str, required=True,
                        help="包含 get_cmqa_no_pad 的 Python 模块路径，例如: modules.data.cmqa_loader")
    parser.add_argument("--data_fn", type=str, default="get_cmqa_no_pad",
                        help="默认调用 get_cmqa_no_pad；如果你函数名不同可改。")
    parser.add_argument("--total_budget", type=int, default=2_000_000,
                        help="传给 get_cmqa_no_pad 的 total_budget（token 预算）。想用更多数据就调大。")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_samples", type=int, default=-1,
                        help="只为控制规模：最多保留多少条正样本（-1 表示不限制）。")
    parser.add_argument("--no_trim_pad", action="store_true",
                        help="默认会裁掉末尾 pad 省算力；加这个参数则不裁。")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", type=str, default="auto",
                        help="auto 或手动逗号分隔，如 q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # train
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=40)
    parser.add_argument("--save_strategy", type=str, default="steps")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--group_by_length", action="store_true",
                        help="可选：让 Trainer 按长度分桶，混合不同 seqlen 的任务时更省显存。")

    args = parser.parse_args(argv)
    set_seed(args.seed)
    bf16=True
    fp16=False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        # LoRA + gradient checkpointing 常见需要这一句（否则可能报 requires_grad 的错误）
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    # LoRA attach (auto-detect targets for structured-pruned model)
    target_modules = infer_lora_target_modules(model, requested=args.lora_target_modules)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # import your data code and call get_cmqa_no_pad
    dm = importlib.import_module(args.data_module)
    fn = getattr(dm, args.data_fn)
    # expected return: (trainloader, valencs)
    trainloader, _valencs = fn(args.total_budget, args.seed, tokenizer)

    train_ds = PositiveOnlyDataset(
        triples=trainloader,
        pad_token_id=tokenizer.pad_token_id,
        trim_pad=(not args.no_trim_pad),
        max_samples=args.max_train_samples,
    )

    collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id, pad_to_multiple_of=8)

    import inspect

    ta_sig = inspect.signature(TrainingArguments.__init__)
    extra = {}

    if "dataloader_num_workers" in ta_sig.parameters:
        extra["dataloader_num_workers"] = 4  # 先从 4 试起，CPU 核多可到 8
    if "dataloader_pin_memory" in ta_sig.parameters:
        extra["dataloader_pin_memory"] = True
    if "dataloader_persistent_workers" in ta_sig.parameters:
        extra["dataloader_persistent_workers"] = True
    if "dataloader_prefetch_factor" in ta_sig.parameters:
        extra["dataloader_prefetch_factor"] = 2  # worker 预取

    # （可选）更快的 fused AdamW（版本/torch 够新才支持）
    if "optim" in ta_sig.parameters:
        extra["optim"] = "adamw_torch_fused"

    train_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=bf16,
        fp16=fp16,
        report_to="wandb",
        remove_unused_columns=False,
        group_by_length=args.group_by_length,
        **extra,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

    print("[OK] done.")
    return model
