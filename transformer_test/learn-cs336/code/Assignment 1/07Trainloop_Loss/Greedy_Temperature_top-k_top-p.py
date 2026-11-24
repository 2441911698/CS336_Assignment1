import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
@torch.no_grad()
def apply_repetition_penalty(logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float):
    """对 batch 中已生成 token 的 logits 进行惩罚（HF 风格，简化版：对整段历史去重后惩罚）。"""
    if penalty == 1.0:
        return logits
    # logits: (B, V); generated_ids: (B, L)
    for b in range(logits.size(0)):
        for tid in set(generated_ids[b].tolist()):
            if logits[b, tid] > 0:
                logits[b, tid] /= penalty
            else:
                logits[b, tid] *= penalty
    return logits


@torch.no_grad()
def top_k_filter(logits: torch.Tensor, k: int):
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, min(k, logits.size(-1)), dim=-1)
    min_v = v[..., -1, None]
    return torch.where(logits < min_v, torch.full_like(logits, float("-inf")), logits)


@torch.no_grad()
def top_p_filter(logits: torch.Tensor, p: float):
    """Nucleus sampling：按降序累加概率至达到 p；将尾部置为 -inf，再在原始索引空间还原。"""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(probs, dim=-1)
    # 保留 cumsum - probs <= p 的位置（首次超过 p 的 token 仍保留在核内，常见实现）
    mask = cumsum - probs > p
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    full = torch.full_like(logits, float("-inf"))
    full.scatter_(-1, sorted_idx, sorted_logits)
    return full


@torch.no_grad()
def greedy_decode(model: nn.Module, prompt_ids: torch.Tensor, max_new_tokens: int):
    ids = prompt_ids
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]
        next_id = logits.argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
    return ids


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
):
    ids = prompt_ids
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]
        logits = apply_repetition_penalty(logits.clone(), ids, repetition_penalty)
        logits = logits / max(temperature, 1e-6)
        logits = top_k_filter(logits, top_k)
        logits = top_p_filter(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids