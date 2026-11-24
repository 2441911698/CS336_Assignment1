import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def cross_entropy_lm_loss(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100):
    """logits: (B, T, V), labels: (B, T)"""
    b, t, v = logits.shape
    return F.cross_entropy(logits.reshape(-1, v), labels.reshape(-1), ignore_index=ignore_index)


def perplexity_from_mean_nll(mean_nll: float) -> float:
    return float(math.exp(mean_nll))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    grad_accum_steps: int = 1,
    max_grad_norm: Optional[float] = 1.0,
    log_interval: int = 10,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss_sum = 0.0
    n_tokens = 0
    t0 = time.time()
    micro_step = 0
    global_step = 0

    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = cross_entropy_lm_loss(logits, y)
        loss = loss / grad_accum_steps
        loss.backward()

        with torch.no_grad():
            total_loss_sum += loss.item() * grad_accum_steps * x.numel()
            n_tokens += x.numel()

        micro_step += 1
        if micro_step % grad_accum_steps == 0:
            grad_norm = None
            if max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % log_interval == 0:
                elapsed = time.time() - t0
                tok_per_sec = n_tokens / max(elapsed, 1e-6)
                lr = optimizer.param_groups[0]["lr"]
                avg_nll = total_loss_sum / max(n_tokens, 1)
                ppl = perplexity_from_mean_nll(avg_nll)
                msg = (
                    f"step={global_step} loss={avg_nll:.4f} ppl={ppl:.2f} "
                    f"lr={lr:.2e} tok/s={tok_per_sec:.0f}"
                )
                if grad_norm is not None:
                    msg += f" grad_norm={grad_norm:.2f}"
                print(msg)

    return total_loss_sum / max(n_tokens, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total, ntok = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = cross_entropy_lm_loss(logits, y)
        total += loss.item() * x.numel()
        ntok += x.numel()
    mean_nll = total / max(ntok, 1)
    return mean_nll, perplexity_from_mean_nll(mean_nll)