import math
from typing import Iterable, List, Dict, Any, Optional, Union

import torch
import torch.nn as nn


Number = Union[float, int]
ParamGroup = List[Dict[str, Any]]


class AdamW:
    """
    纯 PyTorch 张量实现 AdamW，不依赖 torch.optim.AdamW。
    更新: θ ← θ - η * ( m_hat / (sqrt(v_hat) + ε) + λ * θ )
    等价于先 Adam 步，再 θ ← θ - η*λ*θ（decoupled weight decay）。
    """

    def __init__(
        self,
        params: Union[Iterable[torch.nn.Parameter], ParamGroup],
        lr: Number = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.state: Dict[torch.nn.Parameter, Dict[str, Any]] = {}
        if isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
            self.param_groups = []
            for g in params:
                self.param_groups.append(
                    {
                        "params": list(g["params"]),
                        "lr": float(g.get("lr", lr)),
                        "betas": tuple(g.get("betas", betas)),
                        "eps": float(g.get("eps", eps)),
                        "weight_decay": float(g.get("weight_decay", weight_decay)),
                    }
                )
        else:
            self.param_groups = [
                {
                    "params": list(params),
                    "lr": float(lr),
                    "betas": (float(betas[0]), float(betas[1])),
                    "eps": float(eps),
                    "weight_decay": float(weight_decay),
                }
            ]

    def zero_grad(self, set_to_none: bool = False) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def _init_state(self, p: torch.nn.Parameter) -> Dict[str, Any]:
        return {
            "step": 0,
            "exp_avg": torch.zeros_like(p, memory_format=torch.preserve_format),
            "exp_avg_sq": torch.zeros_like(p, memory_format=torch.preserve_format),
        }

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[torch.Tensor]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad

                if p not in self.state:
                    self.state[p] = self._init_state(p)
                st = self.state[p]

                m, v = st["exp_avg"], st["exp_avg_sq"]
                st["step"] += 1
                t = st["step"]

                m.mul_(beta1).add_(g, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)

                m_hat = m / (1.0 - beta1**t)
                v_hat = v / (1.0 - beta2**t)

                denom = v_hat.sqrt().add_(eps)
                p.add_(m_hat.div_(denom), alpha=-lr)
                if wd != 0.0:
                    p.add_(p, alpha=-lr * wd)

        return loss


def build_optimizer(model: nn.Module, lr: float = 3e-4, wd: float = 0.1) -> AdamW:
    """二维及以上参数（多为 weight）使用 weight_decay；一维（多为 bias/LayerNorm）不衰减。"""
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            no_decay.append(p)
    return AdamW(
        [
            {"params": decay, "lr": lr, "weight_decay": wd},
            {"params": no_decay, "lr": lr, "weight_decay": 0.0},
        ]
    )