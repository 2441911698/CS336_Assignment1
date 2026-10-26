import torch
from torch.optim import Optimizer


class AdamWRef(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                # PyTorch格式,罕见的优化器在走一步之前需要重新算一遍loss,Adam完全不需要这个
        for group in self.param_groups:
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state.setdefault(
                    p,
                    {
                        "step": 0,
                        "exp_avg": torch.zeros_like(p),
                        "exp_avg_sq": torch.zeros_like(p),
                    },
                )
                st["step"] += 1
                t = st["step"]
                m, v = st["exp_avg"], st["exp_avg_sq"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1**t)
                v_hat = v / (1 - b2**t)
                p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)
                # 解耦，AdamW 的 W
                if wd:
                    p.add_(p, alpha=-lr * wd)
        return loss