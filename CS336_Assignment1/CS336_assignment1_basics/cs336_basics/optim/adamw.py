import torch
import torch.nn as nn
import math
from typing import Iterable

class AdamW:
    def __init__(
            self,
            params: Iterable[torch.Tensor],
            lr: float = 1e-3,
            betas: tuple = (0.9,0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01
                 ):
        self.param_groups = [{
            "params": list(params),
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay
        }]

        self.state = {}


    def zero_grad(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad.zero_()


    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is not None:
                    grad = p.grad.data
                else:
                    continue

                if p not in self.state:
                    self.state[p] = {
                        "step": 0,
                        "m": torch.zeros_like(p.data),
                        "v": torch.zeros_like(p.data)
                    }

                state = self.state[p]
                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]

                p.data.mul_(1.0 - lr * weight_decay)
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)  # m = beta1 * m + (1 - beta1) * grad
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)  # v = beta2 * v + (1 - beta2) * grad^2

                bias1 = 1.0 - beta1 ** t
                bias2 = 1.0 - beta2 ** t

                step_size = lr / bias1
                bias2_sqrt = math.sqrt(bias2)
                denom = (v.sqrt() / bias2_sqrt).add_(eps)
                p.data.addcdiv_(m, denom, value=-step_size) # p = p - lr * (m / bias1) / denom

