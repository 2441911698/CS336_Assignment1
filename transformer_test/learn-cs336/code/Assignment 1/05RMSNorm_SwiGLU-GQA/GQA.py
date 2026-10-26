import torch
import torch.nn as nn


class GQAProjection(nn.Module):
    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int):
        super().__init__()
        # 确保 Q 的头数能被 K/V 的头数整除，比如 32个Q头，8个KV头，比例就是 4
        assert n_q_heads % n_kv_heads == 0
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_q_heads // n_kv_heads  # 复制倍数 (Group size)

        # 单头维度
        self.d_head = d_model // n_q_heads

        # Q 保持原样，K 和 V 的输出维度大幅缩减
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_head * n_kv_heads, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_head * n_kv_heads, bias=False)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        # 1. 线性投影拿到原始的 q, k, v
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. 拆分多头空间，并将 head 维度移到第 1 维 (Batch, Head, SeqLen, HeadDim)
        q = q.view(B, T, self.n_q_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # 3. 核心动作：在 Head 维度上，把 K 和 V 复制（repeat_interleave）对齐 Q
        # 例如 n_rep = 4，原来的头 0 会变成 0,0,0,0，头 1 变成 1,1,1,1
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        # 此时 q, k, v 的形状完全一致，可以直接进行标准的 Attention 计算
        # 后续就是 QK^T / sqrt(d) -> softmax -> 乘 V
        return q, k, v