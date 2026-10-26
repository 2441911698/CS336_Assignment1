import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


def clones(module, N):
    """克隆 N 个完全相同的网络层"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ----------------------------------------------------
# 1. Scaled Dot-Product Attention (缩放点积注意力)
# ----------------------------------------------------
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    # 计算注意力得分: (Q * K^T) / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)  # 掩码操作，将无效位置填入极小值

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


# ----------------------------------------------------
# 2. Multi-Head Attention (多头注意力)
# ----------------------------------------------------
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        # 定义 4 个线性层 (分别用于 Q, K, V 和最后的输出 O)
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1. 线性投影并划分成 h 个头
        # 形状变化: (batch, seq, d_model) -> (batch, h, seq, d_k)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2. 在所有头上面应用自注意力
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3. 拼接 (Concat) 并通过最后一个线性层
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


# ----------------------------------------------------
# 3. Position-wise Feed-Forward Networks (前馈神经网络)
# ----------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


# ----------------------------------------------------
# 4. Layer Norm & Sublayer Connection (层归一化与残差连接)
# ----------------------------------------------------
class SublayerConnection(nn.Module):
    """
    残差连接加上层归一化: x + Sublayer(LayerNorm(x))
    注意：为了代码稳定性，这里采用了 Pre-Norm 的结构（在输入子层前做归一化）
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # 传入一个子层函数 (如 Attention 或 FFN)
        return x + self.dropout(sublayer(self.norm(x)))