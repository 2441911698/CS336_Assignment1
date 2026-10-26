import torch
import torch.nn as nn
import math


class SublayerConnection(nn.Module):
    """残差连接与层归一化 (Pre-Norm 结构)"""

    def __init__(self, size):
        super().__init__()
        self.norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        # 先归一化，再进入子层处理，最后与原输入相加 (残差连接)
        return x + sublayer(self.norm(x))


class PositionalEncoding(nn.Module):
    """位置编码: 为序列注入位置信息"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建位置编码矩阵 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数维度使用正弦，奇数维度使用余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # 将位置编码直接加到词嵌入向量上
        return x + self.pe[:, :x.size(1)]