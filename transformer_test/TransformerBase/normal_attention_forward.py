import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    """最底层的缩放点积注意力计算"""

    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, mask=None):
        # 获取特征维度大小，用于后续的缩放
        d_k = query.size(-1)

        # 步骤 1: Q 乘以 K的转置，并除以根号 d_k 进行缩放 (防止梯度消失)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 步骤 2: 如果存在掩码 (mask)，将需要遮蔽的位置替换为极小值 (-1e9)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 步骤 3: 经过 Softmax 将得分转化为概率权重
        p_attn = torch.softmax(scores, dim=-1)

        # 步骤 4: 用注意力权重乘以 Value，得到最终输出
        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        # 定义 4 个线性映射层 (分别给 Q, K, V 以及最后的输出合并使用)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 1. 线性映射并切分成多个头: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        # 2. 对每个头独立应用注意力机制
        x, attn = self.attention(q, k, v, mask=mask)

        # 3. 拼接所有头的结果: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

        # 4. 经过最后一个线性层融合信息
        return self.w_o(x)


class PositionwiseFeedForward(nn.Module):
    """位置前馈神经网络"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        # 先将维度放大 (通常放大 4 倍)，再压缩回原维度
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.relu(self.w_1(x)))

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
        #转成指数和对数计算，运算时速度更快，还可避免精度问题
        # 偶数维度使用正弦，奇数维度使用余弦
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # 将位置编码直接加到词嵌入向量上
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    """编码器单层"""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        # 两个子层：自注意力层和前馈层
        self.sublayer1 = SublayerConnection(d_model)
        self.sublayer2 = SublayerConnection(d_model)

    def forward(self, x, mask):
        # 1. 多头自注意力机制 (Q, K, V 均来自 x)
        x = self.sublayer1(x, lambda _x: self.self_attn(_x, _x, _x, mask))
        # 2. 前馈神经网络
        x = self.sublayer2(x, self.ffn)
        return x


class DecoderLayer(nn.Module):
    """解码器单层"""
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        # 三个子层：掩码自注意力、交叉注意力、前馈层
        self.sublayer1 = SublayerConnection(d_model)
        self.sublayer2 = SublayerConnection(d_model)
        self.sublayer3 = SublayerConnection(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 1. 带有掩码的自注意力 (防止看到未来单词)
        x = self.sublayer1(x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask))
        # 2. 交叉注意力 (Q 来自解码器本身 x，K 和 V 来自编码器的输出 memory)
        x = self.sublayer2(x, lambda _x: self.cross_attn(_x, memory, memory, src_mask))
        # 3. 前馈神经网络
        x = self.sublayer3(x, self.ffn)
        return x