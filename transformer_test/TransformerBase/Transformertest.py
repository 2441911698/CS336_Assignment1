import torch
import torch.nn as nn
import math
import copy


# ==========================================
# 工具函数
# ==========================================
def clones(module, N):
    """复制 N 个完全相同的网络层，并放到 nn.ModuleList 中"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """
    生成一个下三角矩阵作为解码器的掩码 (防止模型偷看未来的词)
    比如 size=3:
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    attn_shape = (1, size, size)
    mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return mask == 0


# ==========================================
# 1. 基础组件 (之前的积木)
# ==========================================
class Embeddings(nn.Module):
    """词嵌入层：把词汇表里的词索引变成稠密向量"""

    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        # 乘以根号 d_model 是为了缩放，让它在加上位置编码时数值量级一致
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """位置编码层：给词汇注入位置/顺序信息"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        # PE_{(pos, 2i)} = sin(pos / 10000^{2i/d_{model}})。

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""

    def forward(self, query, key, value, mask=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # 将掩码为 0 的位置替换为极小值，Softmax 后就会变成 0
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn = ScaledDotProductAttention()

    def forward(self, q, k, v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # 为头数维度增加一个维度
        batch_size = q.size(0)

        # 线性映射并切分成多头
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        x, _ = self.attn(q, k, v, mask=mask)

        # 拼接多头并过最后的线性层
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.w_o(x)


class PositionwiseFeedForward(nn.Module):
    """前馈神经网络"""

    #放大
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.relu(self.w_1(x)))


class SublayerConnection(nn.Module):
    """残差连接与层归一化"""

    def __init__(self, size):
        super().__init__()
        self.norm = nn.LayerNorm(size)

    def forward(self, x, sublayer):
        # Pre-norm 结构
        return x + sublayer(self.norm(x))


# ==========================================
# 2. 单层结构
# ==========================================
class EncoderLayer(nn.Module):
    """编码器单层"""

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(num_heads, d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.sublayer1 = SublayerConnection(d_model)
        self.sublayer2 = SublayerConnection(d_model)

    def forward(self, x, mask):
        x = self.sublayer1(x, lambda _x: self.self_attn(_x, _x, _x, mask))
        x = self.sublayer2(x, self.ffn)
        return x


class DecoderLayer(nn.Module):
    """解码器单层"""

    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(num_heads, d_model)
        self.cross_attn = MultiHeadAttention(num_heads, d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        self.sublayer1 = SublayerConnection(d_model)
        self.sublayer2 = SublayerConnection(d_model)
        self.sublayer3 = SublayerConnection(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        # 自注意力：加了目标掩码，不看未来
        x = self.sublayer1(x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask))
        # 交叉注意力：看原文 (memory)
        x = self.sublayer2(x, lambda _x: self.cross_attn(_x, memory, memory, src_mask))
        x = self.sublayer3(x, self.ffn)
        return x


# ==========================================
# 3. 宏观模块 (堆叠多层)
# ==========================================
class Encoder(nn.Module):
    """编码器：由 N 个 EncoderLayer 堆叠而成"""

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.sublayer1.norm.normalized_shape[0])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """解码器：由 N 个 DecoderLayer 堆叠而成"""

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.sublayer1.norm.normalized_shape[0])

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class Generator(nn.Module):
    """最后的输出分类器：将向量转回词汇表上的概率分布"""

    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # 使用 log_softmax 输出对数概率，适配 CrossEntropyLoss
        return torch.log_softmax(self.proj(x), dim=-1)


# ==========================================
# 4. 最终完整架构：Transformer
# ==========================================
class Transformer(nn.Module):
    """
    完整的 Transformer 模型，串联所有组件
    """

    def __init__(self, src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, num_heads=8):
        super().__init__()

        # 编码器和解码器的单层蓝图
        c = copy.deepcopy  # 用深拷贝复用位置编码等模块
        attn = MultiHeadAttention(num_heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff)
        position = PositionalEncoding(d_model)

        # 1. 初始化编码器
        self.encoder = Encoder(EncoderLayer(d_model, num_heads, d_ff), N)
        # 2. 初始化解码器
        self.decoder = Decoder(DecoderLayer(d_model, num_heads, d_ff), N)

        # 3. 源语言输入：词嵌入 + 位置编码
        self.src_embed = nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        # 4. 目标语言输入：词嵌入 + 位置编码
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))

        # 5. 最终的生成器 (输出层)
        self.generator = Generator(d_model, tgt_vocab)

        # 6. 初始化所有参数为均匀分布 (Xavier 初始化，加速模型收敛)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        完整的前向传播过程
        """
        # 步骤 1: 源语言经过编码器
        memory = self.encoder(self.src_embed(src), src_mask)

        # 步骤 2: 目标语言 + 编码器信息，经过解码器
        decoder_output = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

        # 步骤 3: 投影到词汇表概率
        return self.generator(decoder_output)


# ==========================================
# 5. 测试运行代码 (如何使用它)
# ==========================================
if __name__ == "__main__":
    # 配置参数
    src_vocab_size = 1000  # 原文词典大小
    tgt_vocab_size = 1000  # 译文词典大小
    batch_size = 2  # 批次大小 (同时翻译 2 句话)
    seq_len = 10  # 句子长度

    # 实例化完整的模型 (为了演示快，层数 N 和维度调小一点)
    model = Transformer(
        src_vocab=src_vocab_size,
        tgt_vocab=tgt_vocab_size,
        N=2,  # 堆叠 2 层
        d_model=128,  # 向量维度 128
        d_ff=256,  # 前馈网络隐藏层 256
        num_heads=4  # 4 个注意力头
    )

    # 模拟输入数据 (全是整数的词索引)
    src_data = torch.randint(1, src_vocab_size, (batch_size, seq_len))
    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, seq_len))

    # 生成掩码 (Mask)
    # 原文掩码：通常不需要掩盖什么 (或者掩盖 padding 的 <pad> 字符)，这里设为 None
    src_mask = torch.ones((1, 1, seq_len))
    # 译文掩码：下三角矩阵，必须掩盖掉未来的词
    tgt_mask = subsequent_mask(seq_len)

    # 运行一次前向传播
    output = model(src_data, tgt_data, src_mask, tgt_mask)

    print("模型组装成功！")
    print(f"输入源语言形状 (Batch, SeqLen): {src_data.shape}")
    print(f"输入目标语言形状 (Batch, SeqLen): {tgt_data.shape}")
    print(f"模型输出预测形状 (Batch, SeqLen, Tgt_Vocab_Size): {output.shape}")
    # 输出的形状应为 [2, 10, 1000]
    # 代表: 2句话，每句话10个词，每个位置都输出了 1000 个词汇的概率对数。
