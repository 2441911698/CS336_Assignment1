"""
LLaMA-style Transformer language model (PyTorch).
LLaMA 风格的 Transformer 语言模型（PyTorch 实现）基础框架。
"""

from __future__ import annotations
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. 辅助与初始化函数 (Helpers & Initialization)
# =====================================================================

def _ffn_hidden_dim(d_model: int) -> int:
    """计算 SwiGLU 前馈网络隐藏层的确切维度 (按一定规则对齐到 256 的倍数)。"""
    raw = int(2 * 4 * d_model // 3)
    return ((raw + 255) // 256 ) * 256 # 向上取整到 256 的整数倍以使用

def init_linear(module: nn.Linear) -> None:
    """对普通线性层进行 Xavier 均匀初始化。"""
    nn.init.xavier_uniform_(module.weight, gain=1.0)
    if module.bias is not None:
        nn.init.zeros_(module.bias)

def init_embedding(module: nn.Embedding) -> None:
    """对词嵌入层进行正态分布初始化。"""
    nn.init.normal_(module.weight, mean=0.0, std=0.02)

def init_kaiming_linear(module: nn.Linear) -> None:
    """对用于 GLU 门控的线性层进行 Kaiming (He) 均匀初始化。"""
    nn.init.kaiming_uniform(module.weight, a=math.sqrt(5))
    if module.bias is not None:
        nn.init.zeros_(module.bias)
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将张量最后维度对半切分并交错旋转，用于 RoPE 位置编码。"""
    d = x.shape[-1] // 2
    x1, x2 = x[...,:d],x[...,d:]
    return torch.cat((-x2,x1),dim=-1)

def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """将预计算的旋转位置编码 (RoPE) 注入到查询 (Q) 和键 (K) 张量中。"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin) # x·cos - y·sin, y·cos + x·sin
    return q_embed, k_embed

# =====================================================================
# 2. 核心网络组件 (Core Network Components)
# =====================================================================

class RMSNorm(nn.Module):
    """
    均方根层归一化 (Root Mean Square Layer Normalization)。
    相比传统 LayerNorm，去除了计算均值的步骤，计算更高效，是 LLaMA 的标准配置。
    """
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        # TODO: 注册缩放参数 (weight)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 计算均方根，归一化输入，并乘以缩放参数
        dtype = x.dtype
        x_f = x.float()
        rms = torch.rsqrt(x_f.pow(2).mean(-1,keepdim=True) + self.eps)
        return (x_f * rms).to(dtype) * self.weight


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE)。
    预先计算并缓存不同位置和维度的频率信息 (cos 和 sin)，供注意力机制调用。
    """
    def __init__(self, dim: int, max_position_embeddings: int = 8192, base: float = 10000.0) -> None:
        super().__init__()
        # TODO: 预计算逆频率并注册为 buffer
        self.dim = dim # 注意力头维度大小
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据当前序列长度，返回对应的 cos 和 sin 张量，以便广播到 Q 和 K 上。"""
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t,self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        return cos, sin
class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制 (Multi-head attention)。
    集成了 RoPE 位置编码、因果掩码 (Causal Mask，防止看到未来信息)，
    并支持分组查询注意力 (GQA - Grouped Query Attention) 以节省显存。
    """
    def __init__(
        self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None,
        max_position_embeddings: int = 8192, rope_base: float = 10000.0, attn_dropout: float = 0.0
    ) -> None:
        super().__init__()
        # TODO: 初始化 Q, K, V 投影层，输出投影层，RoPE 模块以及 Dropout
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = n_heads // n_kv_heads # 小组共享KV
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model,self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.rotary = RotaryEmbedding(self.head_dim, max_position_embeddings=max_position_embeddings, base=rope_base,)
        self.attn_dropout = nn.Dropout(attn_dropout) #注意力丢弃层，插在Softmax计算完之后、和V相乘之前
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # TODO:
        # 1. 投影得到 Q, K, V
        # 2. 获取并应用 RoPE (cos, sin) 到 Q 和 K
        # 3. 处理 GQA 逻辑 (如果 K,V 头数少于 Q 头数，需重复 K,V)
        # 4. 计算注意力分数 (Q * K^T / sqrt(d))
        # 5. 应用因果掩码 (Causal Mask) 和传入的 attention_mask (如 Padding 掩码)
        # 6. 计算 Softmax 得到权重，与 V 相乘
        # 7. 通过输出投影层返回结果
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos,sin = self.rotary(x, seq_len = seq_len)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q, k = apply_rotary_emb(q, k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matul(q,k.transpose(-2,-1)) * scale

        casual =torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),diagonal=1,)
        attn_scores = attn_scores.masked_fill(casual, float("-inf"))

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(dtype=torch.bool)
            else:
                mask = attention_mask.to(dtype=torch.bool)
            attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = torch.matul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈神经网络 (Feed-Forward Network)。
    LLaMA 取代传统 ReLU FFN 的组件，包含门控分支和激活分支的乘积运算。
    """
    def __init__(self, d_model: int) -> None:
        super().__init__()
        # TODO: 初始化 gate_proj, up_proj, down_proj 线性层
        hidden = _ffn_hidden_dim(d_model)
        self.gate_proj = nn.Linear(d_model, hidden, bias=False)
        self.up_proj = nn.Linear(d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: 实现 down(SiLU(gate(x)) * up(x)) 的计算流
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# =====================================================================
# 3. 模型组装 (Model Assembly)
# =====================================================================

class TransformerBlock(nn.Module):
    """
    单个 Transformer 块。
    结构为：Pre-norm -> MultiHeadAttention -> 残差连接 -> Pre-norm -> SwiGLUFFN -> 残差连接。
    """
    def __init__(
        self, d_model: int, n_heads: int, n_kv_heads: Optional[int],
        max_position_embeddings: int, rope_base: float, attn_dropout: float,
    ) -> None:
        super().__init__()
        # TODO: 初始化注意力前置 Norm、注意力层、FFN 前置 Norm、FFN 层
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, n_kv_heads=n_kv_heads, max_position_embeddings=max_position_embeddings, rope_base=rope_base, attn_dropout=attn_dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # TODO: 执行前向传播并添加残差连接 (x = x + Sublayer(Norm(x)))
        x = x + self.attn(self.attn_norm(x), attention_mask=attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    """
    完整的因果语言模型 (Causal Language Model)。
    由词嵌入、多个 TransformerBlock 堆叠、最终的层归一化和语言模型头 (LM Head) 组成。
    """
    def __init__(
        self, vocab_size: int, d_model: int, n_layers: int, n_heads: int,
        n_kv_heads: Optional[int] = None, max_position_embeddings: int = 8192,
        rope_base: float = 10000.0, attn_dropout: float = 0.0, pad_token_id: Optional[int] = None,
    ) -> None:
        super().__init__()
        # TODO: 初始化词嵌入层 (embed_tokens)
        # TODO: 初始化并堆叠多个 TransformerBlock (layers)
        # TODO: 初始化最终的 RMSNorm (norm)
        # TODO: 声明 lm_head 变量 (稍后在 tie_weights 中初始化)
        # TODO: 调用自定义的 _init_weights 方法
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.n_heads = n_heads
        self.embed_tokens = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    n_kv_heads,
                    max_position_embeddings,
                    rope_base,
                    attn_dropout,
                )
                for _ in range(n_layers)
            ])
        self.norm = RMSNorm(d_model)
        self.lm_head: Optional[nn.Linear] = None
        self._init_weights()
    def tie_weights(self) -> None:
        """
        权重绑定。
        让输出层 (lm_head) 的权重直接使用词嵌入层 (embed_tokens) 的权重，减少参数量。
        """
        if self.lm_head is None:
            self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight


    def _init_weights(self) -> None:
        """遍历模型的所有子模块，根据类型 (Linear, Embedding, RMSNorm 等) 应用合适的初始化策略。"""
        init_embedding(self.embed_tokens)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith("gate_proj")or name.endswith("up_proj"):
                    init_kaiming_linear(module)
                else:
                    init_linear(module)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        接收 Token ID 序列，经过整个网络，输出预测下一个词的逻辑分数 (logits)。
        """
        # TODO:
        # 1. 确保 lm_head 已初始化
        # 2. 根据 pad_token_id 生成 attention_mask
        # 3. Token IDs -> 词嵌入 (x)
        # 4. 依次通过每一个 TransformerBlock 层
        # 5. 通过最终的层归一化
        # 6. 通过 lm_head 映射回词表大小的 logits 空间，并返回
        if self.lm_head is None:
            self.tie_weights()

        attention_mask = None
        if self.pad_token_id is not None:
            attention_mask = input_ids !=self.pad_token_id

        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.norm(x)
        assert self.lm_head is not None
        return self.lm_head(x)

    @torch.no_grad()
    def generate(
        self, input_ids: torch.Tensor, max_new_tokens: int,
        temperature: float = 1.0, top_p: float = 1.0, eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        文本生成推理循环 (Autoregressive sampling)。
        根据输入的提示序列，逐词生成后续文本。支持温度调节 (temperature) 和核采样 (top-p)。
        """
        # TODO:
        # 循环 max_new_tokens 次：
        #   1. 通过 forward 获取当前 logits
        #   2. 取最后一个时间步的 logits 并应用 temperature 缩放
        #   3. (可选) 应用 top-p 过滤截断低概率词
        #   4. 计算 Softmax 概率并采样得到 next_token
        #   5. 将 next_token 拼接到输入序列中作为下一步的输入
        #   6. 检查是否达到 eos_token_id，是则提前退出
        self.eval()
        out = input_ids
        for _ in range (max_new_tokens):
            logits = self.forward(out)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumsum = torch.cumsum(probs, dim=-1)
                mask = cumsum - probs > top_p
                sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                probs = F.softmax(sorted_logits, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(sorted_idx, -1 ,sampled)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            out = torch.cat([out, next_token], dim=-1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return  out
