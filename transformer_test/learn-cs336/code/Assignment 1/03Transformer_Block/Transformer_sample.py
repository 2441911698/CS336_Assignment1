import math
import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """
    单塔 Decoder Block（Pre-Norm）：
    x -> LN -> MHA(causal) -> + -> LN -> FFN -> +
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        self.drop = nn.Dropout(dropout)

    def _causal_mask(self, n: int, device: torch.device) -> torch.Tensor:
        """上三角（不含对角）为 True：这些位置在 softmax 前被置为 -inf。"""
        return torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, d_model]
        b, n, d = x.shape
        assert d == self.d_model

        # Pre-Norm + Multi-Head Causal Self-Attention
        h = self.ln1(x)
        qkv = self.qkv(h).chunk(3, dim=-1)
        q, k, v = qkv

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(b, n, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = map(split_heads, (q, k, v))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = self._causal_mask(n, x.device)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.drop(attn)

        y = torch.matmul(attn, v)
        y = y.transpose(1, 2).contiguous().view(b, n, d)
        y = self.out_proj(y)
        y = self.drop(y)
        x = x + y

        # Pre-Norm + FFN
        h2 = self.ln2(x)
        z = self.drop(self.ffn(h2))
        x = x + z
        return x


class TinyDecoderLM(nn.Module):
    """Embedding + L 层 Block + Final LN + LM Head"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int,
        max_pos: int = 2048,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_pos, d_model)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        b, n = token_ids.shape
        pos = torch.arange(n, device=token_ids.device).unsqueeze(0).expand(b, n)
        x = self.tok_emb(token_ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)