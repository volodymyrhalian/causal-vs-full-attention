"""Pre-LN Transformer implemented from first principles on PyTorch SDPA.

Only the attention mask changes between model variants. The weights, readout,
training objective, data and number of parameters are identical.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F
from .data import VOCAB_SIZE


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = 16
    d_model: int = 48
    num_heads: int = 4
    num_layers: int = 2
    d_ff: int = 96
    dropout: float = 0.0

    def validate(self) -> None:
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.max_seq_len < 9:
            raise ValueError("max_seq_len must be at least 9")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")


def attention_visibility(length: int, causal: bool, *, device=None) -> torch.BoolTensor:
    """True means a query may attend to that key position."""
    if causal:
        return torch.ones(length, length, dtype=torch.bool, device=device).tril()
    return torch.ones(length, length, dtype=torch.bool, device=device)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, config.d_model * 3)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def forward(self, x: torch.Tensor, *, causal: bool, inspect: bool = False):
        batch, length, width = x.shape
        qkv = self.qkv(x).reshape(batch, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        context = F.scaled_dot_product_attention(
            q, k, v, is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attn = None
        if inspect:
            # Diagnostic-only attention weights; not proof of mechanistic causality.
            scores = q @ k.transpose(-2, -1) * (self.head_dim ** -0.5)
            visible = attention_visibility(length, causal, device=x.device)
            scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
        return self.proj(context.transpose(1, 2).reshape(batch, length, width)), attn


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, *, causal: bool, inspect: bool = False):
        attended, weights = self.attn(self.ln1(x), causal=causal, inspect=inspect)
        x = x + attended
        x = x + self.ff(self.ln2(x))
        return x, weights


class ContextTransformer(nn.Module):
    def __init__(self, config: ModelConfig, mode: str):
        super().__init__()
        config.validate()
        if mode not in ("causal", "full"):
            raise ValueError("mode must be causal or full")
        self.config = config
        self.mode = mode
        self.tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.positions = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_ln = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, 2)

    def forward(self, input_ids: torch.LongTensor, query_positions: torch.LongTensor, *, inspect=False):
        batch, length = input_ids.shape
        if length > self.config.max_seq_len:
            raise ValueError("sequence longer than max_seq_len")
        pos = torch.arange(length, device=input_ids.device)
        x = self.tokens(input_ids) + self.positions(pos)[None, :, :]
        attention_maps = []
        for block in self.blocks:
            x, weights = block(x, causal=(self.mode == "causal"), inspect=inspect)
            if inspect:
                attention_maps.append(weights)
        x = self.final_ln(x)
        # Critical: use only the embedding at the QUERY position. Pooling across
        # all positions would leak future information into the causal baseline.
        context = x[torch.arange(batch, device=x.device), query_positions]
        logits = self.classifier(context)
        return (logits, attention_maps) if inspect else logits

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def metadata(self) -> dict:
        return {"architecture": asdict(self.config), "mode": self.mode,
                "parameter_count": self.parameter_count()}
