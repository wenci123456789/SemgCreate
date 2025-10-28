# -*- coding: utf-8 -*-
"""
RoFormerEMG: Rotary Transformer for sEMG → 10 finger-joint regression (NinaPro DB2)
-----------------------------------------------------------------------------------
Backbone:
  Conv1d front-end → [Pre-LN Transformer + RoPE on Q/K] × L → LayerNorm → Linear(→10)

Input  shape: [B, T, C] or [B, C, T]   (C=12 for DB2; code auto-converts to [B, T, C])
Output shape: [B, T, 10]

Notes:
- Only the model is defined here (no training loop).
- Optional μ-law normalization can be enabled via use_mu_law=True.
"""

from typing import Tuple
import math
import torch
import torch.nn as nn


# ---------------- μ-law normalization (optional) ----------------
class MuLaw(nn.Module):
    def __init__(self, mu: float = 220.0, enabled: bool = False, eps: float = 1e-9):
        super().__init__()
        self.mu = float(mu)
        self.enabled = bool(enabled)
        self.eps = eps
        self.den = math.log1p(self.mu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        # works for general ranges; commonly x is already normalized
        return torch.sign(x) * torch.log1p(self.mu * torch.abs(x) + self.eps) / self.den


# ---------------- RoPE utilities ----------------
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # split last dim to even/odd
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    out = torch.stack((-x2, x1), dim=-1)
    return out.flatten(-2)


def apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, Dh]; sin/cos: [1,1,T,Dh] (broadcastable)
    return (x * cos) + (rotate_half(x) * sin)


def build_rope_cache(seq_len: int, dim: int, base: float = 10000.0,
                     device=None, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
    # Returns sin, cos: [T, Dh]; Dh must be even
    assert dim % 2 == 0, "RoPE head_dim must be even"
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=dtype) / dim))
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("t,d->td", t, inv_freq)  # [T, Dh/2]
    sin = torch.repeat_interleave(freqs.sin(), repeats=2, dim=-1)
    cos = torch.repeat_interleave(freqs.cos(), repeats=2, dim=-1)
    return sin, cos


# ---------------- RoFormer blocks ----------------
class MultiheadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, rope_sin: torch.Tensor, rope_cos: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        qkv = self.qkv(x)  # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # split heads: [B, H, T, Dh]
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # apply RoPE to q/k
        sin = rope_sin[:T, :].unsqueeze(0).unsqueeze(0)  # [1,1,T,Dh]
        cos = rope_cos[:T, :].unsqueeze(0).unsqueeze(0)
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        # scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # [B,H,T,T]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # [B,H,T,Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B,T,D]
        out = self.proj_drop(self.out(out))
        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, drop: float = 0.0, activation: str = "mish"):
        super().__init__()
        hidden = d_model * expansion
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.drop = nn.Dropout(drop)
        if activation.lower() == "mish":
            self.act = nn.Mish()
        elif activation.lower() == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class EncoderLayerRoFormer(nn.Module):
    def __init__(self, d_model: int, num_heads: int,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 mlp_drop: float = 0.0, mlp_expansion: int = 4,
                 activation: str = "mish"):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiheadSelfAttentionRoPE(d_model, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, expansion=mlp_expansion, drop=mlp_drop, activation=activation)

    def forward(self, x: torch.Tensor, rope_sin: torch.Tensor, rope_cos: torch.Tensor) -> torch.Tensor:
        # Pre-LN + residual
        x = x + self.attn(self.ln1(x), rope_sin, rope_cos)
        x = x + self.mlp(self.ln2(x))
        return x


class RoFormerEMG(nn.Module):
    def __init__(self,
                 input_dim: int = 12,
                 output_dim: int = 10,
                 d_model: int = 128,
                 num_layers: int = 2,
                 num_heads: int = 5,
                 attn_drop: float = 0.0,
                 proj_drop: float = 0.0,
                 mlp_drop: float = 0.0,
                 mlp_expansion: int = 4,
                 activation: str = "mish",
                 conv_kernel: int = 3,
                 use_mu_law: bool = False,
                 mu: float = 220.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.mu_law = MuLaw(mu=mu, enabled=use_mu_law)

        # Conv1d pre-layer (same length padding)
        pad = (conv_kernel - 1) // 2
        self.conv1 = nn.Conv1d(input_dim, d_model, kernel_size=conv_kernel, padding=pad, bias=True)
        self.conv_act = nn.GELU()

        # RoFormer encoder stack
        self.enc_layers = nn.ModuleList([
            EncoderLayerRoFormer(d_model, num_heads,
                                 attn_drop=attn_drop, proj_drop=proj_drop,
                                 mlp_drop=mlp_drop, mlp_expansion=mlp_expansion,
                                 activation=activation)
            for _ in range(num_layers)
        ])

        self.ln_out = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, output_dim)

        # rope cache will be (re)built on first forward based on seq_len
        self._rope_cache_len = None
        self.register_buffer('_rope_sin', None, persistent=False)
        self.register_buffer('_rope_cos', None, persistent=False)

    def _ensure_rope(self, seq_len: int, head_dim: int, device, dtype):
        if self._rope_cache_len == seq_len and self._rope_sin is not None:
            return
        sin, cos = build_rope_cache(seq_len, head_dim, device=device, dtype=dtype)
        self._rope_sin = sin
        self._rope_cos = cos
        self._rope_cache_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C] or [B, C, T]; returns [B, T, 10]
        """
        if x.dim() != 3:
            raise ValueError("Expected 3D input [B,T,C] or [B,C,T]")

        # auto-convert [B,C,T] → [B,T,C] if C==12
        if x.shape[1] == 12 and x.shape[2] != 12:
            x = x.transpose(1, 2)

        # optional μ-law
        x = self.mu_law(x)
        B, T, C = x.shape

        # Conv1d expects [B,C,T]
        z = self.conv1(x.transpose(1, 2))        # [B, D, T]
        z = self.conv_act(z).transpose(1, 2)     # [B, T, D]

        d_model = z.size(-1)
        head_dim = d_model // self.enc_layers[0].attn.num_heads
        self._ensure_rope(T, head_dim, z.device, z.dtype)

        for layer in self.enc_layers:
            z = layer(z, self._rope_sin, self._rope_cos)

        z = self.ln_out(z)
        out = self.fc_out(z)                     # [B, T, 10]
        return out
if __name__ == "__main__":
    import torch
    import time

    # 自动选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # 实例化模型（按你的默认超参）
    model = RoFormerEMG(
        input_dim=12, output_dim=10,
        d_model=120, num_layers=2, num_heads=5,
        use_mu_law=False
    ).to(device)
    model.eval()

    # -------- 前向测试 1：输入 [B, T, C] --------
    sim_btc = torch.randn(1, 200, 12, device=device)   # 注意：回归模型用 float，不要 .long()
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.time()
    with torch.no_grad():
        out1 = model(sim_btc)
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"[B,T,C]  in={tuple(sim_btc.shape)}  out={tuple(out1.shape)}  time={(time.time()-t0)*1000:.2f} ms")

    # -------- 简单反向梯度检查（训练态） --------
    model.train()
    x = torch.randn(2, 200, 12, device=device, requires_grad=True)
    y = model(x)               # [B,T,10]
    loss = y.mean()            # 伪损失
    loss.backward()
    print(f"[Grad check] x.grad is None? {x.grad is None}")
