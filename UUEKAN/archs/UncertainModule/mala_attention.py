import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Tuple
import math


def rotate_every_two(x):
    """
    Helper function for rotation operation
    Used for RoPE (Rotary Position Embedding)
    """
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack([-x2, x1], dim=-1)
    return x.flatten(-2)


def theta_shift(x, sin, cos):
    """
    Applies rotary position embedding.
    """
    return (x * cos) + (rotate_every_two(x) * sin)


class RoPE(nn.Module):
    """
    2D Rotary Position Embedding for Vision Tasks
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 4))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.register_buffer('angle', angle)

    def forward(self, slen: Tuple[int, int]):
        """
        Args:
            slen: (h, w) - Height and width of the feature map.
        Returns:
            (sin, cos) - Sine and cosine tensors for position embedding.
        """
        index_h = torch.arange(slen[0]).to(self.angle)
        index_w = torch.arange(slen[1]).to(self.angle)
        
        sin_h = torch.sin(index_h[:, None] * self.angle[None, :])  # (h, d1//2)
        sin_w = torch.sin(index_w[:, None] * self.angle[None, :])  # (w, d1//2)
        sin_h = sin_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h, w, d1//2)
        sin_w = sin_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h, w, d1//2)
        sin = torch.cat([sin_h, sin_w], -1)  # (h, w, d1)
        
        cos_h = torch.cos(index_h[:, None] * self.angle[None, :])  # (h, d1//2)
        cos_w = torch.cos(index_w[:, None] * self.angle[None, :])  # (w, d1//2)
        cos_h = cos_h.unsqueeze(1).repeat(1, slen[1], 1)  # (h, w, d1//2)
        cos_w = cos_w.unsqueeze(0).repeat(slen[0], 1, 1)  # (h, w, d1//2)
        cos = torch.cat([cos_h, cos_w], -1)  # (h, w, d1)

        return sin.flatten(0, 1), cos.flatten(0, 1)


class MALAAttention(nn.Module):
    """
    Magnitude-Aware Linear Attention
    
    Core Innovations:
    1. Linear complexity O(n) vs O(n^2).
    2. Retains magnitude information of the Query.
    3. Supports uncertainty masks.
    """

    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # QKV projection
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        
        self.scale = self.head_dim ** -0.5
        self.elu = nn.ELU()
        
        # RoPE position embedding
        self.rope = RoPE(dim, num_heads)
        
        # Cache for position embeddings
        self._cached_rope = None
        self._cached_size = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                H: int, W: int, uncertainty_mask: torch.Tensor = None):
        """
        Args:
            query: [B, N, C] query features.
            key: [B, N, C] key features.
            value: [B, N, C] value features.
            H, W: Height and width of the feature map (N = H * W).
            uncertainty_mask: [B, N, N] optional uncertainty mask.
            
        Returns:
            out: [B, N, C] output features.
        """
        B, N, C = query.shape
        assert N == H * W, f"Sequence length {N} must be equal to H*W ({H}*{W}={H*W})"
        
        # QKV projection
        q = self.q_proj(query)  # [B, N, C]
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape to multi-head format
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply ELU activation (ensures non-negativity)
        q = self.elu(q) + 1
        k = self.elu(k) + 1
        
        # Get or compute RoPE position embeddings
        if self._cached_size != (H, W):
            self._cached_rope = self.rope((H, W))
            self._cached_size = (H, W)
        sin, cos = self._cached_rope
        
        # Core innovation: calculate magnitude information z
        # z captures the global magnitude of the Query, preventing linear attention from ignoring this important information.
        z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) * self.scale  # [B, H, N, 1]
        
        # Apply rotary position embeddings
        q = theta_shift(q, sin, cos)
        k = theta_shift(k, sin, cos)
        
        # Linear attention calculation (Key: compute K^T @ V first)
        # Complexity: O(n*d^2) instead of O(n^2*d)
        kv = (k.transpose(-2, -1) * (self.scale / N) ** 0.5) @ (v * (self.scale / N) ** 0.5)  # [B, H, D, D]
        
        # MALA core formula: combine magnitude information
        # out = q @ kv * (1 + 1/z) - z * v_mean
        res = q @ kv * (1 + 1/(z + 1e-6)) - z * v.mean(dim=2, keepdim=True)  # [B, H, N, D]
        
        # If an uncertainty mask is provided, apply weighting
        if uncertainty_mask is not None:
            # uncertainty_mask: [B, N, N] -> [B, H, N, N]
            uncertainty_mask = uncertainty_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            # Apply uncertainty weighting (simplified version)
            uncertainty_weight = uncertainty_mask.mean(dim=-1, keepdim=True)  # [B, H, N, 1]
            res = res * (1 + uncertainty_weight)
        
        # Merge multi-head and project output
        res = res.transpose(1, 2).contiguous().view(B, N, C)  # [B, N, C]
        out = self.o_proj(res)
        
        return out


class MALAAttentionBlock(nn.Module):
    """
    Complete MALA attention block, including normalization and residual connections.
    Can directly replace the attention part in UncertaintyRefinementAttention.
    """
    
    def __init__(self, dim, num_heads=1, drop=0.):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = MALAAttention(dim, num_heads)
        self.drop = nn.Dropout(drop)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                H: int, W: int, uncertainty_mask: torch.Tensor = None):
        """
        Args:
            query: [B, N, C] or [B, C, H, W]
            key: [B, N, C] or [B, C, H, W]
            value: [B, N, C] or [B, C, H, W]
            H, W: Feature map dimensions.
            uncertainty_mask: Optional uncertainty mask.
            
        Returns:
            out: [B, N, C] format.
        """
        # If input is in [B, C, H, W] format, convert to [B, N, C]
        if len(query.shape) == 4:
            B, C, H_in, W_in = query.shape
            H, W = H_in, W_in
            query = query.flatten(2).transpose(1, 2)
            key = key.flatten(2).transpose(1, 2)
            value = value.flatten(2).transpose(1, 2)
        
        # Normalization
        query_norm = self.norm_q(query)
        key_norm = self.norm_kv(key)
        value_norm = self.norm_kv(value)
        
        # MALA Attention
        out = self.attn(query_norm, key_norm, value_norm, H, W, uncertainty_mask)
        out = self.drop(out)
        
        return out


def test_mala_attention():
    """
    Tests the MALA attention mechanism.
    """
    print("=" * 50)
    print("Testing MALA Attention Mechanism")
    print("=" * 50)
    
    # Test parameters
    B, C, H, W = 2, 256, 32, 32
    N = H * W
    
    # Create test data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    query = torch.randn(B, N, C).to(device)
    key = torch.randn(B, N, C).to(device)
    value = torch.randn(B, N, C).to(device)
    
    # Test MALA attention
    print(f"\nInput shapes: query={query.shape}, key={key.shape}, value={value.shape}")
    
    mala = MALAAttention(C, num_heads=1).to(device)
    
    # Forward pass
    out = mala(query, key, value, H, W)
    print(f"Output shape: {out.shape}")
    
    # Test with uncertainty mask
    uncertainty_mask = torch.rand(B, N, N).to(device)
    out_with_mask = mala(query, key, value, H, W, uncertainty_mask)
    print(f"Output shape with uncertainty mask: {out_with_mask.shape}")
    
    # Complexity comparison
    print(f"\nComplexity comparison (H={H}, W={W}, N={N}, C={C}):")
    print(f"  Traditional Attention O(N^2*C) = O({N**2 * C:,})")
    print(f"  MALA Linear Attention O(N*C^2) = O({N * C**2:,})")
    print(f"  Speedup: {(N**2 * C) / (N * C**2):.2f}x")
    
    # Test MALAAttentionBlock
    print("\nTesting MALAAttentionBlock:")
    block = MALAAttentionBlock(C, num_heads=1).to(device)
    
    # Test [B, C, H, W] input
    query_4d = torch.randn(B, C, H, W).to(device)
    key_4d = torch.randn(B, C, H, W).to(device)
    value_4d = torch.randn(B, C, H, W).to(device)
    
    out_block = block(query_4d, key_4d, value_4d, H, W)
    print(f"Block output shape: {out_block.shape}")
    
    print("\n✅ All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    test_mala_attention()

