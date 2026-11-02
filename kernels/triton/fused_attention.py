"""
OpenAI Triton Fused Attention Kernel (FlashAttention with Online Softmax)
Tiled SRAM block execution with dynamic numerical scaling.
Optimized for mid-range GPUs (e.g. RTX 3050 4GB with Br=64, Bc=64 blocks).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _fused_attention_fwd_kernel(
        Q, K, V, sm_scale,
        Out,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        stride_oz, stride_oh, stride_om, stride_ok,
        Z, H, N_CTX,
        BLOCK_M: tl.constexpr,
        BLOCK_DMODEL: tl.constexpr,
        BLOCK_N: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H

        # Offset pointers for current batch and head
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        Q_ptr = Q + qvk_offset
        K_ptr = K + off_z.to(tl.int64) * stride_kz + off_h.to(tl.int64) * stride_kh
        V_ptr = V + off_z.to(tl.int64) * stride_vz + off_h.to(tl.int64) * stride_vh
        O_ptr = Out + off_z.to(tl.int64) * stride_oz + off_h.to(tl.int64) * stride_oh

        # Block pointers / offsets
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q_ptrs = Q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
        k_ptrs = K_ptr + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk)
        v_ptrs = V_ptr + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
        o_ptrs = O_ptr + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)

        # Initialize online softmax accumulators in registers/SRAM
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        # Load Q block into SRAM
        q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

        # Determine loop limits for causal vs full attention
        hi = (start_m + 1) * BLOCK_M if IS_CAUSAL else N_CTX

        # Stream K and V blocks through SRAM
        for start_n in range(0, hi, BLOCK_N):
            k_mask = (start_n + offs_n[None, :]) < N_CTX
            k = tl.load(k_ptrs + start_n * stride_kn, mask=k_mask, other=0.0)

            # Dot-product Q * K^T in SRAM
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk *= sm_scale

            if IS_CAUSAL:
                # Mask out positions where col > row
                causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
                qk = tl.where(causal_mask, qk, float("-inf"))

            # Online Softmax update
            m_curr = tl.max(qk, 1)
            m_new = tl.maximum(m_i, m_curr)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])

            # Rescale normalizer and accumulator
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            # Accumulate P * V
            v_mask = (start_n + offs_n[:, None]) < N_CTX
            v = tl.load(v_ptrs + start_n * stride_vn, mask=v_mask, other=0.0)
            acc += tl.dot(p.to(v.dtype), v)

            # Update running row max
            m_i = m_new

        # Final normalization: O = acc / l_i
        acc = acc / l_i[:, None]
        tl.store(o_ptrs, acc.to(Q.dtype), mask=offs_m[:, None] < N_CTX)


def fused_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float = None,
) -> torch.Tensor:
    """
    Computes FlashAttention using Triton online softmax kernel.
    Shape: [batch, heads, seq_len, head_dim]
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    if TRITON_AVAILABLE and q.is_cuda and k.is_cuda and v.is_cuda:
        batch_size, num_heads, seq_len, head_dim = q.shape
        out = torch.empty_like(q)

        # Block sizes tuned for RTX 3050 (4GB) L1/SRAM residency
        BLOCK_M = 64
        BLOCK_N = 64

        grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads, 1)

        _fused_attention_fwd_kernel[grid](
            q, k, v, sm_scale,
            out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            batch_size, num_heads, seq_len,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=head_dim,
            BLOCK_N=BLOCK_N,
            IS_CAUSAL=is_causal,
        )
        return out
    else:
        # Fallback to PyTorch native scaled_dot_product_attention
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=sm_scale)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    sm_scale: float = None,
) -> torch.Tensor:
    """PyTorch scaled dot-product attention reference."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=sm_scale)
