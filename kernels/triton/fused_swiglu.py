"""
OpenAI Triton Fused SwiGLU Kernel
Eliminates intermediate High-Bandwidth Memory (HBM) round-trips by computing
SwiGLU activation entirely within GPU SRAM / registers.

SwiGLU(x) = (x_gate * sigmoid(x_gate)) * x_up
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
    def _swiglu_forward_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load tile into SRAM registers (single HBM read per input)
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # In-SRAM activation computation: silu(gate) * up = gate * sigmoid(gate) * up
        # Direct polynomial / exp calculation in registers
        sig_gate = tl.sigmoid(gate)
        silu_gate = gate * sig_gate
        out = silu_gate * up

        # Single HBM write back
        tl.store(out_ptr + offsets, out, mask=mask)

    @triton.jit
    def _swiglu_backward_kernel(
        grad_out_ptr,
        gate_ptr,
        up_ptr,
        grad_gate_ptr,
        grad_up_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # Load gradients and saved activations
        grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # d(silu(g))/dg = sig(g) * (1 + g * (1 - sig(g)))
        sig_gate = tl.sigmoid(gate)
        d_silu = sig_gate * (1.0 + gate * (1.0 - sig_gate))

        # dL/dg = grad_out * up * d(silu(g))/dg
        # dL/dup = grad_out * silu(g)
        grad_gate = grad_out * up * d_silu
        grad_up = grad_out * (gate * sig_gate)

        tl.store(grad_gate_ptr + offsets, grad_gate, mask=mask)
        tl.store(grad_up_ptr + offsets, grad_up, mask=mask)


class FusedSwiGLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(gate, up)
        if TRITON_AVAILABLE and gate.is_cuda and up.is_cuda:
            out = torch.empty_like(gate)
            n_elements = gate.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            _swiglu_forward_kernel[grid](
                gate, up, out, n_elements,
                BLOCK_SIZE=1024
            )
            return out
        else:
            # Fallback PyTorch implementation
            return F.silu(gate) * up

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gate, up = ctx.saved_tensors
        if TRITON_AVAILABLE and grad_output.is_cuda and gate.is_cuda:
            grad_gate = torch.empty_like(gate)
            grad_up = torch.empty_like(up)
            n_elements = gate.numel()
            grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
            _swiglu_backward_kernel[grid](
                grad_output, gate, up, grad_gate, grad_up, n_elements,
                BLOCK_SIZE=1024
            )
            return grad_gate, grad_up
        else:
            # Exact analytical gradient fallback
            sig_gate = torch.sigmoid(gate)
            d_silu = sig_gate * (1.0 + gate * (1.0 - sig_gate))
            grad_gate = grad_output * up * d_silu
            grad_up = grad_output * F.silu(gate)
            return grad_gate, grad_up


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Computes Fused SwiGLU: silu(gate) * up"""
    return FusedSwiGLUFunction.apply(gate, up)


def reference_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Standard PyTorch eager reference implementation."""
    return F.silu(gate) * up


class FusedSwiGLU(nn.Module):
    """Production drop-in nn.Module for Fused SwiGLU."""
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_up = nn.Linear(in_features, hidden_features, bias=False)
        self.w_down = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        fused = fused_swiglu(gate, up)
        return self.w_down(fused)
