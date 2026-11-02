"""
Benchmark Script for FlashAttention Online Softmax vs PyTorch SDPA
Compares latency and effective TFLOPs across sequence lengths (512, 1024, 2048, 4096).
"""

import time
import math
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from kernels.triton.fused_attention import fused_attention, reference_attention


def benchmark_attention():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("============================================================")
    print(f" FlashAttention vs PyTorch SDPA Benchmark (Device: {device})")
    print("============================================================\n")

    batch_size = 2
    num_heads = 8
    head_dim = 64
    seq_lengths = [512, 1024, 2048]
    if device.type == "cuda":
        seq_lengths.append(4096)

    print(f"{'SeqLen':>8} | {'PyTorch SDPA (ms)':>18} | {'Fused Triton (ms)':>18} | {'Speedup':>9} | {'TFLOPs/s':>10}")
    print("-" * 72)

    for seq_len in seq_lengths:
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(5):
            _ = reference_attention(q, k, v, is_causal=True)
            _ = fused_attention(q, k, v, is_causal=True)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark SDPA
        iters = 20
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = reference_attention(q, k, v, is_causal=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        sdpa_time = (time.perf_counter() - t0) / iters

        # Benchmark Fused Triton
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = fused_attention(q, k, v, is_causal=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fused_time = (time.perf_counter() - t0) / iters

        # Attention FLOPs: 4 * batch * heads * seq_len^2 * head_dim (for causal: 2x less)
        flops = 2.0 * batch_size * num_heads * (seq_len ** 2) * head_dim
        tflops = (flops / 1e12) / fused_time
        speedup = sdpa_time / fused_time

        print(f"{seq_len:>8} | {sdpa_time * 1e3:>18.2f} | {fused_time * 1e3:>18.2f} | {speedup:>8.2f}x | {tflops:>10.2f}")

    print("============================================================\n")


if __name__ == "__main__":
    benchmark_attention()
