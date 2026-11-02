"""
Memory Bandwidth and Arithmetic Intensity Benchmark for Fused SwiGLU
Calculates:
- Effective Memory Bandwidth (GB/s)
- Arithmetic Intensity (FLOPs / byte)
- Speedup vs PyTorch Eager Baseline
"""

import time
import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from kernels.triton.fused_swiglu import fused_swiglu, reference_swiglu


def benchmark_swiglu(batch_size=4, seq_len=2048, dim=4096, warmup=10, iters=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking SwiGLU on {device} (Batch={batch_size}, Seq={seq_len}, Dim={dim})")

    gate = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)
    up = torch.randn(batch_size, seq_len, dim, device=device, dtype=torch.float32)

    # 1. Warm-up
    for _ in range(warmup):
        _ = reference_swiglu(gate, up)
        _ = fused_swiglu(gate, up)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # 2. Benchmark PyTorch eager baseline
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = reference_swiglu(gate, up)
    if device.type == "cuda":
        torch.cuda.synchronize()
    eager_time = (time.perf_counter() - t0) / iters

    # 3. Benchmark Fused SwiGLU
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = fused_swiglu(gate, up)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fused_time = (time.perf_counter() - t0) / iters

    # 4. Memory Traffic and Arithmetic Intensity Calculations
    # In SwiGLU:
    # Elements: N = batch_size * seq_len * dim
    # Eager reads gate, computes sigmoid(gate) [W], reads gate and sig [2R], computes silu [W],
    # reads silu and up [2R], writes out [W] -> Total HBM: 5 reads + 3 writes = 8 * N * 4 bytes
    # Fused kernel reads gate and up [2R], writes out [1W] -> Total HBM: 3 * N * 4 bytes
    n_elements = batch_size * seq_len * dim
    bytes_per_elem = 4  # float32

    # Theoretical minimum bytes for fused kernel
    fused_bytes = 3 * n_elements * bytes_per_elem
    eager_bytes = 8 * n_elements * bytes_per_elem

    # FLOPs: 1 exp/sigmoid (~4 FLOPs) + 1 mul + 1 mul = ~6 FLOPs per element
    total_flops = 6 * n_elements
    arithmetic_intensity = total_flops / fused_bytes

    fused_bandwidth = (fused_bytes / 1e9) / fused_time
    eager_bandwidth = (eager_bytes / 1e9) / eager_time
    speedup = eager_time / fused_time

    print("\n------------------------------------------------------------")
    print(f" SwiGLU Performance Summary:")
    print("------------------------------------------------------------")
    print(f"  PyTorch Eager Latency:  {eager_time * 1e3:.3f} ms")
    print(f"  Fused SwiGLU Latency:   {fused_time * 1e3:.3f} ms")
    print(f"  Execution Speedup:      {speedup:.2f}x")
    print(f"  Arithmetic Intensity:   {arithmetic_intensity:.2f} FLOPs/byte")
    print(f"  Effective Bandwidth:    {fused_bandwidth:.2f} GB/s")
    print(f"  Memory Traffic Saved:   {(1.0 - (fused_bytes / eager_bytes)) * 100:.1f}%")
    print("------------------------------------------------------------\n")


if __name__ == "__main__":
    benchmark_swiglu()
