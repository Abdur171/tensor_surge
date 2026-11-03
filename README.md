# tensor-surge: High-Performance GPU Inference Engine with OpenAI Triton & PagedAttention

[![Python 3.10+](https://img.shields.io/badge/Language-Python%203.10%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.2%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/Accelerator-CUDA%2012.4-76B900.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![OpenAI Triton](https://img.shields.io/badge/Compiler-OpenAI%20Triton-blue.svg)](https://github.com/openai/triton)
[![Tests](https://img.shields.io/badge/Unit%20Tests-16%2F16%20Passing-brightgreen.svg)]()

> **DISCLAIMER: FOR TECHNICAL DEMONSTRATION & RESEARCH PURPOSES ONLY.**  
> This software is provided strictly for engineering demonstration, academic research, and educational exploration. Neither the author nor contributors accept any liability or accountability for system downtime, inference errors, software defects, hardware anomalies, or cloud infrastructure expenses incurred while using or deploying this system. Use entirely at your own risk.

---

## 1. Overview & Core Innovations

Serving modern autoregressive Transformer models incurs severe memory-bandwidth and fragmentation bottlenecks:
1. **Memory-Bandwidth Bound Activations**: Standard PyTorch eager SwiGLU implementations write multiple intermediate tensors to High-Bandwidth Memory (HBM) (3x to 5x round trips), bottlenecking memory bus throughput.
2. **Quadratic Attention Complexity**: Naive attention materializes the full N x N attention matrix in HBM, exhausting GPU VRAM on long sequence contexts.
3. **KV-Cache Memory Fragmentation**: Contiguous pre-allocation of KV-cache buffers causes 60–80% memory waste due to internal and external fragmentation and over-reservation.

`tensor-surge` addresses these bottlenecks from first principles:
- **Fused SwiGLU Triton Kernel**: Executes SiLU(A) * B entirely inside GPU registers and SRAM, achieving a **62.5% reduction in HBM memory traffic**.
- **Tiled FlashAttention Online Softmax Kernel**: Tiled SRAM blocks ($B_r = 64, B_c = 64$) with numerically stable running normalizer tracking, eliminating quadratic attention matrices.
- **Virtualized PagedAttention BlockManager**: Partitions key-value cache memory into 16-token non-contiguous physical pages with virtual page table mapping and Copy-on-Write (CoW).
- **Continuous Iteration-Level Batching Scheduler**: Replaces coarse request-level batching with fine-grained iteration-level prefill/decode scheduling, yielding significant token throughput improvements.
- **Asynchronous Streaming Inference API**: Real-time Time To First Token (TTFT) and Inter-Token Latency (ITL) telemetry via async generator streams.

---

## 2. End-to-End System Architecture

```
+==================================================================================================+
|                                  TENSOR-SURGE ENGINE ARCHITECTURE                                |
+==================================================================================================+
                                               |
              +--------------------------------+-------------------------------+
              |                                                                |
              v                                                                v
  [ Client HTTP / REST API ]                                     [ Client High-Throughput gRPC ]
  (Stream Tokens / TTFT / ITL)                                   (Streaming Tensor Protocol)
              |                                                                |
              +--------------------------------+-------------------------------+
                                               |
                                               v
              +----------------------------------------------------------------+
              |           Asynchronous Continuous Batching Scheduler           |
              | - Iteration-Level Request Batching (Orca / vLLM Paradigm)      |
              | - Prefill Queue (Prompt Processing) vs Decode Queue (Tokens)   |
              | - Dynamic Preemption & Recomputation Engine                    |
              +----------------------------------------------------------------+
                                               |
                        +----------------------+-----------------------+
                        |                                              |
                        v                                              v
        +-------------------------------+              +-------------------------------+
        |    PagedAttention Manager     |              |         Model Runner          |
        | - Virtual Page Table Mapping  |              | - Linear Attention Q/K/V Proj |
        | - Non-contiguous 16-tok Pages | <----------> | - SwiGLU Gated MLP Proj       |
        | - Zero External Fragmentation |              | - Logits Sampling & LM Head   |
        | - Copy-on-Write (CoW) Forking |              +-------------------------------+
        +-------------------------------+                              |
                        |                                              |
                        +----------------------+-----------------------+
                                               |
                                               v
                    +======================================================+
                    |           Custom OpenAI Triton GPU Kernels           |
                    +======================================================+
                    |                                                      |
                    |  [ Fused SwiGLU Kernel ]   [ FlashAttention Kernel ] |
                    |  - In-SRAM Sigmoid/Mul     - Tiled SRAM (Br=64,Bc=64)|
                    |  - 62.5% Less HBM Traffic  - Online Running Softmax  |
                    |  - Single Load / Store Tile- O(N) Memory Scaling     |
                    +======================================================+
                                               |
                                               v
                    +------------------------------------------------------+
                    |       Cloud Native Infrastructure (Kubernetes/AKS)   |
                    | - Multi-Stage CUDA 12.4 Runtime Docker Image         |
                    | - Helm Autoscaling (2-8 Replicas @ 75% GPU Load)     |
                    | - Terraform AKS Dedicated GPU Node Pools (NC4as_T4)  |
                    +------------------------------------------------------+
```

---

## 3. Mathematical Formulations & Theoretical Foundation

### 3.1. FlashAttention Online Softmax Recurrence

Standard multi-head attention computes:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Materializing the intermediate score matrix $S = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{N \times N}$ requires $O(N^2)$ memory bandwidth and storage. FlashAttention tiles $Q, K, V$ into blocks that fit entirely in SRAM ($B_r = 64, B_c = 64$) and evaluates the running online softmax incrementally:

For sequence length $N$ partitioned into blocks of column size $B_c$:

$$
\begin{aligned}
m^{(j)} &= \max\left(m^{(j-1)}, \max_{k \in B_c} S_{i,k}\right) \\
\alpha^{(j)} &= \exp\left(m^{(j-1)} - m^{(j)}\right) \\
\ell^{(j)} &= \ell^{(j-1)} \cdot \alpha^{(j)} + \sum_{k \in B_c} \exp\left(S_{i,k} - m^{(j)}\right) \\
O^{(j)} &= O^{(j-1)} \cdot \alpha^{(j)} + \sum_{k \in B_c} \exp\left(S_{i,k} - m^{(j)}\right) V_k \\
O &= \frac{O^{(\text{final})}}{\ell^{(\text{final})}}
\end{aligned}
$$

This recurrence rescales the output accumulator $O^{(j)}$ on the fly without writing or reading the $N \times N$ attention matrix to HBM, cutting memory footprint from $O(N^2)$ to $O(N)$.

### 3.2. Fused SwiGLU Arithmetic Intensity & Memory Traffic Reduction

The SwiGLU activation layer is formulated as:

$$
\text{SwiGLU}(x, W, V, W_2) = \left( (xW) \odot \sigma(xW) \right) \odot (xV) W_2
$$

Where the inner activation is:

$$
\text{silu}(g) = g \cdot \sigma(g) = \frac{g}{1 + e^{-g}}
$$

```
Fused SwiGLU in GPU SRAM:
  HBM (DRAM)              SRAM Registers (L1 Cache)            HBM (DRAM)
+-------------+       +---------------------------------+       +-------------+
| Gate [Tile] | ----> | sig = sigmoid(gate)             |       |             |
| Up   [Tile] | ----> | silu = gate * sig               | ----> | Out [Tile]  |
+-------------+       | out  = silu * up                |       +-------------+
  (Single Read)       +---------------------------------+       (Single Write)
```

**Memory Bandwidth Analysis (FP32, Element Count $N$):**
- **PyTorch Eager**:
  1. Read `gate` ($4N$ bytes) $\to$ compute $\sigma(\text{gate})$ $\to$ write `sig` ($4N$ bytes)
  2. Read `gate` & `sig` ($8N$ bytes) $\to$ compute $\text{silu}$ $\to$ write `silu` ($4N$ bytes)
  3. Read `silu` & `up` ($8N$ bytes) $\to$ compute $\text{silu} \odot \text{up}$ $\to$ write `out` ($4N$ bytes)
  - **Total Eager Traffic**: $5\text{ reads} + 3\text{ writes} = 8N \times 4\text{ bytes} = 32N\text{ bytes}$.
- **Triton Fused Kernel**:
  1. Load tile `gate` and `up` directly into registers ($2\text{ reads} = 8N\text{ bytes}$).
  2. Compute $\sigma(\text{gate})$, $\text{silu}$, and $(\text{silu} \odot \text{up})$ in-register.
  3. Store result tile `out` directly to HBM ($1\text{ write} = 4N\text{ bytes}$).
  - **Total Fused Traffic**: $2\text{ reads} + 1\text{ write} = 3N \times 4\text{ bytes} = 12N\text{ bytes}$.

$$
\text{Traffic Saved} = 1 - \frac{12N}{32N} = 1 - 0.375 = 62.5\%
$$

$$
I = \frac{\text{Total Floating Point Operations}}{\text{Total Bytes Transferred}} = \frac{6 \text{ FLOPs}}{12 \text{ Bytes}} = 0.50 \text{ FLOPs/Byte}
$$

**Analytical Backward Gradient:**

$$
\begin{aligned}
\frac{\partial \text{silu}(g)}{\partial g} &= \sigma(g) \cdot \left(1 + g \cdot (1 - \sigma(g))\right) \\
\frac{\partial \mathcal{L}}{\partial g} &= \frac{\partial \mathcal{L}}{\partial \text{out}} \odot \text{up} \odot \frac{\partial \text{silu}(g)}{\partial g} \\
\frac{\partial \mathcal{L}}{\partial \text{up}} &= \frac{\partial \mathcal{L}}{\partial \text{out}} \odot \text{silu}(g)
\end{aligned}
$$

### 3.3. PagedAttention Virtual Address Translation

PagedAttention partitions continuous sequence tokens into fixed-size physical blocks ($B = 16$ tokens):

```
Virtual Sequence Tokens:
[ Token 0 ... 15 ] [ Token 16 ... 31 ] [ Token 32 ... 47 ]
        |                   |                    |
        v                   v                    v
Logical Block 0      Logical Block 1      Logical Block 2
        |                   |                    |
   (Page Table)        (Page Table)         (Page Table)
        |                   |                    |
        v                   v                    v
Physical Block #42   Physical Block #7    Physical Block #108
(Physical KV Cache)  (Physical KV Cache)  (Physical KV Cache)
```

For logical sequence $s$ and token position $t$:

$$
\begin{aligned}
\text{LogicalBlockIndex} &= \left\lfloor \frac{t}{B} \right\rfloor \\
\text{BlockOffset} &= t \pmod B \\
\text{PhysicalBlockID} &= \text{BlockTable}[s][\text{LogicalBlockIndex}] \\
\text{PhysicalMemoryAddress} &= \text{PhysicalBlockID} \times B + \text{BlockOffset}
\end{aligned}
$$

**Copy-on-Write (CoW) Forking:**
When parallel sampling branches fork from a shared prompt, child sequences inherit parent `BlockTable` pointers and increment `ref_count` without copying tensors:
- If `ref_count` > 1 and a new token is appended to the block, a free block is allocated, the tail contents are duplicated, and the reference count is decremented.
- Achieves O(1) instant sequence branching with near 100% memory efficiency.

---

## 4. Empirical Benchmarks & Performance Analysis

### 4.1. FlashAttention vs PyTorch SDPA
*Benchmarked across sequence lengths $N \in [512, 4096]$ with head dimension $d = 64$:*

| Sequence Length ($N$) | PyTorch SDPA Latency | Fused Triton Latency | Execution Speedup | Memory Complexity |
|:---:|:---:|:---:|:---:|:---:|
| **512** | `1.42 ms` | `0.78 ms` | **1.82x** | O(N) vs O(N^2) |
| **1024** | `4.89 ms` | `2.15 ms` | **2.27x** | O(N) vs O(N^2) |
| **2048** | `13.73 ms` | `8.33 ms` | **1.65x** | O(N) vs O(N^2) |
| **4096** | `72.10 ms` | `26.40 ms` | **2.73x** | O(N) vs O(N^2) |

### 4.2. Fused SwiGLU Activation Performance
*Benchmarked with hidden dimension $4096$ and batch dimension $2048$:*

| Metric | PyTorch Eager Mode | Triton Fused Kernel | Performance Advantage |
|:---|:---:|:---:|:---:|
| **HBM Memory Traffic** | `32N` bytes (5 reads, 3 writes) | `12N` bytes (2 reads, 1 write) | **62.5% reduction** |
| **Arithmetic Intensity** | `0.1875 FLOPs/byte` | `0.50 FLOPs/byte` | **2.67x higher intensity** |
| **Effective Bandwidth** | `63.68 GB/s` | `72.59 GB/s` | **1.14x speedup** |
| **Forward Max Abs Diff** | Reference (`0.0`) | `0.00000000e+00` | **Exact numerical parity** |
| **Backward Gate Grad Diff**| Reference (`0.0`) | `9.53674316e-07` | **Machine-precision parity** |

### 4.3. PagedAttention Memory Fragmentation
*Comparison against contiguous tensor pre-allocation:*

| Architectural Metric | Contiguous Buffer Allocation | PagedAttention Virtual Blocks | System Benefit |
|:---|:---:|:---:|:---:|
| **Block Allocation Unit** | Full Max Sequence Length | Fixed $B = 16$ Tokens | Granular dynamic paging |
| **External Fragmentation** | 35% – 50% | **0.0%** | Completely eliminated |
| **Internal Fragmentation** | 20% – 40% | **< 3.8%** (avg. 8 tokens) | Negligible tail waste |
| **Prefix Memory Footprint** | $K \times \text{Tokens}$ duplicated | Shared with CoW (`ref_count`) | Near 100% prompt deduplication |

---

## 5. Repository Structure

```text
tensor-surge/
├── README.md                      # Architecture and benchmark documentation
├── requirements.txt               # Core runtime dependencies (torch, numpy)
├── requirements-dev.txt           # Development, test, and benchmark dependencies
├── api/
│   ├── __init__.py                # Package initializer
│   └── grpc_server.py             # Streaming inference engine with TTFT & ITL metrics
├── engine/
│   ├── __init__.py                # Package initializer
│   ├── block_manager.py           # PagedAttention memory manager (virtual pages, CoW)
│   ├── scheduler.py               # Continuous iteration-level scheduler (prefill vs decode)
│   └── model_runner.py            # Iteration executor binding attention, SwiGLU, and LM head
├── kernels/
│   ├── __init__.py                # Package initializer
│   └── triton/
│       ├── __init__.py            # Package initializer
│       ├── fused_swiglu.py        # Triton fused SwiGLU forward/backward kernel + CPU fallback
│       └── fused_attention.py     # FlashAttention Triton kernel with online softmax + fallback
├── benchmarks/
│   ├── benchmark_swiglu.py        # Arithmetic intensity & HBM memory bandwidth benchmark
│   ├── benchmark_attention.py     # FlashAttention vs PyTorch SDPA scaling benchmark
│   └── locustfile.py              # Locust load testing suite for concurrent streaming inference
├── infra/
│   ├── Dockerfile                 # Multi-stage production CUDA runtime image
│   ├── helm/                      # Kubernetes Helm Chart for auto-scaling GPU deployment
│   │   ├── Chart.yaml             # Chart metadata
│   │   ├── values.yaml            # Replicas, GPU resource limits, tolerations, autoscaling
│   │   └── templates/
│   │       ├── deployment.yaml    # Kubernetes Deployment with readiness/liveness probes
│   │       └── service.yaml       # Kubernetes LoadBalancer service
│   └── terraform/                 # Infrastructure-as-Code for Azure Kubernetes Service (AKS)
│       ├── main.tf                # AKS cluster with autoscaling GPU node pool
│       ├── variables.tf           # Terraform variables
│       └── outputs.tf             # Cluster name, resource group, GPU node pool ID
└── tests/
    ├── test_block_manager.py      # Unit tests for PagedAttention block allocation, CoW fork, free
    ├── test_fused_attention.py    # Numerical parity unit tests for causal and non-causal FlashAttention
    ├── test_fused_swiglu.py        # Numerical parity and analytical backward autograd tests
    ├── test_grpc_server.py        # Async streaming API and TTFT/ITL latency tests
    └── test_scheduler.py          # ContinuousScheduler dynamic prefill/decode tests
```

---

## 6. Quick Start & Execution

### Prerequisites
- Python 3.10+
- PyTorch 2.2+ with CUDA support (or CPU fallback)
- OpenAI Triton (optional for GPU kernel JIT compilation)

```bash
# Clone the repository
git clone https://github.com/Abdur171/tensor-surge.git
cd tensor-surge

# Install dependencies
pip install -r requirements.txt
```

### Running Test Suite
```bash
python -m unittest discover tests -v
```

### Running Benchmarks
```bash
# SwiGLU Memory Bandwidth & Arithmetic Intensity Benchmark
python benchmarks/benchmark_swiglu.py

# FlashAttention vs PyTorch SDPA Benchmark
python benchmarks/benchmark_attention.py
```
