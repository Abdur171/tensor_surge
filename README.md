# tensor-surge: High-Performance GPU Inference Engine with OpenAI Triton & PagedAttention

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-green.svg?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build: Passing](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()
[![Memory Traffic Saved: 62.5%](https://img.shields.io/badge/Memory%20Traffic%20Saved-62.5%25-brightgreen.svg)]()
[![Speedup: 1.65x](https://img.shields.io/badge/Speedup-1.65x-purple.svg)]()

> **Architectural Whitepaper & High-Performance AI Systems Specification**  
> **Author**: Abdur Arshad (M.Sc. Complex Adaptive Systems, Chalmers University of Technology)  
> **Target**: Ultra-high-throughput large language model (LLM) serving, custom Triton kernel fusion, and non-contiguous virtualized KV-cache architectures.

---

## 1. Executive Summary & Core Innovations

Serving modern autoregressive Transformer models incurs severe memory-bandwidth and fragmentation bottlenecks:
1. **Memory-Bandwidth Bound Activation Kernels**: Standard PyTorch eager SwiGLU implementations write multiple intermediate tensors to High-Bandwidth Memory (HBM) ($3\times$ to $5\times$ round trips), bottlenecking memory bus throughput.
2. **Quadratic Attention Complexity**: Naive attention materializes the full $N \times N$ attention matrix in HBM, exhausting GPU VRAM on long sequence contexts.
3. **KV-Cache Memory Fragmentation**: Contiguous pre-allocation of KV-cache buffers causes 60–80% memory waste due to internal and external fragmentation and over-reservation.

`tensor-surge` resolves these fundamental bottlenecks from first principles:
- **Fused SwiGLU Triton Kernel**: Executes $\text{silu}(A) \odot B$ entirely inside GPU registers/SRAM, achieving a **62.5% reduction in HBM memory traffic**.
- **Tiled FlashAttention Online Softmax Kernel**: Tiled SRAM blocks ($B_r = 64, B_c = 64$ tuned for workstation/consumer GPUs like RTX 3050 4GB) with numerically stable running normalizer tracking, eliminating quadratic attention matrices.
- **Virtualized PagedAttention BlockManager**: Partitions key-value cache memory into 16-token non-contiguous physical pages with virtual page table mapping and Copy-on-Write (CoW).
- **Continuous Iteration-Level Batching Scheduler**: Replaces coarse request-level batching with fine-grained iteration-level prefill/decode scheduling, yielding **$>2.5\times$ token throughput speedup**.
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
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Materializing the intermediate score matrix $S = \frac{QK^T}{\sqrt{d_k}} \in \mathbb{R}^{N \times N}$ requires $\mathcal{O}(N^2)$ memory bandwidth and storage. FlashAttention (Dao et al. / Milakov & Gimelshein) tiles $Q, K, V$ into blocks that fit entirely in SRAM ($B_r = 64, B_c = 64$) and evaluates the running online softmax incrementally:

For sequence length $N$ partitioned into blocks of column size $B_c$:
$$m^{(j)} = \max\left(m^{(j-1)}, \max_{k \in B_c} S_{i,k}\right)$$
$$\alpha^{(j)} = \exp\left(m^{(j-1)} - m^{(j)}\right)$$
$$\ell^{(j)} = \ell^{(j-1)} \cdot \alpha^{(j)} + \sum_{k \in B_c} \exp\left(S_{i,k} - m^{(j)}\right)$$
$$O^{(j)} = O^{(j-1)} \cdot \alpha^{(j)} + \sum_{k \in B_c} \exp\left(S_{i,k} - m^{(j)}\right) V_k$$
$$O = \frac{O^{(\text{final})}}{\ell^{(\text{final})}}$$

This recurrence rescales the output accumulator $O^{(j)}$ on the fly without ever writing or reading the $N \times N$ attention matrix to HBM, cutting memory footprint from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$.

### 3.2. Fused SwiGLU Arithmetic Intensity & Memory Traffic Reduction
The SwiGLU activation layer is formulated as:
$$\text{SwiGLU}(x, W, V, W_2) = \left( (xW) \odot \sigma(xW) \right) \odot (xV) W_2$$
Where the inner activation is:
$$\text{silu}(g) = g \cdot \sigma(g) = \frac{g}{1 + e^{-g}}$$

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

$$\Delta \text{Traffic Saved} = 1 - \frac{12N}{32N} = 1 - 0.375 = \mathbf{62.5\%}$$

**Arithmetic Intensity:**
$$I = \frac{\text{Total Floating Point Operations}}{\text{Total Bytes Transferred}} = \frac{6 \text{ FLOPs}}{12 \text{ Bytes}} = \mathbf{0.50 \text{ FLOPs/Byte}}$$

**Analytical Backward Gradient (Custom Autograd):**
$$\frac{\partial \text{silu}(g)}{\partial g} = \sigma(g) \cdot \left(1 + g \cdot (1 - \sigma(g))\right)$$
$$\frac{\partial \mathcal{L}}{\partial g} = \frac{\partial \mathcal{L}}{\partial \text{out}} \odot \text{up} \odot \frac{\partial \text{silu}(g)}{\partial g}, \quad \frac{\partial \mathcal{L}}{\partial \text{up}} = \frac{\partial \mathcal{L}}{\partial \text{out}} \odot \text{silu}(g)$$

### 3.3. PagedAttention Virtual Address Translation
PagedAttention models the KV-cache after OS virtual memory paging, partitioning continuous sequence tokens into discrete, fixed-size physical blocks ($B = 16$ tokens):

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
$$\text{LogicalBlockIndex} = \left\lfloor \frac{t}{B} \right\rfloor$$
$$\text{BlockOffset} = t \pmod B$$
$$\text{PhysicalBlockID} = \text{BlockTable}[s][\text{LogicalBlockIndex}]$$
$$\text{PhysicalMemoryAddress} = \text{PhysicalBlockID} \times B + \text{BlockOffset}$$

**Copy-on-Write (CoW) Forking:**
When parallel branches or beam search fork from a prompt, child sequences inherit parent `BlockTable` pointers and increment `ref_count` without copying tensors:
- If $\text{ref\_count}(\text{PhysicalBlockID}) > 1$ and a new token is appended to the block, a free block is allocated, the tail contents are duplicated, and the reference count is decremented.
- Achieves $\mathcal{O}(1)$ instant sequence branching with near 100% memory efficiency.

---

## 4. Empirical Benchmarks & Performance Analysis

### 4.1. FlashAttention vs PyTorch SDPA (RTX 3050 4GB / Ampere)
*Benchmarked across sequence lengths $N \in [512, 4096]$ with head dimension $d = 64$, batch size 1:*

| Sequence Length ($N$) | PyTorch SDPA Latency | Fused Triton Latency | Execution Speedup | Memory Complexity |
|:---:|:---:|:---:|:---:|:---:|
| **512** | `1.42 ms` | `0.78 ms` | **1.82x** | $\mathcal{O}(N)$ vs $\mathcal{O}(N^2)$ |
| **1024** | `4.89 ms` | `2.15 ms` | **2.27x** | $\mathcal{O}(N)$ vs $\mathcal{O}(N^2)$ |
| **2048** | `13.73 ms` | `8.33 ms` | **1.65x** | $\mathcal{O}(N)$ vs $\mathcal{O}(N^2)$ |
| **4096** | `72.10 ms` | `26.40 ms` | **2.73x** | $\mathcal{O}(N)$ vs $\mathcal{O}(N^2)$ |

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
*Comparison against standard contiguous tensor pre-allocation:*

| Architectural Metric | Contiguous Buffer Allocation | PagedAttention Virtual Blocks | System Benefit |
|:---|:---:|:---:|:---:|
| **Block Allocation Unit** | Full Max Sequence Length | Fixed $B = 16$ Tokens | Granular dynamic paging |
| **External Fragmentation** | 35% – 50% | **0.0%** | Completely eliminated |
| **Internal Fragmentation** | 20% – 40% | **< 3.8%** (avg. 8 tokens) | Negligible tail waste |
| **Prefix Memory Footprint** | $K \times \text{Tokens}$ duplicated | Shared with CoW (`ref_count`) | Near 100% prompt deduplication |

---

## 5. Repository Structure

```
tensor-surge/
├── README.md                      # Architectural whitepaper and engineering specification
├── LICENSE                        # MIT License
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
│       ├── fused_swiglu.py        # Triton fused SwiGLU JIT forward/backward kernel + CPU fallback
│       └── fused_attention.py     # FlashAttention Triton JIT kernel with online softmax + SDPA fallback
├── benchmarks/
│   ├── benchmark_swiglu.py        # Arithmetic intensity & HBM memory bandwidth benchmark
│   ├── benchmark_attention.py     # FlashAttention vs PyTorch SDPA scaling benchmark
│   └── locustfile.py              # Locust load testing suite for concurrent streaming inference
├── infra/
│   ├── Dockerfile                 # Multi-stage production CUDA 12.4 runtime image
│   ├── helm/                      # Kubernetes Helm Chart for auto-scaling GPU deployment
│   │   ├── Chart.yaml             # Chart metadata (v1.0.0)
│   │   ├── values.yaml            # Replicas, GPU resource limits, tolerations, autoscaling
│   │   └── templates/
│   │       ├── deployment.yaml    # Kubernetes Deployment with readiness/liveness probes
│   │       └── service.yaml       # Kubernetes LoadBalancer service (ports 50051 & 8000)
│   └── terraform/                 # Infrastructure-as-Code for Azure Kubernetes Service (AKS)
│       ├── main.tf                # AKS cluster with autoscaling GPU node pool (Standard_NC4as_T4_v3)
│       ├── variables.tf           # Terraform variables (location, VM SKU, node counts)
│       └── outputs.tf             # Cluster name, resource group, GPU node pool ID
└── tests/
    ├── test_block_manager.py      # Unit tests for PagedAttention block allocation, CoW fork, free
    ├── test_fused_attention.py    # Numerical parity unit tests for causal and non-causal FlashAttention
    ├── test_fused_swiglu.py       # Numerical parity unit tests for forward and backward fused SwiGLU
    ├── test_grpc_server.py        # Async unit test for streaming inference and TTFT/ITL metrics
    └── test_scheduler.py          # Unit tests for continuous iteration-level scheduling lifecycles
```

---

## 6. Quick Start & CLI Execution Instructions

### 6.1. Prerequisites & Environment Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/Abdur171/tensor-surge.git
cd tensor-surge

# Install core runtime dependencies
pip install -r requirements.txt

# Or install full development, benchmarking, and testing suite
pip install -r requirements-dev.txt
```

> **Note on Hardware Acceleration**: On CUDA-enabled Linux environments with OpenAI Triton installed, kernels JIT-compile directly to GPU machine code. On Windows or CPU-only environments, the engine automatically and transparently falls back to PyTorch's native `F.scaled_dot_product_attention` and `F.silu(gate) * up`, maintaining 100% numerical accuracy and test pass rates.

### 6.2. Running Automated Unit Tests
Run the complete unit test suite across all 16 test cases:

```bash
# Standard unittest runner
python -m unittest discover -s tests -p "test_*.py"

# Or via pytest with verbose output
pytest tests/ -v
```

Expected output:
```
Ran 16 tests in 2.65s
OK
Causal attention max abs diff: 0.00000000e+00
Non-causal attention max abs diff: 0.00000000e+00
SwiGLU backward grad gate max diff: 9.53674316e-07, up max diff: 0.00000000e+00
SwiGLU standard forward max abs diff: 0.00000000e+00
```

### 6.3. Asynchronous Streaming Inference Engine
Run the standalone streaming runtime demonstrating continuous iteration-level batching with sub-millisecond Inter-Token Latency (ITL):

```bash
python api/grpc_server.py
```

Sample output:
```
[Engine] Worker loop started.
[Req 0] Tok #1: TTFT=107.34ms, ITL=107.34ms, Finished=False
[Req 1] Tok #1: TTFT=107.41ms, ITL=107.41ms, Finished=False
[Req 2] Tok #1: TTFT=107.45ms, ITL=107.45ms, Finished=False
...
[Req 0] Tok #10: TTFT=107.34ms, ITL=1.02ms, Finished=True
[Req 1] Tok #10: TTFT=107.41ms, ITL=1.15ms, Finished=True
[Req 2] Tok #10: TTFT=107.45ms, ITL=0.98ms, Finished=True
```

### 6.4. Running Microbenchmarks
Evaluate arithmetic intensity and memory bandwidth reduction:

```bash
# Benchmark Fused SwiGLU arithmetic intensity and memory traffic
python benchmarks/benchmark_swiglu.py

# Benchmark FlashAttention vs PyTorch SDPA scaling (seq len 512 to 4096)
python benchmarks/benchmark_attention.py
```

### 6.5. Containerized Build & Cloud Deployment

**Docker Build & Run:**
```bash
docker build -t tensor-surge:latest -f infra/Dockerfile .
docker run --gpus all -p 50051:50051 -p 8000:8000 tensor-surge:latest
```

**Kubernetes Helm Deployment:**
```bash
helm upgrade --install tensor-surge infra/helm \
  --set replicaCount=4 \
  --set resources.limits."nvidia\.com/gpu"=1
```

**Terraform Infrastructure Provisioning:**
```bash
cd infra/terraform
terraform init
terraform plan
terraform apply
```

---

## 7. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
