# Megakernels!

## Key Innovations

### 1. GPU Software Virtual Machine — Single-Kernel Full-Model Inference

**Problem:** Traditional LLM inference launches dozens of CUDA kernels per Transformer layer (RMS Norm, MatVec, Attention, SiLU, etc.). Kernel launch overhead and repeated global memory reads/writes significantly hurt latency, especially in single-batch decode where each kernel's compute is tiny and launch overhead can account for 30-50% of execution time.

**Solution:** Compile the entire Transformer forward pass into a single CUDA kernel ("megakernel") that runs a software-interpreted virtual machine on each SM:
- 20 warps divided into 5 specialized roles: **Controller** (instruction fetch / page allocation / semaphore construction), **Loader** (TMA async weight loading), **Launcher** (Tensor Core coordination), **Consumer** (16 compute warps for MMA), **Storer** (TMA writeback)
- Each SM fetches instructions from its own queue; a 2-stage pipeline overlaps fetch/execute to hide latency
- Shared memory is divided into 13 pages of 16KB each, dynamically allocated and recycled per instruction
- 32 dynamic semaphores provide fine-grained synchronization between loader/consumer/storer
- Cross-SM synchronization uses a global memory barrier (opcode-indexed atomic counters with nanosleep spin-loop), exposed via `gmem_barrier_wait()` / `gmem_barrier_signal()` utilities

**Applicable scenarios:** Single-batch low-latency decode, time-to-first-token critical services

**Compatible models:** LLaMA-series, Qwen2/Qwen3, any Decoder-only Transformer

### 2. Unified Encoder/Decoder Pipeline — Adaptive Prefill/Decode Dispatch

**Problem:** Prefill (multi-token matrix multiply) and Decode (single-token vector multiply) have fundamentally different compute patterns. Traditional approaches require two independent kernel implementations, leading to code duplication, complex scheduling, and switching overhead.

**Solution:** An adaptive dual-mode pipeline architecture:
- **MatVec Pipeline** (Decode mode): Single-vector input, weight tiles streamed through a 3-stage pipeline, consumer warps broadcast the activation vector and perform element-wise multiply + row-sum reduction
- **Matrix Pipeline** (Encoder mode): Multi-row matrix input, uses Tensor Core `mma_ABt` for matrix multiply, iterates over `(tokens_num + TOKEN_STEP - 1) / TOKEN_STEP` row steps, with `rms_matrix_pipeline` adding RMS norm before the matrix multiply
- Compile-time dispatch: when `tokens_num <= 16`, the decoder matvec pipeline is reused; when `> 16`, the encoder matrix pipeline activates automatically
- The encoder pipeline introduces a new opcode (`OPCODE_RMS_QKV_MatrixRopeAppend = 8`) and a `parsed_instruction` that carries `token_start` / `token_count` fields for row-block scheduling

**Applicable scenarios:** Online inference services requiring both prefill and decode, multi-batch scenarios

**Compatible models:** LLaMA, Qwen2/Qwen3

### 3. Quantization-Aware Pipeline — W4A8/INT4 End-to-End Integration

**Problem:** Weight and activation quantization is key to reducing memory and accelerating inference, but the overhead of dequantization often negates compute benefits, especially in kernel-fused scenarios where quantization operations cannot seamlessly integrate with existing pipelines.

**Solution:** A quantization-aware pipeline integrated into the megakernel VM:
- **QuantMode=1 (INT4-GPTQ):** 4-bit packed weights with group scaling factors (group_size=128), dequantized to bf16 before matmul via `dequant_int4_to_bf16()`
- **QuantMode=2 (W4A8-AWQ):** 4-bit weights dequantized to FP8 directly, FP8 activations fed into Tensor Core `mma_ABt`, with AWQ alpha channel-rescaling applied inline
- A dedicated **Scale Pipeline** (`scale_pipeline_stages=3`) with `scales_arrived/scales_finished` semaphores, loaded asynchronously by the loader warp
- The `quantized_matvec_pipeline` template extends the base matvec pipeline with `QuantMode` as a compile-time parameter, adding the scale pipeline stages and `dequant_weight()` hook in the consumer loop
- `quant_traits<QuantMode>` provides compile-time trait queries (`has_quant`, `has_fp8_act`, `scale_pipeline_stages`)

**Applicable scenarios:** Deploying quantized large models, memory-constrained environments, throughput-optimized inference

**Compatible models:** W4A8-AWQ quantized models, GPTQ quantized LLaMA models

---

## New Files

| File | Description |
|------|-------------|
| `demos/low-latency-llama/matrix_pipeline.cuh` | Multi-row matrix multiply pipeline for encoder/prefill path |
| `demos/low-latency-llama/quantized_pipeline.cuh` | Quantization-aware matvec pipeline (INT4-GPTQ / W4A8-AWQ) |
| `demos/low-latency-llama/rms_matrix_rope_append.cu` | Encoder-mode RMS+QKV op using matrix_pipeline (opcode 8) |

## Modified Files

| File | Change |
|------|--------|
| `include/util.cuh` | Added `gmem_barrier_wait()` / `gmem_barrier_signal()` cross-SM synchronization utilities |

---

## Installation

Clone this repo and run:

```bash
git submodule update --init --recursive
pip install uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install -e .
```

## Low-Latency Llama Demo

First, to compile the megakernel, run:

```bash

# from the repo root
export THUNDERKITTENS_ROOT=$(pwd)/ThunderKittens
export MEGAKERNELS_ROOT=$(pwd)
export PYTHON_VERSION=3.12 # adjust if yours is different
export GPU=H100 # options are {H100, B200}, else defaults to B200
cd demos/low-latency-llama
make

```

To start an interactive chat session with the model, run:

```bash

# from the repo root
python megakernels/scripts/llama_repl.py

```

To benchmark the megakernel, run:

```bash

# from the repo root
python megakernels/scripts/generate.py mode=mk prompt="tell me a funny joke about cookies" ntok=100

```
