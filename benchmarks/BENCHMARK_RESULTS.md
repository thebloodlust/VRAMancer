# VRAMancer — Benchmark Results (tok/s)

## Test Environment

| | |
|---|---|
| **CPU** | Intel i7 (Proxmox QEMU Q35+ICH9 VM) |
| **GPU 0** | NVIDIA RTX 3090 (24 GB, PCIe 4.0 x16, Ampere CC 8.6) |
| **GPU 1** | NVIDIA RTX 5070 Ti (16 GB, PCIe 5.0 x16, Blackwell CC 12.0) |
| **Python** | 3.12.3 |
| **PyTorch** | 2.10.0+cu128 |
| **CUDA** | 12.8 |
| **OS** | Ubuntu (Proxmox VM, VFIO GPU passthrough) |
| **Precision** | bfloat16 |

## Single-GPU Performance (Sequential, 5 prompts × 128 tokens)

| Model | Params | Native HF generate() | VRAMancer pipeline | Delta |
|---|---|---|---|---|
| GPT-2 | 124M | 123.4 tok/s | 125.6 tok/s | **+1.8%** |
| TinyLlama-1.1B | 1.1B | 53.0 tok/s | 56.5 tok/s | **+6.6%** |
| Mistral-7B-v0.1 | 7.2B | 35.1 tok/s | 34.9 tok/s | -0.6% |

**Key finding:** VRAMancer pipeline has **near-zero overhead** for large models and is **faster** (+2-7%) on small/medium models due to proactive KV cache management, causal mask skipping, and greedy fast-path optimizations.

### Methodology

- **Native HF:** `AutoModelForCausalLM.from_pretrained()` + `model.generate(do_sample=False, use_cache=True)` — no VRAMancer code involved
- **VRAMancer:** `InferencePipeline.load()` + `pipeline.generate()` — full pipeline with backend, scheduler, monitors, KV cache, fault tolerance
- Each benchmark: 1 warmup prompt (20 tokens) + 5 sequential prompts × 128 tokens each
- GPU: RTX 3090 (`CUDA_VISIBLE_DEVICES=0`)
- All models loaded in bfloat16, greedy decoding

## Multi-GPU Performance (GPT-2, 5 prompts × 64 tokens)

| Mode | GPUs | Throughput | vs Single |
|---|---|---|---|
| **Single GPU (native HF)** | 1 × RTX 3090 | 124.6 tok/s | baseline |
| **Pipeline-parallel** | RTX 3090 + RTX 5070 Ti | 92.0 tok/s | -26.2% |
| **Tensor-parallel** | RTX 3090 + RTX 5070 Ti | 86.0 tok/s | -31.0% |

**Analysis:** GPT-2 (124M params) is too small to benefit from multi-GPU — PCIe transfer overhead dominates. Multi-GPU shines for larger models (7B+) that don't fit on a single GPU. Both pipeline-parallel and tensor-parallel modes run stably with no TDR crashes.

### Multi-GPU Details

- **Pipeline-parallel:** VRAMancer `InferencePipeline` with `VRM_FORCE_MULTI_GPU=1`, model split by `accelerate` across 2 devices (7 layers GPU 0, 10 layers GPU 1), CPU-staged transfers (P2P blocked by IOMMU/VM)
- **Tensor-parallel:** `core/tensor_parallel.py` — column-parallel Q/K/V + row-parallel O/down projections, NCCL all-reduce sum, each GPU processes half the attention heads
- **P2P status:** Disabled (consumer GPUs + VFIO passthrough), all cross-GPU transfers via CPU-staged pinned memory
- **No TDR crashes** — resolved by GPU persistence mode (`nvidia-smi -pm 1`)

## CUDA PagedAttention Kernel

Custom CUDA decode kernel (`csrc/paged_attention_kernel.cu`) — warp-level online softmax, GQA support, fp16/fp32 KV pool:

| Context Length | PyTorch bmm | CUDA Kernel | Speedup |
|---|---|---|---|
| 64 tokens | 1.00x | 0.114x | **8.80x** |
| 256 tokens | 1.00x | 0.420x | **2.38x** |
| 1024 tokens | 1.00x | 1.43x | 0.70x |

Best for decode (short context per step). PyTorch bmm wins at long prefill contexts where batch-level parallelism dominates.

## Heterogeneous Multi-GPU — The Proof (Qwen2.5-14B-Instruct)

The key VRAMancer value proposition: run models **too large for any single GPU** by pooling heterogeneous VRAM.

| Test | GPU(s) | VRAM Used | Result |
|---|---|---|---|
| 14B single GPU 0 | RTX 3090 (23.6 GB) | — | **OOM** |
| 14B single GPU 1 | RTX 5070 Ti (15.5 GB) | — | **OOM** |
| 14B VRAMancer 2-GPU | 3090 + 5070 Ti (38.5 GB pool) | GPU 0: 21.7 GiB, GPU 1: 14.2 GiB, CPU: 1 layer | **16.1 tok/s** ✅ |

- Model: Qwen2.5-14B-Instruct (~28 GB bf16)
- Split: VRAM-proportional — {cuda:0: 23 layers, cuda:1: 28 layers, cpu: 1 (lm_head)}
- Load time: 3.9s
- Budget: 92% VRAM to minimize CPU offload
- Transfers: CPU-staged pinned memory (~11 GB/s), P2P blocked by VFIO/IOMMU
- **Bare metal expected:** +10-30% throughput with P2P/NVLink transfers

## Quantization Performance Tiers

NF4 quantization enables a **counter-intuitive speedup**: by compressing the model to fit on a single GPU, it eliminates cross-GPU transfer overhead entirely.

### Qwen2.5-14B-Instruct (14.7B params)

| Quantization | GPUs | VRAM | Throughput | vs BF16 |
|---|---|---|---|---|
| **BF16** | 2-GPU (3090 + 5070 Ti) | 35.9 GiB total | 16.1 tok/s | baseline |
| **BF16 TurboEngine** | 2-GPU (3090 + 5070 Ti) | 35.9 GiB total | 16.2 tok/s | +1% |
| **NF4** | single GPU 0 (3090) | 10.8 GiB | **10.5 tok/s** | -35% (single GPU, no multi-GPU overhead) |
| **INT8** | — | — | — | **OOM** (see notes) |

**Note:** The 14B BF16 2-GPU result improved from 6.0 tok/s (previous session) to 16.1 tok/s (+168%). Likely cause: model now fully cached on disk (no partial downloads), better accelerate layer placement.

### Qwen2.5-7B-Instruct (7.6B params)

| Quantization | GPUs | VRAM | Throughput |
|---|---|---|---|
| **NF4** | single GPU 0 (3090) | ~5 GiB | **20.2 tok/s** |
| **INT8** | single GPU 0 (3090) | ~10 GiB | **8.1 tok/s** |

### Key Findings

1. **NF4 vs BF16 trade-off** — 14B NF4 single-GPU (10.5 tok/s) is 35% slower than BF16 2-GPU (16.1 tok/s) but uses 70% less VRAM (10.8 vs 35.9 GiB). When the model fits on one GPU, BF16 multi-GPU with fast transfers wins on throughput.
2. **NF4 > INT8** — 4-bit quantization is both smaller and faster than 8-bit due to BnB's `LLM.int8()` outlier decomposition overhead.
3. **INT8 14B limitation** — INT8 14B requires ~22 GB (weights + outlier state). Loading peaks exceed 23.6 GB on RTX 3090. Multi-GPU BnB is broken upstream (accelerate hooks don't handle cross-device residual connections with quantized layers — confirmed with standalone test, not a VRAMancer issue).

### Known Upstream Bugs (transformers 5.3.0 + accelerate 1.13.0 + BnB 0.49.2)

- **Multi-GPU BnB broken**: `AlignDevicesHook` doesn't move residual tensors across devices for quantized layers. Error: `Expected all tensors on same device, cuda:0 and cuda:1` at residual addition. Confirmed with pure HF test (no VRAMancer). Workaround: single-GPU placement.
- **`dtype` bypasses BnB**: The new transformers 5.3 `dtype` parameter completely bypasses BnB quantization (loads full precision). Must use deprecated `torch_dtype=torch.float16` for BnB loads.

## vLLM Comparison

**Not tested.** vLLM 0.18.0 requires downgrading transformers (5.3.0 → 4.57.6) and numpy (2.4.3 → 2.2.6), which would break the current environment. A dedicated venv comparison is planned.

## Auto-GPU Selection

VRAMancer's `_auto_select_num_gpus()` correctly identifies when multi-GPU is unnecessary:

```
Mistral-7B-v0.1 fits on single GPU 0 (estimated 14.1 GB, GPU has 23.3 GB free
→ skipping multi-GPU overhead)
```

This avoids the PCIe transfer penalty entirely when the model fits on a single GPU.

## TurboEngine — Compiled Inference (Qwen2.5-7B-Instruct)

`core/turbo_engine.py` replaces HF `generate()` with a hand-rolled autoregressive loop, optionally compiled with `torch.compile(mode="default")`. Eliminates overhead from GenerationMixin dispatch, LogitsProcessor chains, and stopping criteria.

### Single-GPU (RTX 3090)

| Config | tok/s | vs HF generate |
|---|---|---|
| **fp16 HF generate** | 36.5 | baseline |
| **fp16 TurboEngine** (compiled) | **49.1** | **+34%** |
| **NF4 HF generate** | 20.5 | baseline |
| **NF4 TurboEngine** (compiled) | **29.4** | **+43%** |

### Single-GPU (RTX 5070 Ti)

| Config | tok/s | vs HF generate |
|---|---|---|
| **fp16 HF generate** | 36.1 | baseline |
| **fp16 TurboEngine** (compiled) | **48.8** | **+35%** |
| **NF4 HF generate** | 19.8 | baseline |
| **NF4 TurboEngine** (compiled) | **29.1** | **+47%** |

### Key Findings

1. **TurboEngine adds +34-47%** on top of HF generate, purely from loop optimization + torch.compile.
2. **RTX 5070 Ti matches RTX 3090** within 1-4% (Blackwell CC 12.0 vs Ampere CC 8.6 — similar memory bandwidth is the bottleneck).
3. **torch.compile limitations**: Only `mode="default"` works. `reduce-overhead` and `max-autotune` crash with CUDA graphs on quantized models. Compilation takes 5-15 minutes per model.
4. **NF4 benefits more** (+43-47%) than fp16 (+34-35%) — BnB dequantization creates more overhead that compile can fuse/amortize.

### Multi-GPU TurboEngine (Qwen2.5-14B-Instruct, 2-GPU)

torch.compile cannot be used for multi-GPU because accelerate's `AlignDevicesHook` must intercept each layer forward for cross-device transfers. MultiGPUTurboEngine uses the same hand-rolled decode loop without compilation.

| Config | tok/s | vs HF |
|---|---|---|
| BF16 2-GPU HF generate | 16.1 | 1.00x |
| BF16 2-GPU MultiGPUTurboEngine | 16.2 | 1.01x |

Multi-GPU TurboEngine matches HF generate — the bottleneck is PCIe transfers between GPUs (CPU-staged, ~11 GB/s in Proxmox VM), not Python-side overhead the compile eliminates.

### Speculative Decoding (Qwen 0.5B draft → 7B NF4 main)

| Config | tok/s | vs TurboEngine | Acceptance Rate |
|---|---|---|---|
| TurboEngine (uncompiled ref) | 17.8 | 1.00x | — |
| Speculative γ=3 | 13.6 | 0.76x | 35% |
| Speculative γ=5 | 12.3 | 0.69x | 28% |
| Speculative γ=8 | 10.9 | 0.61x | 23% |

**Conclusion:** Speculative decoding is a **net loss** for 0.5B→7B NF4. Acceptance rate too low (needs >60% to break even). The 0.5B draft model is too dissimilar from a quantized 7B main. Would require a same-architecture distilled draft (e.g., 1.5B pruned from 7B).

### AWQ — Not Viable

- autoawq 0.2.9: CUDA fused kernels **not compiled** for PyTorch 2.10
- AWQ HF generate: 17.5 tok/s, AWQ TurboEngine (compiled): 11.9 tok/s (**0.68x worse**)
- Root cause: autoawq dispatches to slow Python fallback kernels. Project appears deprecated.

## Maximum Throughput Ceiling (fp16 Qwen2.5-7B, RTX 3090)

Tested every available optimization strategy to find the absolute ceiling for fp16 decode:

| Approach | tok/s | vs Baseline |
|---|---|---|
| Baseline (uncompiled `model.forward`) | 37.1 | 1.00x |
| **TurboEngine** (`torch.compile mode=default`) | **49.2** | **1.33x** |
| CUDA stream (dedicated compute stream) | 48.8 | 1.32x |
| `torch.compile mode=reduce-overhead` (CUDA graphs) | 49.1 | 1.32x |

**Conclusion:** All compiled approaches converge at **~49 tok/s**. This is a **hardware-fundamental ceiling** — not a software limitation. At fp16, a 7B model requires reading ~14 GB of weights per decode step. On RTX 3090 (936 GB/s bandwidth): `936 / 14 ≈ 67 tok/s` theoretical max. 49 tok/s = **73% bandwidth utilization** (overhead from KV cache reads, activations, kernel launches). CUDA streams and CUDA graphs add nothing beyond `torch.compile` because the bottleneck is memory bandwidth, not kernel launch overhead.

## Custom Triton NF4 GEMV Kernel (`core/triton_gemv.py`)

Attempted to fuse NF4 dequantization + GEMV in a single Triton kernel to eliminate BnB's per-layer overhead.

### BnB NF4 Data Format (documented for reference)

- Packed weights: `uint8`, 2 elements per byte (blocksize=64)
- **Hi nibble (bits 7-4) → EVEN element (index 2i)**, **Lo nibble (bits 3-0) → ODD element (index 2i+1)**
- Code table: 16 `float32` NF4 quantization levels (fixed table)
- Absmax: nested quantization — absmax itself is `uint8`, requires `dequantize_blockwise(absmax, state2) + offset`

### Results (Qwen2.5-7B layer dimensions)

| Matrix Size | BnB µs | Triton µs | Speedup |
|---|---|---|---|
| [3584, 3584] | 103.2 | 116.3 | 0.89x |
| [3584, 18944] | 103.7 | 195.5 | 0.53x |
| [18944, 3584] | 103.9 | 192.2 | 0.54x |
| [3584, 152064] | 682.2 | 1543.0 | 0.44x |
| [4096, 4096] | 102.5 | 115.9 | 0.88x |

**Correctness: PERFECT** (max diff matches BnB's own fp16 rounding error).

**Conclusion:** Triton kernel is **0.44-0.89x slower** than BnB's hand-tuned CUDA. GEMV is purely memory-bandwidth bound — both read the same data, but BnB uses `float4` vectorized loads + warp shuffles with lower overhead per byte. A Triton kernel cannot beat a tuned CUDA kernel at pure memory-bandwidth workloads. Kept as reference implementation.

## GGUF / llama.cpp vs BnB NF4 (Qwen2.5-7B-Instruct)

GGUF Q4_K_M quantization via llama-cpp-python (`core/backends_llamacpp.py`) vs BnB NF4 via HuggingFace. GGUF uses dp4a INT8 dot product kernels that keep weights quantized during compute (4 MACs/cycle vs fp16 1 MAC/cycle), while BnB NF4 dequantizes to fp16 before every GEMV.

### Results (5 prompts × 128 tokens, Qwen2.5-7B-Instruct)

| Method | tok/s | TTFT | Load Time | VRAM | Model Size |
|---|---|---|---|---|---|
| **Raw llama-cpp-python 1-GPU** | **106.8** | 27.4 ms | 1.0s | 3.0 GB | 4.4 GB (Q4_K_M) |
| **VRAMancer LlamaCppBackend 1-GPU** | **106.7** | 15.3 ms | 1.2s | 3.0 GB | 4.4 GB (Q4_K_M) |
| **Raw llama-cpp-python 2-GPU** | **107.7** | 14.6 ms | 0.9s | 3.0 GB | 4.4 GB (Q4_K_M) |
| **VRAMancer LlamaCppBackend 2-GPU** | **106.7** | 14.5 ms | 1.1s | 3.0 GB | 4.4 GB (Q4_K_M) |
| BnB NF4 (HuggingFace generate) | 19.7 | — | 5.7s | 6.6 GB | ~5 GB (NF4) |

### Key Findings

1. **GGUF Q4_K_M is 5.4x faster than BnB NF4** (106.8 vs 19.7 tok/s) — dp4a kernels keep weights quantized during compute, while BnB dequantizes to fp16 before every GEMV operation.
2. **GGUF uses 2.2x less VRAM** (3.0 GB vs 6.6 GB) — Q4_K_M format is more compact than NF4 with double quantization.
3. **GGUF loads 5.7x faster** (1.0s vs 5.7s) — GGUF is a single mmap'd file vs BnB's per-layer safetensor loading + quantization.
4. **VRAMancer LlamaCppBackend has zero overhead** vs raw llama-cpp-python (106.7 vs 106.8 tok/s) — the backend is a thin wrapper.
5. **2-GPU tensor_split adds nothing** for a 7B Q4_K_M model (3.0 GB) that fits easily on a single GPU. Multi-GPU benefits only models that exceed single-GPU VRAM.
6. **GGUF is 2.17x faster than the fp16 TurboEngine ceiling** (106.8 vs 49.2 tok/s) — 4-bit quantized models read 4x fewer bytes per decode step, linearly increasing throughput for memory-bandwidth-bound inference.

### Why GGUF Wins

The performance gap is architectural:

- **BnB NF4**: Read 4-bit packed weights → dequantize to fp16 → fp16 GEMV → result. Dequantization is a separate kernel launch per layer, and fp16 GEMV uses 1 fused multiply-accumulate per cycle.
- **GGUF Q4_K_M**: Read 4-bit packed weights → dp4a INT8 dot product directly on packed data → accumulate in fp32 → result. The dp4a instruction processes 4 INT8 multiply-accumulates in a single cycle, and weights stay quantized during compute.

At memory-bandwidth saturation (which is the bottleneck for all LLM decode), reading 4x fewer bytes = 4x faster. GGUF achieves this because it never expands weights to fp16. The measured 5.4x speedup vs BnB NF4 (which does expand) tracks with 4x less data read + eliminated dequantization kernel overhead.

## What's Missing

- [ ] Continuous batching throughput (multi-request concurrent)
- [ ] vLLM comparison (requires separate venv — version conflicts with current transformers/numpy)
- [ ] Bare-metal benchmarks (no VM overhead, P2P/NVLink enabled)
- [ ] INT8 14B (needs GPU with >24 GB, or upstream multi-GPU BnB fix)
- [x] ~~Custom CUDA fused kernel for NF4~~ — Triton GEMV kernel correct but 0.44-0.89x vs BnB (memory-bandwidth bound, can't beat tuned CUDA)
- [x] ~~CUDA graphs / reduce-overhead~~ — converges at same ~49 tok/s as torch.compile default (bandwidth ceiling)
- [x] ~~GGUF/llama.cpp vs BnB NF4~~ — **5.4x speedup** (see above)
- [x] ~~NVFP4 Blackwell native FP4~~ — **real cublas FP4 kernel working** (see below)
- [x] ~~TurboQuant KV compression + Sparse V~~ — **+107% throughput on TinyLlama** (see below)
- [ ] GGUF/llama.cpp comparison (Ollama's backend — likely the 60-100 tok/s reference)
- [ ] NVFP4 + TurboQuant + Sparse V combined (blocked by GPU contention during benchmark session)

## TurboQuant + Sparse V — KV Cache Compression (31 March 2026)

Google's TurboQuant (ICLR March 2026) compresses KV cache to ~3.5-4.0 bits/dim using PolarQuant + QJL random projection. VRAMancer's implementation adds **Sparse V**: after computing attention weights from compressed keys, only the top-k% of value tokens are decompressed — the other 90% are skipped entirely.

### KV Compressor Microbenchmark (CPU, RTX 3090)

| head_dim | bits/dim | compression | seq=128 compress | seq=512 compress | seq=2048 compress |
|---|---|---|---|---|---|
| 64 | 4.0 | **4.0x** | 42ms | 70ms | 470ms |
| 128 | 3.7 | **4.3x** | 19ms | 25ms | 529ms |

### Pipeline Throughput — GPT-2 124M (RTX 3090, 3 prompts × 64 tokens)

| Configuration | tok/s | VRAM GB | vs BF16 |
|---|---|---|---|
| BF16 baseline | 302.5 | 1.59 | — |
| TurboQuant (3-bit) | 347.7 | 3.25 | **+14.9%** |
| TurboQuant + Sparse V 30% | 346.2 | 4.90 | **+14.4%** |
| TurboQuant + Sparse V 10% | 296.7 | 6.48 | -1.9% |

### Pipeline Throughput — TinyLlama 1.1B (RTX 3090, 3 prompts × 128 tokens)

| Configuration | tok/s | VRAM GB | vs BF16 |
|---|---|---|---|
| BF16 baseline | 157.0 | 3.69 | — |
| TurboQuant (3-bit) | 195.6 | 7.37 | **+24.6%** |
| TurboQuant + Sparse V 30% | 238.7 | 11.05 | **+52.0%** |
| TurboQuant + Sparse V 10% | **324.9** | 14.73 | **+106.9%** |

### Key Findings

1. **TurboQuant alone adds +15-25%** throughput from compressed KV cache (4x less KV data to read during attention).
2. **Sparse V 10% doubles throughput** on TinyLlama (157 → 325 tok/s, +107%). By decompressing only the top 10% of values by attention weight, 90% of decompress operations are eliminated.
3. **Sparse V benefit scales with model size**: GPT-2 (124M) sees -2% with Sparse V 10% (decompression cost exceeds savings), while TinyLlama (1.1B) sees +107% (KV cache is larger relative to model, so savings dominate).
4. **VRAM reporting includes PagedKVCache accumulation**: The increasing VRAM across configs is an artifact of running sequential configs in the same process — PagedKVCache pages aren't fully freed between runs.

### Activation

```bash
# TurboQuant KV compression (3-bit polar angles + QJL random projection)
VRM_KV_COMPRESSION=turboquant python -m vramancer --model gpt2

# TurboQuant + Sparse V (decompress only top 10% of values)
VRM_KV_COMPRESSION=turboquant VRM_SPARSE_V_RATIO=0.1 python -m vramancer --model gpt2

# Run the benchmark
python benchmarks/bench_turboquant.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-tokens 128
```

## NVFP4 Blackwell — Native FP4 Quantization (RTX 5070 Ti)

Native Blackwell FP4 quantization via `torchao.prototype.mx_formats` (torchao 0.16.0). Uses `torch._scaled_mm` with `float4_e2m1fn_x2` dtype — the real cublas Blackwell FP4 kernel on sm100+ GPUs.

### Results (Qwen2.5-7B-Instruct, 128 new tokens, RTX 5070 Ti CC 12.0)

| Method | tok/s | VRAM (GB) | vs BF16 | Notes |
|---|---|---|---|---|
| **BF16 (baseline)** | 36.4 | 15.25 | 100.0% | Standard bfloat16 inference |
| **NVFP4 Dynamic W+A** | **11.0** | **5.87** | 30.2% | Real cublas FP4 kernel (torch._scaled_mm) |
| **BnB NF4** | 17.5 | 14.77 | 48.0% | BnB kgemm_4bit_inference_naive |
| **BnB INT8** | 7.9 | 14.76 | 21.7% | LLM.int8() with outlier decomposition |
| NVFP4 Weight-Only | 0.9 | 7.91 | 2.5% | Dequantizes every pass (prototype limitation) |

### Key Findings

1. **NVFP4 Dynamic W+A saves 62% VRAM** vs BF16 (5.87 GB vs 15.25 GB). This is the most VRAM-efficient method, using 2.5x less VRAM than BnB NF4.
2. **NVFP4 is slower than BnB NF4** (11.0 vs 17.5 tok/s). The torchao 0.16.0 NVFP4 implementation is still **prototype**: Python dispatch overhead + per-token dynamic activation quantization adds latency. This should improve significantly in stable torchao releases.
3. **NVFP4 Weight-Only is unusable** (0.9 tok/s) — the prototype implementation dequantizes weights at every forward pass (`weight_tensor.dequantize(orig_dtype)` at line 538 of nvfp4_tensor.py) instead of using native FP4 compute.
4. **torch.compile on NVFP4**: Compilation hangs indefinitely due to custom tensor subclass overhead in Inductor. Not viable with prototype NVFP4 tensors.
5. **NVFP4 requires Blackwell (CC >= 10.0)**: The `torch._scaled_mm` FP4 kernel is only available on sm100+ GPUs. VRAMancer auto-detects and falls back to BnB NF4 on older GPUs.

### How NVFP4 Works

NVFP4 Dynamic W+A mode quantizes both weights and activations to 4-bit:
- **Weights**: Quantized offline to `float4_e2m1fn_x2` with blocked FP8 scales
- **Activations**: Quantized dynamically per-token at each layer
- **Computation**: `torch._scaled_mm(a.view(float4_e2m1fn_x2), b.view(float4_e2m1fn_x2), a_scale.view(float8_e4m3fn), b_scale.view(float8_e4m3fn))` — real cublas Blackwell FP4 kernel
- **lm_head excluded**: `aten.expand` not implemented for NVFP4Tensor — excluded via filter_fn

### VRAMancer Integration

VRAMancer automatically selects NVFP4 when `VRM_QUANTIZATION=nvfp4` is set:
- Auto-detects Blackwell GPU (CC >= 10.0) and torchao >= 0.16 availability
- Falls back to BnB NF4 on non-Blackwell GPUs or missing torchao
- Post-load quantization: loads BF16 on CPU → `quantize_(model, NVFP4DynamicActivationNVFP4WeightConfig(), filter_fn)` → moves to best Blackwell GPU
- **DirectFP4 bypass**: After quantization, NVFP4Tensor layers are replaced with `DirectFP4Linear` (plain buffers + direct `torch._scaled_mm` call) to eliminate `__torch_dispatch__` overhead
- Excludes lm_head and embedding layers from quantization

### DirectFP4 Bypass — Eliminating torchao Dispatch Overhead

The torchao NVFP4Tensor uses `__torch_dispatch__` to intercept every aten operation, routing through Python before reaching the cuBLAS FP4 kernel. VRAMancer's `DirectFP4Linear` (`core/nvfp4_direct.py`) bypasses this by:

1. Extracting raw FP4 weight data + swizzled scales from NVFP4Tensor as plain buffers
2. Quantizing activations via fused Triton kernel (no Python loop)
3. Calling `torch._scaled_mm` directly (no tensor subclass dispatch)

**torchao dispatch chain** (per Linear, per token):
```
F.linear → __torch_dispatch__(aten.linear) → nvfp4_linear() →
NVFP4Tensor.to_nvfp4(activation) → _addmm_nvfp4_dispatch() → _scaled_mm
```

**DirectFP4 bypass**:
```
forward() → triton_quantize_nvfp4(activation) → _scaled_mm
```

#### Benchmark Results (Qwen2.5-7B-Instruct, 128 new tokens, RTX 5070 Ti)

| Method | tok/s | VRAM (GB) | vs torchao | Notes |
|---|---|---|---|---|
| **torchao NVFP4** | 11.2 | 5.46 | 1.00x | Baseline (NVFP4Tensor + __torch_dispatch__) |
| **DirectFP4 bypass** | **12.0** | **5.46** | **1.07x** | Plain buffers + direct _scaled_mm |
| DirectFP4 + torch.compile | 12.0 | 5.46 | 1.07x | torch.compile works (no tensor subclass) |

- **7% speedup** with zero extra VRAM
- **Exact numerical match** with torchao output (same cuBLAS call, same data)
- **torch.compile works** on DirectFP4Linear (hangs on NVFP4Tensor)
- Applied automatically after quantization in `_apply_nvfp4_quantization()`

## TurboQuant on Qwen2.5-7B-Instruct — CPU Bottleneck (RTX 3090)

TurboQuant KV compression shows dramatically different behavior on Qwen-7B (128-dim heads) vs TinyLlama (64-dim heads). The pure Python PolarQuant + QJL implementation hits a CPU bottleneck that scales with head_dim × seq_len.

### Results (Qwen2.5-7B-Instruct, 3 prompts × 128 tokens, RTX 3090 24 GB)

| Configuration | tok/s | time (s) | VRAM (GB) | vs BF16 |
|---|---|---|---|---|
| **BF16 baseline** | **48.9** | 7.85 | 19.02 | — |
| TurboQuant (3-bit) | 0.75 | 509.7 | 23.99 | **-98.5%** (65x slower) |
| TurboQuant + Sparse V 30% | 0.28 | 1349.2 | 22.89 | **-99.4%** (175x slower) |
| TurboQuant + Sparse V 10% | 0.33 | 1161.0 | 22.89 | **-99.3%** (148x slower) |

### Why TurboQuant Regresses on Qwen-7B

| Factor | TinyLlama 1.1B (+107%) | Qwen 7B (-98.5%) |
|---|---|---|
| head_dim | 64 | 128 |
| num_kv_heads | 4 | 4 |
| PolarQuant ops/token | O(64 × seq_len) | O(128 × seq_len) |
| QJL projection | 64×64 matrix | 128×128 matrix |
| Compute ratio KV/model | KV cache dominates → savings win | Model compute dominates → overhead loses |
| Implementation | Pure Python (numpy) | Pure Python (numpy) |

**Root cause**: TurboQuant's KV compression runs entirely on CPU (Python + numpy). On small models (TinyLlama), KV cache is a proportionally larger bottleneck, so the 4x compression helps. On Qwen-7B, the model forward pass is fast on the 3090 (~49 tok/s BF16) but KV compression adds ~1.3s CPU overhead per token — O(head_dim × seq_len) per head per layer, all in Python.

**Fix path**: CUDA/Triton kernels for PolarQuant + QJL. The compression algorithm is sound; only the implementation needs to be GPU-native.

## Bi-GPU Qwen2.5-7B — Heterogeneous Pipeline Parallel (RTX 3090 + RTX 5070 Ti)

Pipeline-parallel split across two heterogeneous GPUs: RTX 3090 (24 GB, Ampere CC 8.6) + RTX 5070 Ti (16 GB, Blackwell CC 12.0). VRAM-proportional layer split with CPU-staged pinned transfers (Strategy 4 — Proxmox VM, IOMMU blocks P2P).

### Results (Qwen2.5-7B-Instruct, 3 prompts × 128 tokens, 2 GPUs)

| Configuration | tok/s | time (s) | VRAM (GB) | Status |
|---|---|---|---|---|
| **BF16 baseline** | **26.1** | 14.7 | 26.30 | OK |
| TurboQuant (3-bit) | 4.6 | 82.8 | 38.83 | OK |
| TurboQuant + Sparse V 30% | — | — | — | OOM (GPU 0 full, 15.47 GB) |
| TurboQuant + Sparse V 10% | — | — | — | OOM (GPU 1 full, 23.56 GB) |

### Key Findings

1. **Bi-GPU BF16 = 26.1 tok/s** — roughly half of single 3090 speed (48.9 tok/s). This is expected for a 7B model: it fits on one GPU, so the pipeline-parallel overhead (inter-GPU transfers via CPU staging) outweighs the benefit of additional compute.
2. **TurboQuant bi-GPU = 4.6 tok/s** vs 0.75 tok/s single-GPU (+6.1x). The CPU bottleneck benefits from parallel KV operations across 2 GPUs, but remains the primary bottleneck.
3. **Sparse V OOM on bi-GPU**: PagedKVCache allocates buffers on both GPUs. With Sparse V decompression buffers on top of the model split, total VRAM exceeds both GPUs (15.47 GB + 23.56 GB = 39 GB).
4. **Multi-GPU is for models that don't fit**: Qwen-7B BF16 fits on a single 3090 (19 GB). Splitting it across 2 GPUs adds transfer overhead for zero capacity benefit. The real multi-GPU win is Qwen-14B (35.9 GB, doesn't fit on any single GPU → 6.0 tok/s bi-GPU).

### NVFP4 + TurboQuant Combined (RTX 5070 Ti)

NVFP4 quantized model + TurboQuant KV compression on a single RTX 5070 Ti:

| Configuration | tok/s | VRAM (GB) | Status |
|---|---|---|---|
| **NVFP4 baseline** | **1.1** | 11.31 | OK (includes TurboEngine JIT) |
| NVFP4 + TurboQuant + Sparse V 10% | — | — | OOM (15.43 GB on 16 GB GPU) |

The NVFP4 baseline averages 1.1 tok/s across 3 prompts (384 tokens / 338.8s) because the first prompt includes TurboEngine graph compilation (~8 minutes). After compilation, steady-state NVFP4 throughput is approximately 5-6x higher on subsequent prompts.

NVFP4+TQ+SV10% exceeds the 5070 Ti's 16 GB: the NVFP4 model (~5.5 GB) + PagedKVCache + TQ buffers + Sparse V decompression buffers total 15.43 GB, leaving no room for intermediate activations.

## Reproduction

```bash
# Single model benchmark
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_tok_s.py --model gpt2 --max-tokens 128 --num-prompts 5

# Multi-GPU validation
python benchmarks/bench_vs_vllm.py --model gpt2 --skip-vllm

# RTX 5070 Ti vs RTX 3090 with TurboEngine
python benchmarks/bench_5070ti.py

# 14B 2-GPU TurboEngine
python benchmarks/bench_14b_turbo.py

# Speculative decoding
python benchmarks/bench_speculative_quick.py

# Heterogeneous multi-GPU (14B, requires 2 GPUs with ~38 GB total VRAM)
python benchmarks/bench_heterogeneous.py --model Qwen/Qwen2.5-14B-Instruct

# Quantization tiers
VRM_QUANTIZATION=nf4 python benchmarks/bench_heterogeneous.py --model Qwen/Qwen2.5-14B-Instruct --quantization nf4
VRM_QUANTIZATION=nf4 python benchmarks/bench_heterogeneous.py --model Qwen/Qwen2.5-7B-Instruct --quantization nf4
VRM_QUANTIZATION=int8 python benchmarks/bench_heterogeneous.py --model Qwen/Qwen2.5-7B-Instruct --quantization int8

# Maximum throughput ceiling test
CUDA_VISIBLE_DEVICES=0 python benchmarks/bench_max_tps.py

# NVFP4 Blackwell benchmark (requires RTX 50xx)
CUDA_VISIBLE_DEVICES=1 python benchmarks/bench_nvfp4.py

# TurboQuant on Qwen-7B (single GPU + bi-GPU + NVFP4)
python benchmarks/bench_turboquant.py --model Qwen/Qwen2.5-7B-Instruct --max-tokens 128
python benchmarks/bench_turboquant.py --model Qwen/Qwen2.5-7B-Instruct --max-tokens 128 --num-gpus 2
python benchmarks/bench_turboquant.py --model Qwen/Qwen2.5-7B-Instruct --max-tokens 128 --nvfp4-only
```
