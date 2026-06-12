# VRAMancer Compatibility Matrix

> Last updated: 2026-04 · Version 1.5.0

## Backend × Quantization

| Backend | BF16 / FP16 | NF4 (BnB) | INT8 (BnB) | NVFP4 (Blackwell) | GGUF (Q4/Q8…) |
|---------|:-----------:|:---------:|:----------:|:-----------------:|:-------------:|
| HuggingFace | ✅ | ✅ (single-GPU only¹) | ✅ (single-GPU only¹) | ✅ CC≥10.0 only | ❌ |
| vLLM | ✅ | ✅ | ✅ | ⚠️ experimental | ❌ |
| Ollama | ✅ | ❌ | ❌ | ❌ | ✅ |
| llama.cpp / llama-server | ✅ | ❌ | ❌ | ❌ | ✅ |

¹ **BnB multi-GPU upstream bug** (accelerate 1.13.0 + BnB 0.49.2 + transformers 5.3.0):
`AlignDevicesHook` does not handle residual connections across devices with quantized layers.
VRAMancer forces single-GPU when BnB quantization is requested.

## Backend × OS

| Backend | Linux | macOS (Apple Silicon) | Windows |
|---------|:-----:|:--------------------:|:-------:|
| HuggingFace | ✅ CUDA / ROCm | ✅ MPS | ✅ CUDA |
| vLLM | ✅ | ❌ (no official support) | ❌ |
| Ollama | ✅ | ✅ | ✅ |
| llama.cpp | ✅ | ✅ Metal | ✅ CUDA |
| WebGPU (experimental) | ⚠️ Chrome/Firefox | ⚠️ | ⚠️ |

## Accelerator × Backend

| Accelerator | HuggingFace | vLLM | Ollama | llama.cpp |
|-------------|:-----------:|:----:|:------:|:---------:|
| NVIDIA CUDA | ✅ | ✅ | ✅ | ✅ |
| AMD ROCm (HIP) | ✅ | ✅ | ✅ | ✅ |
| Apple MPS | ✅ | ❌ | ✅ | ✅ Metal |
| Intel XPU (oneAPI) | ✅ | ⚠️ | ❌ | ❌ |
| CPU (fallback) | ✅ | ❌ | ✅ | ✅ |

## Multi-GPU Strategies

| Strategy | Supported Backends | Notes |
|----------|-------------------|-------|
| Pipeline Parallel (PP) | HuggingFace | Default. Layers split proportionally to free VRAM. |
| Tensor Parallel (TP) | HuggingFace | `VRM_PARALLEL_MODE=tp`. NCCL required. GPT-2, Llama, Qwen tested. |
| vLLM native | vLLM | Handled internally by vLLM. PP/TP both supported. |
| llama-server tensor-split | llama.cpp | `--tensor-split` flag. Local GPUs only (no RPC split). |
| RPC remote nodes | llama.cpp | `--rpc host:port`. Requires `llama-rpc-server` on remote. |

## KV Cache Compression

| Method | Flag | Reduction | Backends | Notes |
|--------|------|-----------|----------|-------|
| TurboQuant (PolarQuant+QJL) | `VRM_KV_COMPRESSION=turboquant` | ~4.6× | HuggingFace | ~3.5 bits/dim. Triton GPU kernels. |
| None | (default) | 1× | all | Full precision KV. |

## VRAM Lending Pool

| Condition | Available |
|-----------|-----------|
| Multi-GPU, HuggingFace backend | ✅ default-on |
| vLLM / llama.cpp backend | ❌ (they manage own VRAM) |
| Single GPU | ❌ |
| Disable explicitly | `VRM_VRAM_LENDING=0` |

## Known Limitations

- **NVFP4** requires Compute Capability ≥ 10.0 (Blackwell, RTX 50xx). Falls back to BF16 on older GPUs.
- **BnB quantization** is forced to single-GPU due to upstream bug (see footnote above).
- **VM / IOMMU (Proxmox)**: P2P (Strategy 1.5) blocked by IOMMU. Only CPU-staged transfer (Strategy 4) works. ~10–15% overhead.
- **ROCm-SMI fallback** in `core/monitor.py` is not tested on real AMD hardware.
- **WebGPU backend** is experimental/POC — not suitable for production workloads.
- **Windows**: Triton not available → TurboEngine (torch.compile) disabled. GGUF via llama.cpp recommended.
