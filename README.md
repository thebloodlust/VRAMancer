# VRAMancer

VRAMancer is a heterogeneous multi-GPU Python orchestrator for Large Language Model (LLM) inference. It provides load balancing and tensor operations across mismatched GPU architectures natively.

## Key Features

- **Asymmetric GPU Tensor Parallelism:** Distribute large models (14B, 32B, 70B parameters) across GPUs of varying sizes and compute capabilities via adaptive allocation.
- **Cross-Architecture Inference Support:** Enables FP16/BF16 activation transfer between newer (e.g., Blackwell/NVFP4) and older (e.g., Ampere) GPU architectures over native PCIe using a Rust-based P2P bridge, bypassing strict hardware incompatibility issues.
- **Hierarchical Memory Management:** Manages weight and activation transfers across VRAM, pinned memory, and NVMe based on available compute and bandwidth.
- **Wake-On-Inference:** Supports network-level wake-on-LAN (WoL) triggers for dormant inference nodes before generation requests.
- **RESTful API:** Provides an OpenAI-compatible API endpoint (`/v1/completions`, `/api/generate`) for integration with standard frontends and services.
- **Web Interface:** Includes a real-time visualization dashboard and chat completion interface with Server-Sent Events (SSE) streaming support.

## Performance (Verified Benchmarks)

Single-GPU benchmarks on RTX 3090, CUDA 12.8, PyTorch 2.10, Mistral-7B/TinyLlama/GPT-2:

| Model | Native HuggingFace | VRAMancer | Delta |
|-------|-------------------|-----------|-------|
| GPT-2 (124M) | 126.8 tok/s | 126.5 tok/s | -0.2% |
| TinyLlama-1.1B | 53.1 tok/s | 57.3 tok/s | **+7.9%** |
| Mistral-7B-v0.1 | 35.7 tok/s | 36.2 tok/s | **+1.4%** |

Full details: [`benchmarks/BENCHMARK_RESULTS.md`](benchmarks/BENCHMARK_RESULTS.md)

**Not yet benchmarked:** Multi-GPU throughput (blocked by CUDA TDR in VM), VRAMancer vs vLLM comparison.

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA / ROCm / MPS compatible hardware

### Setup

```bash
git clone https://github.com/vramanccer/VRAMancer.git
cd VRAMancer
pip install -r requirements.txt
```

### Running the Orchestrator

Start the main orchestrator and the API server:

```bash
python -m vramancer.main --port 8000
```

To use the fallback WebGPU execution engine or configure the cluster, adjust settings in `config.yaml` or through environment variables (e.g., `VRM_BACKEND_ALLOW_STUB=1`).

## Development and Testing

Testing is managed via `pytest`:

```bash
VRM_MINIMAL_TEST=1 pytest tests/
```

## Architecture Summary

VRAMancer routes incoming HTTP requests to a singleton `InferencePipeline`. This pipeline provisions memory blocks, assesses GPU capabilities (like NVFP4 computation and VRAM capacity ratios, targeting 80% compute scoring / 20% VRAM balancing), and dispatches operations natively. Transfers between devices are executed securely via asynchronous, zero-copy mechanisms.

---

*For detailed architectural references and metrics handling, review the `docs/` and `core/` directories.*
