# VRAMancer

Run LLM models that don't fit on a single GPU, across heterogeneous GPUs, in one command.

```bash
# Load a 14B model across a RTX 3090 + RTX 5070 Ti (neither alone has enough VRAM)
vramancer run Qwen/Qwen2.5-14B-Instruct

# With 4-bit quantization (fits on a single GPU, 75% faster)
vramancer run Qwen/Qwen2.5-14B-Instruct -q nf4

# One-shot generation
vramancer run Qwen/Qwen2.5-7B-Instruct -p "Explain gradient descent in 3 sentences"
```

VRAMancer auto-detects all GPUs, splits the model proportionally to available VRAM, and runs inference block-by-block with CPU-staged transfers between GPUs. No config files, no YAML, no manual device maps.

## Proven: What it actually does

**Heterogeneous multi-GPU inference** — the core feature, benchmarked:

| Model | Setup | Result |
|-------|-------|--------|
| Qwen2.5-14B (28 GB bf16) | Single RTX 3090 (24 GB) | **OOM** |
| Qwen2.5-14B (28 GB bf16) | Single RTX 5070 Ti (16 GB) | **OOM** |
| Qwen2.5-14B (28 GB bf16) | VRAMancer 2-GPU (3090 + 5070 Ti) | **6.0 tok/s** ✓ |
| Qwen2.5-14B NF4 | Single GPU | **10.5 tok/s** (75% faster, 10.8 GB) |
| Qwen2.5-7B GGUF Q4_K_M | llama.cpp backend | **106.8 tok/s** (3.0 GB) |

_Hardware: RTX 3090 + RTX 5070 Ti, Proxmox VM, PCIe passthrough, CPU-staged transfers ~11 GB/s._
_Bare-metal expected +10-30% with P2P/NVLink._

**Single-GPU performance** — near-zero overhead:

| Model | HuggingFace native | VRAMancer | Delta |
|-------|-------------------|-----------|-------|
| GPT-2 (124M) | 123.4 tok/s | 125.6 tok/s | +1.8% |
| TinyLlama-1.1B | 53.0 tok/s | 56.5 tok/s | **+6.6%** |
| Mistral-7B-v0.1 | 35.1 tok/s | 34.9 tok/s | -0.6% |

Full benchmark details: [benchmarks/BENCHMARK_RESULTS.md](benchmarks/BENCHMARK_RESULTS.md)

## Install

```bash
git clone https://github.com/vramancer/VRAMancer.git
cd VRAMancer
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch 2.1+, CUDA (or ROCm/MPS).

## Usage

### One-command inference (recommended)

```bash
# Interactive mode — loads model, then prompts you
vramancer run Qwen/Qwen2.5-7B-Instruct

# One-shot with prompt
vramancer run mistralai/Mistral-7B-v0.1 -p "What is VRAM?" --max-tokens 128

# With quantization (nf4, int8, nvfp4 for Blackwell)
vramancer run Qwen/Qwen2.5-14B-Instruct -q nf4

# Force specific GPU count
vramancer run meta-llama/Llama-3-8B --gpus 2

# GGUF via llama.cpp backend
vramancer run TheBloke/Mistral-7B-v0.1-GGUF --backend llamacpp
```

### API server (OpenAI-compatible)

```bash
# Start server with model pre-loaded
vramancer serve --model Qwen/Qwen2.5-7B-Instruct --port 5030

# Then query it
curl http://localhost:5030/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VRM_API_TOKEN" \
  -d '{"prompt": "Hello", "max_tokens": 64}'
```

### Other commands

```bash
vramancer status      # Show GPUs, memory, backend
vramancer health      # System health check
vramancer hub Qwen/Qwen2.5-14B-Instruct  # Browse model formats on HF
vramancer benchmark   # GPU matmul benchmark
vramancer split Qwen/Qwen2.5-14B-Instruct --gpus 2  # Preview model split
```

## How it works

1. **Auto-detect GPUs** — enumerates all CUDA/ROCm/MPS devices with free VRAM
2. **VRAM-proportional split** — assigns model layers to GPUs based on available memory (e.g., 23 layers to 24 GB GPU, 28 layers to 16 GB GPU)
3. **Block-by-block inference** — sequential forward pass through blocks, transferring activations between GPUs via CPU-staged pinned memory
4. **Quantization** — optional NF4/INT8/NVFP4 to reduce VRAM footprint
5. **KV cache management** — paged attention with optional PolarQuant+QJL compression (~3.5 bits/dim, ~4.6x reduction)

## Backends

| Backend | Install | Best for |
|---------|---------|----------|
| **huggingface** (default) | `pip install transformers accelerate` | General use, multi-GPU split |
| **llamacpp** | `pip install llama-cpp-python` | GGUF models, fastest small models |
| **vllm** | `pip install vllm` | High-throughput serving |
| **ollama** | Install [Ollama](https://ollama.ai) | Easy local models |

## Configuration

VRAMancer is configured via environment variables (`VRM_*`), not config files:

```bash
VRM_QUANTIZATION=nf4          # Quantization mode
VRM_KV_COMPRESSION=turboquant # KV cache compression (~4.6x reduction)
VRM_PARALLEL_MODE=pp           # pp (pipeline) or tp (tensor parallel)
VRM_API_TOKEN=your-token       # API authentication
VRM_PRODUCTION=1               # Strict security mode
```

Full list: [.github/copilot-instructions.md](.github/copilot-instructions.md#variables-denvironnement-essentielles)

## Development

```bash
# Tests (901 passed, works without GPU)
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 pytest tests/ -q

# Lint
flake8 core/ tests/
```

## Architecture

```
vramancer run model
    └─ InferencePipeline
        ├─ backends.select_backend()  → HuggingFace / vLLM / Ollama / llama.cpp
        ├─ model_splitter             → VRAM-proportional layer assignment
        ├─ TransferManager            → GPU-to-GPU data movement (P2P or CPU-staged)
        ├─ StreamManager              → prefetch, swap, eviction
        └─ continuous_batcher         → batched inference (optional)
```

Detailed architecture: [docs/architecture.md](docs/architecture.md)

## License

MIT
