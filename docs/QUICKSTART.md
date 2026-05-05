# VRAMancer Quickstart — 5 minutes from zero to inference

## Prerequisites

- Python 3.10+ (3.12 recommended)
- NVIDIA GPU with CUDA 12.1+ (or AMD ROCm 6+, or Apple Silicon MPS)
- 16 GB RAM minimum

## Install

```bash
# From PyPI
pip install vramancer

# Or from source (recommended for latest features)
git clone https://github.com/thebloodlust/VRAMancer
cd VRAMancer && pip install -e .
```

## Verify GPU detection

```bash
vramancer health
# Lists your GPU(s) with VRAM and compute capability.
# Example output:
#   [0] NVIDIA RTX 5070 Ti  16303 MiB  cuda  CC 12.0
#   [1] NVIDIA RTX 3090      24576 MiB  cuda  CC 8.6
```

## Run inference (3 commands)

### 1. Single-GPU, BF16 (quickest start)

```bash
vramancer serve gpt2 --port 8000
```

### 2. Multi-GPU with auto-split (for large models)

```bash
vramancer serve Qwen/Qwen2.5-7B --num-gpus 2 --port 8000
# VRAMancer splits layers proportionally across GPUs by free VRAM.
```

### 3. Quantized (NF4 — fits on an 8 GB GPU)

```bash
vramancer serve mistralai/Mistral-7B-v0.1 --quantization nf4 --port 8000
# NF4 = 4-bit NormalFloat (bitsandbytes). Reduces VRAM ~4x with minimal quality loss.
```

### 4. GGUF via llama.cpp (fastest single-GPU, no CUDA compile needed)

```bash
vramancer serve Qwen/Qwen2.5-7B-Instruct-GGUF --backend llamacpp --port 8000
```

## Test the API

```bash
# Chat completion (OpenAI-compatible)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt2",
    "messages": [{"role": "user", "content": "Hello, what are you?"}],
    "max_tokens": 100
  }'

# Health check
curl http://localhost:8000/health
```

VRAMancer is **100% OpenAI-compatible** — all standard SDKs work:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
response = client.chat.completions.create(
    model="gpt2",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

## Multi-GPU model that doesn't fit on a single GPU

VRAMancer's killer feature — run a 14B model across two consumer GPUs:

```bash
# Qwen2.5-14B needs ~28 GB VRAM in BF16 → fits across RTX 3090 (24 GB) + RTX 5070 Ti (16 GB)
vramancer serve Qwen/Qwen2.5-14B --num-gpus 2 --port 8000
```

## Environment variables (optional)

| Variable | Effect |
|----------|--------|
| `VRM_QUANTIZATION=nf4` | 4-bit NormalFloat (same as `--quantization nf4`) |
| `VRM_QUANTIZATION=nvfp4` | FP4 for Blackwell GPUs (RTX 5xxx, CC≥10.0) |
| `VRM_CONTINUOUS_BATCHING=1` | Enable vLLM-style continuous batching for multi-user |
| `VRM_PARALLEL_MODE=tp` | Tensor Parallel instead of Pipeline Parallel |

## Next steps

- **Multi-node cluster** : `docs/CLUSTER_SETUP.md`
- **Production deployment** : `docs/PRODUCTION.md`
- **Backend matrix** (HF / vLLM / llama.cpp / Ollama) : `docs/COMPATIBILITY.md`
- **ReBAR Proxmox setup** for near-bare-metal performance : `docs/reports/REBAR_PROXMOX_BENCHMARK.md`
- **API reference** : `http://localhost:8000/docs` (when running)
