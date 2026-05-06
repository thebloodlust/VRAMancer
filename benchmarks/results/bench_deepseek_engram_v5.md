# DeepSeek V4 Flash + engram KV offload bench (V5 P13→P16)

> **Phase 1 PASS**: `deepseek-ai/DeepSeek-V4-Flash` tilelang FP8 GEMM kernel compiles and executes on RTX 5070 Ti (SM 12.0) via three `.venv` patches (see bench_deepseek_engram_v5.json).
> Phase 3 download in progress (~159 GB). Phases 4–5 (convert + load test) pending.
>
> **PARTIAL@P13.1**: Proxy run while blocked on hardware/toolchain. Target was `deepseek-ai/DeepSeek-V4-Flash` (158B params).
> 158B >> 40GB VRAM (RTX 5070 Ti 16GB + RTX 3090 24GB). Proxy: `Qwen/Qwen2.5-7B-Instruct`.

**Model:** `Qwen/Qwen2.5-7B-Instruct`
**GPUs:** RTX 5070 Ti (16GB) + RTX 3090 (24GB) — 40 GB VRAM total
**KV offload:** engram DRAM store (cap 200 GB) via VRM_KV_OFFLOAD_ENGRAM=1

| Context | Actual tok | tok/s | VRAM Δ G0 MB | VRAM Δ G1 MB | DRAM Δ MB | Offload evictions |
|---------|-----------|-------|--------------|--------------|-----------|------------------|
| 512 | 512 | 14.00 | 140 | 114 | 1560 | 0 |
| 2048 | 2048 | 20.72 | 278 | 278 | 0 | 0 |
| 4096 | 4096 | 16.50 | 446 | 444 | 0 | 0 |
| 8192 | 8192 | 11.49 | 888 | 888 | 0 | 0 |
| 16384 | 16384 | 6.77 | 1776 | 1776 | 0 | 0 |

**Engram offload active:** yes (VRM_KV_OFFLOAD_ENGRAM=1)
**DRAM Δ > 0 at ctx:** see table — first row where DRAM Δ > 0 is offload inflection