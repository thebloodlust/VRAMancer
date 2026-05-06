# DeepSeek + engram KV offload bench (V5 P13)

> **PARTIAL@P13.1**: Target was `deepseek-ai/DeepSeek-V4-Flash` (158B params).
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