# Index des résultats benchmarks (racine du repo)

Fichiers de résultats bruts en racine du dépôt. Générés lors des sessions de benchmark des sprints V3/V4.

## Fichiers JSON (résultats structurés)

| Fichier | Description |
|---------|-------------|
| `bench_gpt2_cuda_turboquant.json` | GPT-2 124M + TurboQuant — baseline CUDA |
| `bench_tinyllama_cuda_turboquant.json` | TinyLlama-1.1B + TurboQuant — 1-GPU CUDA |
| `bench_7b_1gpu_cuda_turboquant.json` | Qwen2.5-7B + TurboQuant — 1-GPU CUDA |
| `bench_7b_bigpu_cuda_turboquant.json` | Qwen2.5-7B + TurboQuant — 2-GPU hétérogène |
| `bench_14b_bigpu_turboquant.json` | Qwen2.5-14B + TurboQuant — 2-GPU hétérogène |
| `bench_14b_bigpu_cuda_turboquant.json` | Qwen2.5-14B + TurboQuant + CUDA Stream — 2-GPU |
| `bench_5070ti_nvfp4_turboquant.json` | RTX 5070 Ti — NVFP4 + TurboQuant (Blackwell) |
| `bench_bigpu_nvfp4_turboquant.json` | 2-GPU — NVFP4 + TurboQuant |
| `bench_bigpu_turboquant_nvfp4.json` | 2-GPU — TurboQuant + NVFP4 (variante) |
| `bench_kv_migration.json` | KV cache migration impact — latence P2P |
| `bench_p2p_impact.json` | P2P bandwidth + latency RTX 3090 ↔ RTX 5070 Ti |
| `bench_3node_result.json` | VTP 3-node cluster (Linux + macOS MLX) |
| `bench_wan_4g_jeje1_synology_me.json` | WAN 4G inference test (jeje1 ↔ synology.me) |

## Fichiers TXT (logs bruts)

| Fichier | Description |
|---------|-------------|
| `bench_gpt2_out.txt` | GPT-2 baseline stdout |
| `bench_micro_out.txt` | Micro-benchmark (matmul, kernel launch) |
| `bench_tinyllama_out.txt` | TinyLlama stdout |
| `bench_tinyllama_1gpu_out.txt` | TinyLlama 1-GPU run stdout |
| `bench_qwen7b_out.txt` | Qwen2.5-7B BF16 stdout |
| `bench_qwen7b_nvfp4_out.txt` | Qwen2.5-7B NVFP4 stdout |
| `bench_webgpu_gpt2_out.txt` | WebGPU GPT-2 POC stdout |
| `bench_webnpu_gpt2_out.txt` | WebNPU GPT-2 POC stdout |
| `bench_webnpu_ipadm4_out.txt` | WebNPU iPad M4 stdout |

## Hardware de référence

- **GPU0 (torch FAST_FIRST order):** NVIDIA GeForce RTX 3090 — 24 GB GDDR6X
- **GPU1:** NVIDIA GeForce RTX 5070 Ti — 16 GB GDDR7 (Blackwell, CC 10.0)
- **CPU:** AMD EPYC 7413P (Proxmox VM)
- **OS:** Linux 6.x, CUDA 13.0 (driver), torch 2.11.0+cu130

## Scripts de benchmark correspondants

Voir `benchmarks/` pour les scripts qui ont généré ces résultats.
