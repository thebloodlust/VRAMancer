# VRAMancer — Liste complète des fonctionnalités

> Version 1.5.0 — 26 mars 2026  
> Orchestrateur multi-GPU hétérogène pour l'inférence LLM distribuée

---

## Inférence multi-GPU hétérogène

- **Split VRAM-proportionnel** — Répartition automatique des couches basée sur la VRAM libre de chaque GPU (`model_splitter.py`)
- **Inférence cross-architecture** — RTX 3090 (Ampere) + RTX 5070 Ti (Blackwell) dans le même pipeline
- **CPU overflow** — Les couches qui ne tiennent sur aucun GPU sont automatiquement placées en RAM
- **Pipeline parallélisme** — Forward séquentiel bloc-par-bloc avec transfers GPU-to-GPU entre les couches
- **Tensor Parallélisme** — Column/row-parallel + NCCL all-reduce pour GPT-2, Llama, Qwen (`tensor_parallel.py`)
- **Continuous Batching** — Token-level batching avec chunked prefill (`continuous_batcher.py`, `VRM_CONTINUOUS_BATCHING=1`)
- **Speculative Decoding** — Auto-mapping draft models par famille (Qwen→0.5B, Llama→1B) (`speculative_decoding.py`)

## Backends LLM

- **HuggingFace** — Backend principal avec `transformers`, split automatique via `accelerate` (`backends.py`)
- **llama.cpp / GGUF** — Via `llama-cpp-python`, multi-GPU tensor_split, HF Hub auto-download Q4_K_M (`backends_llamacpp.py`)
- **vLLM** — Wrapper avec retry OOM (`backends_vllm.py`)
- **Ollama** — REST bridge synchrone (`backends_ollama.py`)
- **Stub** — Mode test sans dépendances lourdes (`VRM_BACKEND_ALLOW_STUB=1`)

## Quantization

- **BF16** — Précision par défaut
- **NF4** — 4-bit NormalFloat via BitsAndBytes. **75% plus rapide** que BF16 pour 14B (modèle tient sur 1 GPU)
- **INT8** — LLM.int8 via BitsAndBytes
- **NVFP4 Blackwell** — Quantization native FP4 via torchao sur CC≥10.0 (RTX 5070 Ti). Kernel cublas `_scaled_mm` avec `float4_e2m1fn_x2`
- **DirectFP4 Bypass** — Remplace les NVFP4Tensor par des plain buffers + `_scaled_mm` direct, éliminant le overhead `__torch_dispatch__`. **+7% vs torchao**, 0 VRAM extra (`nvfp4_direct.py`)
- **GGUF Q4_K_M** — Via llama.cpp, kernels dp4a INT8 natifs. **5.4x plus rapide** que BnB NF4, **2.2x moins de VRAM**
- **Triton GEMV NVFP4** — Kernel Triton pour GEMV avec nibble unpacking, per-device LUT (`triton_gemv_nvfp4.py`)

## KV Cache & Mémoire

- **Paged Attention** — Cache KV paginé avec pages physiques/virtuelles, allocation dynamique (`paged_attention.py`)
- **Kernel CUDA PagedAttention** — Warp-level online softmax, GQA, fp16/fp32. **8.8x vs PyTorch @ctx64** (`csrc/paged_attention_kernel.cu`)
- **TurboQuant KV Compression** — Walsh-Hadamard → polar recursif → QJL 1-bit sur résidu. **~3.5 bits/dim, ~4.6x réduction** (`turboquant.py`, `VRM_KV_COMPRESSION=turboquant`)
- **VRAM Lending** — Pool coopératif de prêt VRAM inter-GPU : leases, scoring, reclaim automatique (`vram_lending.py`, `VRM_VRAM_LENDING=1`)
- **Hierarchical Memory** — 6 niveaux : VRAM → DRAM → NVMe → réseau, scoring LRU/LFU hybride (`hierarchical_memory.py`)
- **Stream Manager** — Prefetch, swap, eviction avec monitoring background (`stream_manager.py`)

## Transport GPU-to-GPU

Chaîne de stratégies (fallback automatique) :

1. **Strategy 0 — Cross-Vendor Bridge** — AMD↔NVIDIA via PipelinedTransport double-buffer async
2. **Strategy 1 — CUDA P2P** — `cudaMemcpyPeerAsync()` direct si P2P accessible
3. **Strategy 1.5 — Rust DtoD/Pipeline** — Bypass via `vramancer_rust` async pipeline (Tokio + CUDA FFI)
4. **Strategy 1.7 — ReBAR Full-Window** — Activation si BAR0 > 4 GB, DMA chunks jusqu'à 64 MB (`cross_vendor_bridge.py`)
5. **Strategy 2 — CPU-Pipelined** — Double-buffer GPU→pinned→GPU en recouvrement
6. **Strategy 3 — NCCL** — Distribué multi-nœuds (quand MASTER_ADDR set)
7. **Strategy 4 — CPU-staged** — Fallback simple via RAM pinned (seule option en VM Proxmox IOMMU)

## API & Sécurité

- **API Flask OpenAI-compatible** — `/v1/completions`, `/api/generate`, `/api/infer`, `/api/models/load` (`production_api.py`)
- **Authentification** — Token HMAC-SHA256 + JWT + PBKDF2 (`security/__init__.py`, `auth_strong.py`)
- **RBAC** — Contrôle d'accès par rôle
- **Rate Limiting** — Par route, désactivable via `VRM_DISABLE_RATE_LIMIT=1`
- **Circuit Breaker** — Protège timeout et SSE (`api/circuit_breaker.py`)
- **CSP** — Content Security Policy avec whitelist CDN spécifiques, sans `unsafe-eval`

## Monitoring & Observabilité

- **GPUMonitor** — Polling VRAM, détection overload, ROCm-SMI fallback (`monitor.py`)
- **Métriques Prometheus** — ~35 compteurs/gauges (`metrics.py`)
- **Health Checks** — Diagnostics composites (`health.py`)
- **GPU Fault Tolerance** — State machine HEALTHY→DEGRADED→FAILED→OFFLINE avec probe et recovery (`gpu_fault_tolerance.py`)
- **Layer Profiler** — Profiling par couche (latence, FLOPS, mémoire), placement DP-optimal (`layer_profiler.py`)
- **OpenTelemetry** — Tracing distribué (`VRM_TRACING=1`)
- **Grafana Dashboard** — Dashboard pré-configuré (`monitoring/`)

## Réseau & Cluster

- **Cluster Discovery** — mDNS + UDP broadcast + Bully leader election, JSONL membership (`cluster_discovery.py`)
- **RDMA Transport** — RDMA verbs (ibverbs/RoCE), GPUDirect RDMA si nvidia_peermem, zero-copy TCP (`fibre_fastpath.py`)
- **NAT Traversal** — STUN RFC 5389, UDP hole punch, relay, ULA IPv6 (`nat_traversal.py`)
- **Transport Factory** — Sélection automatique NCCL (même nœud) ou RDMA/TCP (réseau) selon localité (`transport_factory.py`)
- **AITP Protocol** — AI Transport Protocol UDP/IPv6 avec FEC Reed-Solomon (`aitp_protocol.py`, `aitp_fec.py`)
- **Wake-on-Inference** — Réveil de nœuds dormants via WoL magic packets (`wake_on_inference.py`)

## Extension Rust (vramancer_rust)

- **GpuPipeline** — Pipeline persistant avec streams/events/pinned buffers pré-alloués
- **Async GPU Transfer** — DtoH/HtoD overlappé sur CUDA streams séparés via libloading FFI
- **TCP Chunked Transfer** — Tokio async avec HMAC-SHA256 par chunk
- **XOR Parity** — Erasure coding pour résilience
- **HMAC Crypto** — 100× plus rapide que Python hmac

## Configuration

- **Multi-OS** — XDG_CONFIG_HOME (Linux), ~/Library (macOS), %APPDATA% (Windows)
- **Hiérarchique** — defaults → config.yaml → env vars `VRM_*`
- **Hot-reload** — Rechargement config sans redémarrage
- **Auto-détection GPU** — Hétérogène, VM, P2P, tiers, politique lending (`hetero_config.py`)
- **Multi-accélérateur** — CUDA, ROCm, MPS, CPU via `detect_backend()` (`utils.py`)

---

## Benchmarks vérifiés (RTX 3090 + RTX 5070 Ti, Proxmox VM)

| Test | Résultat |
|------|----------|
| Qwen2.5-14B 2-GPU (impossible sur 1 GPU) | **6.0 tok/s** |
| Qwen2.5-14B NF4 single GPU | **10.5 tok/s** (+75% vs BF16 2-GPU) |
| GGUF Q4_K_M 7B (llama.cpp) | **106.8 tok/s** |
| DirectFP4 vs torchao NVFP4 | **+7% speedup**, 0 VRAM extra |
| CUDA kernel PagedAttention | **8.8x** vs PyTorch @ctx64 |
| TurboQuant KV compression | **~4.6x** reduction, ~3.5 bits/dim |
| Tests | **853 passed**, 38 skipped, 0 failed |
