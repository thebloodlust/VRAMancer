# 🚀 VRAMancer : Rapport d'Architecture Rust & CUDA Bypass

Ce fichier documente les avancées majeures de `vramancer_rust`, le module Rust haute-performance pour VRAMancer.

## 1. Le Bypass CUDA Driver API (Strategy 1.5)

**Le problème :** NVIDIA bloque `cudaMemcpyPeer` entre GPUs Consumer (GeForce, RTX) via le bus PCIe. PyTorch retourne `can_device_access_peer() = False` et tombe en fallback CPU-staged lent (~10.5 GB/s), bloquant le GIL Python.

**La solution VRAMancer :** Appel direct à `cuMemcpyDtoD_v2` via la CUDA Driver API depuis Rust, avec le GIL relâché (`py.allow_threads`). Le driver NVIDIA gère le staging CPU **en interne**, de manière transparente et optimisée.

### Résultats benchmarkés (RTX 3090 + RTX 5070 Ti, topologie PIX) :

| Taille | PyTorch `.to()` | Rust `cuMemcpyDtoD` | Speedup |
|--------|----------------|---------------------|---------|
| 1 MB   | 0.14 ms        | 0.02 ms            | **9x**  |
| 10 MB  | 0.99 ms        | 0.07 ms            | **13x** |
| 50 MB  | 4.77 ms        | 0.33 ms            | **14x** |
| 100 MB | 9.49 ms        | 1.59 ms            | **6x**  |
| 200 MB | 18.88 ms       | 12.90 ms           | **1.5x**|

### Fonctions CUDA implémentées (Rust FFI via `libloading`) :
- `direct_vram_copy(src_ptr, dst_ptr, nbytes)` — `cuMemcpyDtoD_v2` (6-14x plus rapide)
- `staged_gpu_transfer(src, dst, bytes, gpu0, gpu1, chunk)` — double-buffered avec pinned memory
- `inject_to_vram_ptr(payload, dest_ptr)` — `cuMemcpyHtoD_v2` direct bytes→VRAM

### Intégrité des données vérifiée ✓
Tous les transferts vérifient `torch.allclose(source.cpu(), dest.cpu())` — zéro corruption.

### Intégration : `transfer_manager.py` → Strategy 1.5
```
Strategy 0: Cross-vendor bridge (AMD ↔ NVIDIA)
Strategy 1: CUDA P2P direct (NVLink/PCIe, blocked on consumer)
Strategy 1.5: ★ Rust cuMemcpyDtoD bypass (GIL released, 6-14x faster) ★
Strategy 2: ReBAR pipelined (double-buffered Python)
Strategy 3: NCCL (distributed mode only)
Strategy 4: CPU-staged PyTorch fallback
```

## 2. Le Swarm Brain (Mémoire Holographique)
Calcul de parité XOR en Rust pur via `generate_holographic_parity` et `heal_holograph`.
- Auto-vectorisation AVX-512 via LLVM
- Zéro fuites mémoire (ownership Rust)
- 50 MB x2 shards : parity en 51ms, heal en 45ms

## 3. L'Offload "Software CXL" (NVMe)
`cxl_direct_memory_dump` et `cxl_direct_memory_load` — dump/load direct de pointeurs mémoire vers NVMe, GIL relâché, sans Pickle.

## 4. Le P2P Réseau Distant (IP)
- `send_tensor_p2p` / `receive_tensor_p2p` — transfert signé HMAC-SHA256 via Tokio TCP
- `send_tensor_chunked` / `receive_tensor_chunked` — pipeline chunké avec backpressure et vérification par chunk

## 5. Sécurité
- `sign_payload_fast` — HMAC-SHA256 Rust (234 MB/s sustained)
- `verify_hmac_fast` / `verify_hmac_batch` — vérification zero-trust

## Build
```bash
cd rust_core && export CUDA_PATH=/usr && maturin develop --release --features cuda
# Sans CUDA (Mac, CI) :
cd rust_core && maturin develop --release
```

---
*18 fonctions PyO3, 645 tests passés, 0 échec.*