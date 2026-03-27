# VRAMancer — Roadmap Performances

> Derniere mise a jour : session 23 mars 2026
> Hardware de reference : RTX 3090 (24 GB, PCIe 4.0 x16) + RTX 5070 Ti (16 GB, PCIe 5.0 x16), VM Proxmox VFIO

---

## Baselines mesurees (point de depart)

| Metrique | Valeur actuelle | Conditions |
|---|---|---|
| Transfer GPU→GPU (pipelined) | 22 GB/s | PipelinedTransport, 8 MB chunks, CPU-staged |
| Transfer GPU→GPU (PyTorch .to()) | ~11 GB/s | Simple .to(device), synchrone |
| Transfer GPU→GPU (Rust GpuPipeline) | ~14 GB/s* | *Non mesure en prod — extrapole 1.3x baseline |
| NVFP4 Direct (Qwen2.5-7B, 5070 Ti) | 12.0 tok/s | DirectFP4 bypass, torchao 0.16 |
| NVFP4 torchao natif | 11.2 tok/s | torchao NVFP4Tensor dispatch |
| BF16 (Qwen2.5-7B, 5070 Ti) | 36.4 tok/s | Baseline pleine precision |
| BnB NF4 (Qwen2.5-7B, 3090) | 20.2 tok/s | Single-GPU, bitsandbytes |
| GGUF Q4_K_M (Qwen2.5-7B, 3090) | 106.8 tok/s | llama-cpp-python dp4a kernels |
| Qwen2.5-14B 2-GPU BF16 | 6.0 tok/s | Split 57%/43%, CPU-staged transfers |
| Reseau inter-noeuds (RDMA) | ~3 μs latence | ibverbs QP, mesure locale |
| Reseau inter-noeuds (TCP) | ~50 μs latence | Tokio zero-copy TCP |

---

## Chantier 1 — Rust/Tokio Transfer Pipeline

### Etat actuel du code

`rust_core/src/lib.rs` — **1077 LOC, fonctionnel**

Ce qui **existe et marche** :
- `cuda_ffi` : FFI directe vers libcuda.so.1 (memcpy DtoD/DtoH/HtoD sync+async, streams, events, contexts, pinned mem alloc/free)
- `staged_copy_double_buffered()` : transfer synchrone double-buffer via pinned memory
- `async_staged_transfer()` : transfer asynchrone avec overlap DtoH/HtoD sur streams separes + events
- `GpuPipeline` (PyO3 class) : pipeline persistant pre-alloue (streams, events, pinned buffers), `transfer()` avec GIL release
- `direct_vram_copy()` : cuMemcpyDtoD_v2 brut
- `send_tensor_p2p/receive_tensor_p2p` : Tokio TCP + HMAC-SHA256
- `send_tensor_chunked/receive_tensor_chunked` : pipeline chunke avec backpressure + signature par chunk
- `GpuPipeline::Drop` : cleanup propre (streams, events, pinned bufs)

Ce qui **manque** :
- cuMemcpyPeerAsync (P2P entre GPUs — le DtoD actuel echoue si P2P bloque par IOMMU)
- Multi-buffer pipeline (3+ buffers pour saturer PCIe bidirectionnel)
- Benchmark integre pour valider les gains vs PyTorch .to()
- Integration dans `transfer_manager.py` Strategy 1.5 : le code Python essaie `vramancer_rust.direct_vram_copy()` mais n'utilise PAS `GpuPipeline` correctement (il faut passer par `_get_gpu_pipeline()`)

### Objectif

| Metrique | Baseline | Target | Comment |
|---|---|---|---|
| Latence small tensor (< 1 MB) | ~80 μs (PyTorch) | ~50 μs | cuMemcpyDtoD avec ctx pre-cached |
| Throughput large tensor (> 100 MB) | 22 GB/s (PipelinedTransport Python) | 24-26 GB/s | Rust async pipeline, 3 buffers, 32 MB chunks |
| GIL hold time pendant transfer | ~100% (PyTorch .to()) | 0% | Deja le cas avec GpuPipeline |

### Plan d'implementation

1. **Ajouter `cuMemcpyPeerAsync`** dans `cuda_ffi` — permet le P2P hardware quand dispo (bare metal)
2. **Triple-buffering** dans `GpuPipeline` — 3 pinned buffers au lieu de 2, overlap complet
3. **Chunk size auto-tuning** — detecter BAR size, ajuster chunks (ReBAR > 256 MB → chunks 64 MB)
4. **`bench_rust_transfer` PyO3 function** — benchmark integre appele depuis Python
5. **Integrer proprement dans transfer_manager.py** — Strategy 1.5 utilise `GpuPipeline.transfer()` pour tout > 1 MB

### Risques

- **IOMMU VM** : cuMemcpyPeerAsync retourne CUDA_ERROR_PEER_ACCESS_UNSUPPORTED → fallback staged
- **Proxmox VFIO overhead** : ~10-15% sur PCIe DMA, non eliminable cote soft
- Le gain reel vs PipelinedTransport Python est peut-etre < 5% (les DMA engines sont le goulot, pas le CPU)

---

## Chantier 2 — NVFP4 Kernel Natif (Blackwell)

### Etat actuel du code

`core/nvfp4_direct.py` — **343 LOC, production**

Ce qui **existe et marche** :
- `DirectFP4Linear` : remplace les NVFP4Tensor par plain buffers + `torch._scaled_mm` direct
- Activation quantization via Triton kernel (`triton_quantize_nvfp4`) ou fallback Python
- GEMV path (M=1) via `core/triton_gemv_nvfp4.py` (LUT kernel)
- `replace_with_direct_fp4()` : remplacement automatique post-quantization
- Resultat : +7% vs torchao (12.0 vs 11.2 tok/s), 0 VRAM extra

Ce qui **manque** :
- Le bottleneck est la **quantization dynamique des activations** (amax + scale + pack a chaque forward)
- `torch._scaled_mm` est un appel cuBLAS FP4 correct, mais la preparation des inputs est couteuse
- Pas de fusion quantize-GEMM : on quantize les activations, PUIS on appelle `_scaled_mm`
- Le GEMV Triton (M=1, decode) n'a pas ete benchmark isolement

### Objectif

| Metrique | Baseline | Target | Comment |
|---|---|---|---|
| NVFP4 decode (M=1, 7B) | 12.0 tok/s | 15-18 tok/s | Fused quantize+GEMM kernel |
| NVFP4 prefill (M>1, 7B) | ~80% of BF16 | ~90% of BF16 | Reduce activation quant overhead |
| VRAM | 5.46 GB | 5.46 GB | Pas de regression |

### Plan d'implementation

1. **Benchmark le GEMV Triton isolement** — mesurer la proportion du temps dans act_quantize vs _scaled_mm
2. **Fused activation quantize** — kernel Triton qui compute amax + scale + pack FP4 en un seul pass (au lieu de 3+ ops PyTorch)
3. **Persistent activation scale cache** — pour les tokens decode (M=1), la scale change peu entre steps
4. **Activer torch.compile sur DirectFP4Linear.forward()** — deja compatible, mais pas teste en prod

### Risques

- torchao evolue vite (0.16 → 0.17+) : nos kernels custom peuvent devenir obsoletes si torchao rattrape
- Le vrai bottleneck est peut-etre cuBLAS FP4 lui-meme (torchao prototype, pas le kernel final NVIDIA)
- Blackwell FP4 Tensor Cores ne sont peut-etre pas encore a plein regime (driver/firmware immaturs)

---

## Chantier 3 — XDP Network-to-VRAM Bypass

### Etat actuel du code

`csrc/aitp_xdp_bypass.c` — **145 LOC, Mode 0 fonctionnel**

Ce qui **existe et marche** :
- Programme eBPF/XDP valide, compile avec clang -target bpf
- Parse ETH → IPv6 → UDP → magic "VT" (port 9109)
- Mode 0 : `bpf_redirect_map(&xsks_map)` → AF_XDP userspace socket
- Stats BPF maps (bytes, packets, drops, redirects)
- Config map runtime pour switch entre modes

Ce qui **manque** :
- **Mode 1 (GPU DMA)** : le kernel drop les paquets au lieu de les ecrire en VRAM (commentaire "future GPUDirect path")
- **`core/network/aitp_receiver.py`** : reference dans le header mais le fichier **n'existe pas**
- Pas de loader Python (libbpf/bcc) pour attacher le programme XDP
- IPv4 non supporte (seulement IPv6)
- Pas de reassemblage de tenseurs multi-paquets

`core/network/network_transport.py` — **620 LOC, RDMA production**
- ibverbs QP state machine complet
- GPUDirect RDMA via nvidia_peermem
- Zero-copy TCP fallback
- Chunked transfers

### Objectif

| Metrique | Baseline | Target | Comment |
|---|---|---|---|
| Latence tensor 1 KB reseau | ~50 μs (TCP userspace) | ~10 μs | AF_XDP zero-copy + cuMemcpyHtoD |
| Throughput tensor reseau | ~5 GB/s (TCP) | ~10 GB/s | AF_XDP + pinned DMA |
| CPU utilisation pendant transfer | ~30% | < 5% | Kernel bypass |

### Plan d'implementation

1. **Creer `core/network/aitp_receiver.py`** — loader AF_XDP (via bcc ou pyroute2), UMEM ring, cuMemcpyHtoD vers VRAM
2. **Ajouter IPv4** dans `parse_aitp()` — trivial (check `ETH_P_IP` + ipv4hdr)
3. **Reassemblage multi-paquets** — header AITP avec tensor_id + offset + total_size
4. **Integration dans `transport_factory.py`** — Locality::SAME_RACK et REMOTE utilisent le path XDP quand disponible

### Risques

- Necessite root/CAP_NET_ADMIN pour charger le programme XDP
- AF_XDP necessite un NIC avec support XDP (la plupart des NICs modernes)
- Mode 1 (GPU DMA direct depuis kernel) est **tres experimental** — nvidia_peermem ne fonctionne pas avec XDP facilement
- En VM Proxmox : XDP ne s'attache pas aux vNICs facilement (virtio-net supporte xdpgeneric seulement, pas xdpdrv)

---

## Chantier 4 — WebGPU Worker Reel

### Etat actuel du code

`core/backends_webgpu.py` — **339 LOC, POC/template**

Ce qui **existe** (mais ne fait rien d'utile) :
- WebSocketServer asyncio pour recevoir des commandes
- Quantization 8-bit reelle (weights → int8)
- Structure de nodes (topology_data)
- "Speculative Network Decoding" = batching optimiste, PAS du vrai speculative decode

Ce qui **manque completement** :
- Aucun shader WGSL (pas de compute sur le navigateur)
- Aucun client web (pas de HTML/JS/WASM)
- Les nodes ne sont jamais peuplees (topology vide)
- Pas de WebRTC DataChannel (seulement WebSocket)

### Objectif

| Metrique | Target | Comment |
|---|---|---|
| Latence round-trip WebSocket | < 50 ms LAN | Pour layers de < 50 MB |
| GEMM sur navigateur (M=1, K=4096, N=4096) | > 500 GFLOPS | Chrome/Firefox WebGPU |
| Fallback local si browser lent | < 200 ms timeout | Pipeline pas bloque |

### Plan d'implementation

1. **Shader WGSL matmul** — `dashboard/worker/matmul.wgsl`, tiled 16x16, workgroup shared memory
2. **Worker JS** — `dashboard/worker/worker.js`, initialise WebGPU, charge les shaders, ecoute WebSocket
3. **Page HTML** — `dashboard/worker/index.html`, lance le worker, UI status
4. **Remplacer backends_webgpu.py** — WebSocket server qui envoie poids + activations, recoit output
5. **Timeout + fallback** — si le browser met > 200 ms, recalcul local

### Risques

- WebGPU n'est pas disponible sur tous les navigateurs (Safari partial, mobile limite)
- La bande passante WebSocket est ~100x plus lente que PCIe — seuls les petits modeles/layers sont viables
- La precision FP16 en WebGPU est legerement differente de CUDA (rounding modes)
- Securite : les poids du modele transitent en clair vers le navigateur client

---

## Priorites et dependances

```
Chantier 1 (Rust Transfer) ──────► Impact immediat sur 2-GPU (6 tok/s → 7+ tok/s sur 14B)
     │
     └─► Chantier 2 (NVFP4)  ───► Impact sur single-GPU Blackwell (12 → 15+ tok/s)
              │
              └─► Independant des chantiers 3 et 4

Chantier 3 (XDP) ───────────────► Impact uniquement en multi-noeud (reseau)
     │                              Pre-requis : Linux bare metal, NIC XDP
     └─► Depend de network_transport.py (RDMA) deja fonctionnel

Chantier 4 (WebGPU) ────────────► Impact uniquement swarm distribue navigateurs
                                    Aucune dependance technique sur 1/2/3
```

**Ordre d'execution : 1 → 2 → 3 → 4**

Le chantier 1 est le seul qui ameliore directement les benchmarks multi-GPU sur la machine actuelle.
Le chantier 2 ameliore le tok/s single-GPU sur Blackwell.
Les chantiers 3 et 4 sont pour du scale-out reseau/distribue (pas mesurable sur la machine actuelle).
