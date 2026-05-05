# CUDA Graph Multi-GPU Feasibility Audit

> Date : 2026-05  
> Fichier audité : `core/cuda_graph_decode.py` (~250 LOC)

## Status actuel

### Single-device confirmation

```python
# cuda_graph_decode.py L164-175
device = input_ids.device  # single device
static_mask = torch.ones(bs, self.max_seq_len, dtype=torch.long, device=device)
static_pos = torch.tensor([[cur_seq_len - 1]], device=device)
static_cache_pos = torch.tensor([cur_seq_len - 1], dtype=torch.long, device=device)
```

Le graph est capturé sur **un seul device** (`input_ids.device`). La classe `CUDAGraphRunner` ne prend pas de paramètre `devices` ou `num_gpus`.

### Limitation fondamentale documentée dans le code

```
docstring: "Static buffers for attention_mask, position_ids and cache_position
are managed internally and updated between replays."
```

Et dans les notes internes :
```
"NCCL / P2P ops inside the captured region are not supported."
```

## Réponses aux questions de faisabilité

### 1. Le code capture sur 1 stream / 1 device — confirmé ?
✅ **OUI.** `device = input_ids.device` — capture sur le device du premier token, single device uniquement.

### 2. Multi-device capture possible avec PyTorch 2.5+ ?
❌ **NON, fondamentalement.** `torch.cuda.CUDAGraph` capture les kernel launches sur **un seul device CUDA**. Les opérations NCCL (all-reduce, send/recv inter-GPU) et cudaMemcpyPeerAsync ne peuvent pas être capturées dans un graph — elles utilisent des streams séparés et des primitives de synchronisation qui cassent le modèle de replay du graph.

Référence : PyTorch documentation CUDA Graphs — "Ops that synchronize the CPU and GPU (e.g., to_(), to(), item()) are not supported within a graph region."

### 3. Risques si on essaie quand même ?
- **Correctness** : le graph replay utilise des adresses mémoire fixes. Si une op inter-GPU modifie des buffers capturés depuis un autre device, les adresses sont invalides → corruption silencieuse.
- **Deadlock** : NCCL all-reduce dans un graph peut deadlock si les replays ne sont pas synchronisés entre tous les ranks.
- **DynamicCache incompatibilité** : déjà gérée (`pkv_type == "DynamicCache"` → eager fallback), mais multi-device aggraverait les cas edge.

### 4. Estimation effort
**XL** — requiert :
- Séparation du prefill (hors graph) et du decode (dans graph)
- Un graph par device par batch_size
- Synchronisation manuelle entre les graphs (events inter-device)
- Tests extensifs pour la correctness numérique

### 5. Estimation gain performance
**M** — Le bénéfice principal du CUDA Graph est d'éliminer le dispatch Python overhead. Sur multi-GPU, le bottleneck est la bande passante PCIe inter-GPU, pas le dispatch Python. Gain estimé : 2-5% sur Qwen 14B 2-GPU (single-GPU show ~10-15% gain).

## Verdict

**Non-recommandé : Effort XL, Gain M, risque de régression de correctness.**

La recommandation actuelle : laisser CUDA Graph single-GPU, qui bénéficie aux cas d'usage single-GPU (benchmarks, petits modèles).

Multi-GPU ne bénéficierait pas significativement tant que le pipeline parallel est < 70% utilisation GPU.

**Alternative plus viable** : optimiser le CUDA Stream overlap dans `TransferManager` (voir `PREFETCH_OVERLAP_AUDIT.md`) — effort S/M, gain similaire sans risque.
