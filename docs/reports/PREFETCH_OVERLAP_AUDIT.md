# Prefetch Overlap Audit — VRAMancer

> Date : 2026-05  
> Fichiers audités : `core/stream_manager.py`, `core/inference_pipeline.py`

## Status actuel

### `core/stream_manager.py` — `prefetch_layers()` (L175-218)

```python
def prefetch_layers(self, current_layers, lookahead=3):
    predicted = self.scheduler.predict_next_layers(current_layers, lookahead)
    for layer_idx in predicted:
        if name not in self.loaded_layers:
            size_mb = self._estimate_layer_size(layer_idx)
            # NON-BLOCKING: dispatch to ThreadPoolExecutor
            self._io_executor.submit(_do_preload, layer_idx, size_mb)
    return prefetched
```

**Comportement :**
- Le prefetch CPU-side (chargement depuis disque/NVMe/RAM) est async via `ThreadPoolExecutor`
- Méthode non-bloquante : return immédiat après `.submit()`
- Le background thread fait `preload_layer()` qui move le module vers le device cible

### `core/inference_pipeline.py` — Sync points

```bash
grep -c "synchronize" core/inference_pipeline.py → 0
grep -c "\.cpu()" core/inference_pipeline.py → 0
grep -c "\.item()" core/inference_pipeline.py → 0
```

**Aucun `torch.cuda.synchronize()` explicite dans `inference_pipeline.py`.**  
La synchronisation est déléguée au backend (HuggingFace Accelerate / vLLM).

## Bottleneck identifié ?

**Partiel.** Le prefetch CPU I/O est correct (async ThreadPool). Cependant :

1. **Pas de CUDA Stream overlap** : aucune utilisation de `torch.cuda.Stream` pour pipeliner compute(N) // transfer(N→N+1). Les activations inter-GPU sont transférées séquentiellement via `TransferManager.transfer()`.

2. **Impact réel :** Sur une configuration multi-GPU (RTX 3090 + RTX 5070 Ti) avec ReBAR actif, le bottleneck principal est la bande passante PCIe (~32 GB/s P2P). Un stream overlap pourrait masquer ~1-3 ms de latence de transfer par layer.

3. **Mitigation existante :** `TransferManager` utilise `cudaMemcpyPeerAsync` (via Rust) qui est async par nature — la synchronisation implicite arrive lors du prochain accès au tenseur.

## Recommandation

**Verdict : Prefetch CPU-async = déjà optimal pour le cas d'usage principal.**

Amélioration possible (non-urgente) :
- Créer un stream CUDA dédié au transfer dans `TransferManager`
- Lancer le transfer de l'activation block N+1 pendant que block N compute
- Sync juste avant que block N+1 commence son forward

**Effort : M | Gain attendu : ~5-10% throughput sur gros modèles (>13B) | Priorité : Basse**

Tracker dans `TECHNICAL_DEBT.md` sous ID `PREFETCH_CUDA_STREAM` si à implémenter.
