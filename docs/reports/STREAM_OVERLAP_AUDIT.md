# Stream Overlap Audit — VRAMancer V4 P2.1

**Date :** 2026-05-05
**Auteur :** Claude Sonnet 4.6 (exécutant V4 P2)

## Sync points actuels dans TransferManager

| Ligne | Méthode | Sync |
|-------|---------|------|
| 637 | `_transfer_p2p` | `tensor.to(..., non_blocking=True)` → sync implicite défault stream |
| 640-644 | `_transfer_p2p` | `copy_stream = torch.cuda.Stream(); copy_stream.synchronize()` |
| 704-705 | `_transfer_cpu_staged` (src) | `cpu_tensor.copy_(non_blocking=True); stream.synchronize()` |
| 707-710 | `_transfer_cpu_staged` (dst) | `src_stream = Stream; cpu_tensor.copy_(non_blocking=True); src_stream.synchronize()` |
| 713-718 | `_transfer_cpu_staged` (pin) | `dst_stream = Stream; non_blocking=True; dst_stream.synchronize()` |
| 905, 911 | `benchmark()` | `torch.cuda.synchronize()` |

## Architecture réelle des transferts (GPU 0 RTX3090 ↔ GPU 1 RTX5070Ti)

Résultat de `_can_p2p(0, 1)` : **False** (CUDA IOMMU consumer GPU).

Chemin effectivement emprunté en production :

```
Strategy 0 (cross-vendor) → skip (même vendor)
Strategy 1 (CUDA P2P)     → skip (_can_p2p = False)
Strategy 1.5 (Rust GpuPipeline) → ACTIF — cuMemcpyPeerAsync via PyO3
Strategy 1.7 (ReBAR)      → fallback si 1.5 échoue
Strategy 4 (CPU-staged)   → fallback final
```

**Conséquence :** Le stream explicite passé à `_execute_transfer` n'est **pas utilisé** par
la Strategy 1.5 (Rust layer gère ses propres streams CUDA en interne). Le flag
`VRM_TRANSFER_OVERLAP` injecte un stream dans `effective_stream`, qui est relayé à
`_execute_transfer`, mais Strategy 1.5 ignore `stream` (paramètre non utilisé dans le code Rust).

## Modification implémentée (P2.3)

- `__init__` : `_overlap_enabled` (flag), `_transfer_streams: Dict[int, Stream]`
- `_get_transfer_stream(src_gpu)` : lazy-init Stream priority=-1 par GPU source
- `send_activation` : `effective_stream = _get_transfer_stream(src)` si overlap activé

Flag : `VRM_TRANSFER_OVERLAP=1` (défaut `0`).

## Conclusion P2.5 — Décision MERGE/REVERT

**La modification est conservative et SAFE** même si le gain est nul :
- Quand `VRM_TRANSFER_OVERLAP=0` (défaut) : comportement identique à avant P2.3.
- Quand `VRM_TRANSFER_OVERLAP=1` : le stream est créé et passé, mais Strategy 1.5
  (Rust) l'ignore en pratique. Gain effectif = ~0% sur le setup RTX3090+5070Ti.
- La valeur est pour les configs qui utilisent Strategy 1 (CUDA P2P direct), Strategy 3
  (NCCL), ou Strategy 4 pure (CPU-staged `_transfer_p2p`).

**Décision : KEEP (flag off par défaut)**. Le code n'alourdit pas le hot path (vérif bool),
préserve la fonctionnalité pour configs qui ont un P2P CUDA direct, sans régression.

Bench formel non réalisé sur Qwen 14B car Strategy 1.5 Rust n'utilise pas le stream Python —
gain serait 0% sur ce setup. Documenté honnêtement ici.

## Pour travaux futurs

Si on veut de l'overlap réel avec la Strategy 1.5 Rust :
- Exposer `GpuPipeline.transfer_async(src_ptr, dst_ptr, nbytes, cuda_stream_handle)` en PyO3
- Passer le stream handle depuis Python vers la layer Rust
- Effort estimé : 1-2j (Rust + PyO3 + benchmark)
