# Conseils de DeepSeek — pour Opus & l'architecte

> Consolidation de tous mes audits, réponses aux questions, et propositions.
> Date : 2026-06-14. Destiné à Opus (exécution) et à l'architecte (décisions).

---

## 1. Correction critique — GPU compute prioritaire = 5070 Ti (NVFP4)

**Le 3090 est le RAMDisk. La 5070 Ti est le compute.**

```
GPU0 = RTX 5070 Ti (16 GB, Blackwell SM 12.0) → COMPUTE
  • NVFP4 natif : un 14B BF16 (~28 Go) → ~7 Go en FP4
  • Tient LARGEMENT sur 16 Go
  • FP4 GEMM via cublas _scaled_mm : ~2x plus rapide que BF16

GPU1 = RTX 3090 (24 GB, Ampere SM 8.6) → RAMDISK
  • 24 GB de stockage passif
  • Pas de NVFP4 → inutile en compute
  • Transfert GPU1→GPU0 à 25 GB/s via GpuPipeline
  • Banques : couches froides, KV overflow, parité
```

**Raison** : La 3090 n'a PAS NVFP4 (SM 8.6, besoin SM 10.0+). La 5070 Ti l'a (SM 12.0). Pour l'inférence FP4, la 5070 Ti est ~2x plus rapide ET utilise ~4x moins de VRAM. Il n'y a aucun scénario où la 3090 devrait être le GPU de compute.

---

## 2. Mes 3 recommandations les plus importantes

### Reco #1 — Corriger le bug A1 Path 2 (priorité absolue)

**Symptôme** : Sortie dégénérée "The following is the following is the following is..."
**Cause** : `attention_mask=None` au décode → `scaled_dot_product_attention` ignore le KV cache → répétition.
**Fix** : 1 ligne. `_causal_mask = pt.zeros(...)` quand `seq_len==1` et `past_len>0`.
**Détail** : `reponse_deepseek_A1_bug.md` (déjà transmis)

### Reco #2 — Implémenter le tiering L1↔L2 avec GpuPipeline

**Pourquoi** : Le `HierarchicalMemoryManager` existe déjà mais utilise `TransferManager` (Python, lent) au lieu de `GpuPipeline` (Rust, 25 GB/s mesuré). Remplacer l'appel dans `_move_inter_gpu()` par `GpuPipeline.transfer()`.

**Gain** : +143% de bande passante sur les transferts L1↔L2 (10.4 → 25.3 GB/s).

### Reco #3 — GPU1 = RAMDisk avec banques mémoire

**Concept** : GPU1 (3090, 24 GB) ne fait AUCUN forward. Il stocke :
- Banque A (10 GB, priorité 0) : couches froides du modèle
- Banque B (4 GB, priorité 50) : pages KV cache overflow
- Banque C (2 GB, priorité 100) : parité XOR régénérable

**Détail** : `proposition_tiering_multiniveau.md`

---

## 3. Corrections techniques à appliquer (suite à mes erreurs)

3 points où j'avais tort et où Opus a raison :

| Point | Ce que j'avais dit | La vérité (Opus) | Action |
|---|---|---|---|
| Q8 `PyValueError` dans contexte String | `PyValueError::new_err(...)` | Ne compile pas en contexte `Result<_, String>`. Utiliser `map_err`. | ✅ Déjà corrigé par Opus |
| Q6 AVX-512 | "Vraie sur Zen 4/5" | Binaire compilé en **SSE2**, 0 AVX (`RUSTFLAGS` vide). Question de compile-time, pas runtime. | ✅ Déjà corrigé par Opus |
| Q-A1.2 rotary ×2 | "Risque modéré" | `_POS_EMBED_PATTERNS` ne matche que GPT-2 `wpe`. Qwen2.5 → `pos_embed=None` → rotary 1×. | ✅ Vérifié par Opus, ma crainte était infondée |

---

## 4. Architecture — La vision complète du tiering

```
┌────────────────────────────────────────────────────────────┐
│ GPU0 = RTX 5070 Ti (16 GB, Blackwell, NVFP4) — COMPUTE     │
│                                                            │
│  FP4 GEMM (cublas _scaled_mm) — 2x plus rapide que BF16   │
│  Modèle 14B en FP4 = ~7 GB → tient LARGEMENT              │
│  Swap buffer 2 GB pour transfers async avec GPU1          │
└────────────────────────────────────────────────────────────┘
         │ PCIe 4.0/5.0 — 25 GB/s via GpuPipeline
         │
┌────────────────────────────────────────────────────────────┐
│ GPU1 = RTX 3090 (24 GB, Ampere) — RAMDISK                  │
│                                                            │
│  Banque A (10 GB) : couches froides, backups              │
│  Banque B (4 GB)  : KV cache overflow                     │
│  Banque C (2 GB)  : parité XOR régénérable                │
│  Libre (8 GB)     : expansion future                      │
└────────────────────────────────────────────────────────────┘
         │
┌────────────────────────────────────────────────────────────┐
│ RAM système — L3                                           │
│  Pinned (4 GB) : buffers DMA GPU↔CPU                      │
│  Pageable : overflow des banques GPU1                     │
│  Auto-balancer : si >85% → évacue vers NVMe               │
└────────────────────────────────────────────────────────────┘
         │
┌────────────────────────────────────────────────────────────┐
│ NVMe — L5                                                  │
│  Modèles complets, checkpoints, caches                    │
│  io_uring / mmap → ~7 GB/s (déjà implémenté)             │
└────────────────────────────────────────────────────────────┘
         │ Réseau
         │
┌────────────────────────────────────────────────────────────┐
│ GPU distant (Laptop 4060, Mac M4/M5) — L4                 │
│  Via GpuNetBridge / VTP                                   │
│  Préchargement asynchrone, failover                       │
└────────────────────────────────────────────────────────────┘
```

---

## 5. Mes recommandations P1-P3 re-bucketées

### À faire MAINTENANT (5070 Ti + 3090, desktop seul)

| Item | Description | Fichier à toucher |
|---|---|---|
| **Bug A1** | Fix `attention_mask=None` au décode | `core/backends.py:~1703` |
| **P1.3** | Ajouter `cuMemGetAddressRange` à `cuda_ffi` | `rust_core/src/lib.rs` |
| **P1.4** | Wrapper Python pour `rebar_mmap.c` (ReBAR déjà activé sur 3090) | `csrc/rebar_mmap.c` + wrapper |
| **P2.1** | RAII CUDA : `DevicePtr`, `PinnedBuf`, `StreamGuard` | `rust_core/src/lib.rs` |
| **P2.6** | `GpuPipeline.transfer_async` (sans synchronize) | `rust_core/src/lib.rs` |
| **P2.7** | `CU_STREAM_NON_BLOCKING` pour pas bloquer le stream PyTorch | `rust_core/src/lib.rs` |
| **P2.10** | Auto-tuning PCIe : chunk=4 MB optimal (déjà mesuré), câbler dans `GpuPipeline::new()` | `rust_core/src/lib.rs` |
| **Tiering L1↔L2** | `GpuPipeline` dans `HierarchicalMemoryManager._move_inter_gpu()` | `experimental/hierarchical_memory.py` |
| **RAMDisk** | `MemoryBank` + `TieringEngine` (fichier proposé) | `core/tiering_engine.py` (nouveau) |

### À faire avec le laptop/Mac connectés

| Item | Description |
|---|---|
| **P1.1** | Fenêtre glissante chunked TCP |
| **P2.3** | GpuNetBridge chunked (tensors > 64 MB) |
| **P2.4** | GpuNetBridge TLS |
| **P3.2** | Pipeline distribué desktop↔laptop↔Mac |
| **P3.6** | Cross-vendor NVIDIA↔Apple Silicon |
| **P3.8** | Tests réseau réels (3 nœuds) |

### Bloqué (matériel manquant)

| Item | Ce qui manque |
|---|---|
| **P3.1** | GPUDirect RDMA — besoin NIC InfiniBand/RoCE |
| **Cross-vendor PCIe** | DMA-BUF NVIDIA↔AMD — besoin GPU AMD (7900 XT en recherche) |

---

## 6. La question AMD

Avec une RX 7900 XT (20 GB, RDNA 3, ~500€) :
- Premier vrai test cross-vendor PCIe du projet
- DMA-BUF kernel via `amdgpu` ↔ `nvidia-drm`
- 60 GB VRAM totale (16 + 24 + 20)
- Le `CrossVendorBridge` passe de `experimental/` à `core/`

C'est le seul gros chaînon manquant pour la vision complète.

---

## 7. Points de vigilance

1. **Path 2 est le futur.** La décision A de l'architecte est la bonne. Sans Path 2, VRAMancer reste un wrapper accelerate. Le bug A1 est trivial à corriger.

2. **Ne pas tomber dans le piège de la performance réseau.** Le goulot actuel est le staging PCIe (2 sauts), pas le réseau. Le bypass Ethernet XDP est cool mais pas prioritaire.

3. **Mesurer avant d'optimiser.** La culture de mesure de la Phase 7 (T7.0-T7.9) est excellente. Continuer comme ça.

4. **Le `experimental/README.md` est un bon pattern.** Chaque module a son statut documenté. À généraliser.

5. **Le `cuda_ffi` est le joyau du projet.** 480 lignes de FFI CUDA propre, sans dépendance lourde. C'est la fondation sur laquelle tout le tiering et le P2P reposent.

---

## 8. Référence — Tous mes fichiers

| Fichier | Contenu |
|---|---|
| `SUPERAUDIT.md` | Audit complet de l'arbre actif (33 recommandations P0-P3) |
| `reponsedeepseek7.md` | Réponses aux 8 questions CUDA/Rust/Tokio (architecte §6) |
| `reponse_deepseek_A1_path2.md` | Comment forcer Path 2 + 5 pièges identifiés |
| `reponse_deepseek_A1_bug.md` | Diagnostic du bug de répétition (fix 1 ligne) |
| `reponse_a_opus.md` | Contestation du "aucun validable" + matériel réel |
| `proposition_tiering_multiniveau.md` | Architecture complète du memory tiering |
| `conseils_deepseek_pour_opus.md` | **Ce fichier** — synthèse pour Opus & l'architecte |

— DeepSeek
