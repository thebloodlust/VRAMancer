# Architecture — Memory Tiering Multi-Niveau VRAMancer

> Proposé par DeepSeek · 2026-06-14
> Basé sur l'audit de `experimental/hierarchical_memory.py` (44 KB, 6 niveaux)
> et les mesures réelles : `GpuPipeline` 25.3 GB/s 3090↔5070 Ti.

---

## 1. Le principe : GPU0 compute, tout le reste = stockage

```
┌────────────────────────────────────────────────────────────────────┐
│ GPU0 (3090, 24GB) — SEUL compute                                  │
│                                                                    │
│  WORKING SET (~10 GB)              SWAP BUFFER (~2 GB)             │
│  ┌─────────────────────────┐      ┌──────────────────────────┐    │
│  │ Couche active + KV page │ ←──→ │ Double-buffer pour les    │    │
│  │ courante                │      │ transfers async avec GPU1 │    │
│  └─────────────────────────┘      └──────────────────────────┘    │
│                                                                    │
│  Le GPU0 ne "sait" pas que le reste du modèle est ailleurs.        │
│  Le MemoryManager intercepte les page-faults et swap à 25 GB/s.   │
└────────────────────────────────────────────────────────────────────┘
         │ PCIe 4.0 x16 — 25.3 GB/s (mesuré !)
         │
┌────────────────────────────────────────────────────────────────────┐
│ GPU1 (5070 Ti, 16GB) — L2 : Banques mémoire                       │
│                                                                    │
│  BANQUE A (priorité 0, ~10 GB)   BANQUE B (priorité 1, ~4 GB)    │
│  ┌──────────────────────────┐   ┌──────────────────────────┐     │
│  │ Couches froides (poids)   │   │ Pages KV overflow        │     │
│  │ Utilisées 1-2× par step   │   │ Contexte long, tokens    │     │
│  └──────────────────────────┘   │ 0-2000                    │     │
│                                  └──────────────────────────┘     │
│  BANQUE C (priorité 2, ~2 GB)                                    │
│  ┌──────────────────────────┐                                     │
│  │ Parité XOR / redondance   │  ← régénérable, première à jeter   │
│  └──────────────────────────┘                                     │
└────────────────────────────────────────────────────────────────────┘
         │ PCIe bus local (même desktop)
         │
┌────────────────────────────────────────────────────────────────────┐
│ CPU RAM — L3 : Pinned + Pageable DRAM                              │
│                                                                    │
│  PINNED (~4 GB)                    PAGEABLE (~restant)             │
│  ┌──────────────────────────┐     ┌──────────────────────────┐    │
│  │ Buffers de transfert     │     │ Couches overflow du GPU1  │    │
│  │ GPU↔CPU DMA (réutilisés) │     │ (quand L2 est plein)      │    │
│  └──────────────────────────┘     └──────────────────────────┘    │
│                                                                    │
│  Auto-balancer : si RAM > 85% → évacue vers NVMe                  │
└────────────────────────────────────────────────────────────────────┘
         │ PCIe/NVMe bus local
         │
┌────────────────────────────────────────────────────────────────────┐
│ NVMe SSD — L5 : Stockage froid                                     │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Poids de modèles complets, checkpoints, caches de prefixe    │ │
│  │ Chargé à ~7 GB/s via io_uring ou mmap (déjà implémenté)     │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
         │ Réseau (Ethernet / WiFi)
         │
┌────────────────────────────────────────────────────────────────────┐
│ GPU distant (Laptop 4060, Mac M4) — L4 : RAMDisk réseau            │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Préchargement asynchrone de couches via VTP/GpuNetBridge     │ │
│  │ Utile pour : modèles très larges (70B+), failover, batch     │ │
│  │ Lent mais capacité quasi-illimitée                           │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
```

---

## 2. Ce qui existe déjà (ne pas réinventer)

| Composant | Fichier | Ce qu'il fait | Ce qui manque |
|---|---|---|---|
| `HierarchicalMemoryManager` | `experimental/hierarchical_memory.py` | 6 niveaux, scoring LFU, migration, balancer auto | Intégration `GpuPipeline` |
| `GpuPipeline` | `rust_core/src/lib.rs` | Transfert GPU↔GPU 25 GB/s, triple-buffering | Appelé depuis le tiering |
| `async_gpu_transfer` | `rust_core/src/lib.rs` | DMA overlappé asynchrone | Appelé depuis le prefetch |
| `FastNVMeTransfer` | `experimental/hierarchical_memory.py` | io_uring, DirectStorage, mmap par plateforme | Déjà complet |
| `PagedKVCache` | `core/paged_attention.py` | Pages KV avec IDs, swap par page | Mapping page→banque |
| `VRAMLendingPool` | `experimental/vram_lending.py` | Leases GPU, scoring, reclaim auto | Intégré dans L1↔L2 |
| `GpuNetBridge` | `rust_core/src/lib.rs` | GPU→TCP→GPU, buffers pinnés, timeouts | Intégré dans L3↔L4 |

**Le constat : 80% du code existe. Le travail restant = intégration + politique de tiering.**

---

## 3. La politique de tiering (le "cerveau")

```python
@dataclass
class TieringPolicy:
    """Règles de décision pour la montée/descente de niveau."""
    
    # Seuils de promotion (montée vers GPU0)
    promote_to_l1_if_accessed_in_last_s: float = 5.0    # 5 secondes
    promote_to_l1_if_frequency_above: float = 0.5        # > 0.5 accès/seconde
    
    # Seuils de rétrogradation (descente depuis GPU0)
    demote_from_l1_if_idle_s: float = 30.0               # 30 secondes sans accès
    
    # Seuils de pression mémoire
    l1_high_watermark: float = 0.85   # Si GPU0 > 85% plein → évacuer
    l2_high_watermark: float = 0.90   # Si GPU1 > 90% plein → déplacer vers RAM
    ram_high_watermark: float = 0.85  # Si RAM > 85% → évacuer vers NVMe
    
    # Stratégie d'éviction
    eviction_strategy: str = "lfu_decay"  # LFU avec decay exponentiel

class TieringEngine:
    """
    À chaque étape d'inférence :
    1. Le modèle demande un bloc → engine.check(block_id)
    2. Si le bloc est sur GPU0 → hit, rien à faire
    3. Si le bloc est ailleurs → engine.promote(block_id, target="L1")
       - L2→L1 : GpuPipeline 25 GB/s
       - L3→L1 : pinned→GPU DMA
       - L5→L1 : NVMe→RAM→GPU (2 sauts)
    4. Si GPU0 est plein → engine.evict() libère le bloc le plus froid
    5. Le bloc évincé descend d'un niveau (L1→L2, L1→L3, etc.)
    """
    
    def touch(self, block_id: str):
        """Enregistre un accès à un bloc. Met à jour le score LFU."""
        now = time.time()
        score = self._hot_scores.get(block_id, 0.0)
        elapsed = now - self._last_touch.get(block_id, now)
        # Decay exponentiel : demi-vie = 60 secondes
        decay = 0.5 ** (elapsed / self._decay_half_life)
        self._hot_scores[block_id] = score * decay + 1.0
        self._last_touch[block_id] = now
    
    def should_promote(self, block_id: str) -> bool:
        """Ce bloc mérite-t-il d'être sur GPU0 ?"""
        score = self._hot_scores.get(block_id, 0)
        last = self._last_touch.get(block_id, 0)
        now = time.time()
        return (now - last < self.policy.promote_to_l1_if_accessed_in_last_s 
                or score > self.policy.promote_to_l1_if_frequency_above)
    
    def find_eviction_candidate(self) -> str:
        """Trouve le bloc le plus froid sur GPU0."""
        l1_blocks = [bid for bid, info in self._registry.items() 
                     if info['tier'] == 'L1']
        return min(l1_blocks, key=lambda bid: self._hot_scores.get(bid, 0))
```

---

## 4. La boucle d'inférence avec tiering transparent

```python
def infer_with_tiering(model_layers, tiering: TieringEngine, input_ids):
    """
    GPU0 exécute TOUT le calcul.
    Le tiering engine swap les poids automatiquement.
    Le modèle ne sait pas que ses couches sont réparties sur 5 niveaux.
    """
    hidden = embedding(input_ids)  # GPU0
    
    for i, layer in enumerate(model_layers):
        layer_id = f"layer_{i}"
        
        # ── Étape 1 : Vérifier si la couche est sur GPU0 ──────
        tier = tiering.get_tier(layer_id)
        
        if tier != "L1":
            # ── Étape 2 : Pas sur GPU0 → la charger ──────────
            if tiering.l1_is_full():
                victim = tiering.find_eviction_candidate()
                tiering.demote(victim)  # descend d'un niveau
            
            tiering.promote(layer_id, target="L1")
        
        # ── Étape 3 : Précharger la couche suivante (async) ──
        if i + 1 < len(model_layers):
            next_id = f"layer_{i+1}"
            if tiering.get_tier(next_id) != "L1":
                tiering.promote_async(next_id, target="L1")
                # → GpuPipeline lance le transfert sur un stream dédié
                # → GPU0 continue le calcul de la couche i en parallèle
        
        # ── Étape 4 : Calcul ──────────────────────────────────
        hidden = layer(hidden)  # GPU0
        
        # ── Étape 5 : Marquer l'accès ─────────────────────────
        tiering.touch(layer_id)
    
    logits = lm_head(hidden)  # GPU0
    return logits
```

---

## 5. Ce qui manque pour coder ça (~300 lignes)

### Fichier à créer : `core/tiering_engine.py`

```python
# structure proposée :

class MemoryBank:
    """Une région allouée sur un GPU/tier spécifique."""
    name: str
    tier: Tier
    capacity_bytes: int
    used_bytes: int
    priority: int  # 0 = garder, 100 = jeter
    allocations: dict[str, Allocation]  # block_id -> (ptr, size, shape, dtype)

class TieringEngine:
    """Cerveau du tiering. Utilise HierarchicalMemoryManager pour le transport."""
    
    def __init__(self, compute_gpu=0, storage_gpus=[1], nvme_dir=".tiering_cache"):
        self.compute_gpu = compute_gpu
        
        # Banques sur GPU1
        self.banks = [
            MemoryBank("cold_layers", "L2", capacity=10*1024**3, priority=0),
            MemoryBank("kv_overflow", "L2", capacity=4*1024**3, priority=50),
            MemoryBank("parity", "L2", capacity=2*1024**3, priority=100),
        ]
        
        # Transport : GpuPipeline (25 GB/s) pour L1↔L2
        self._gpu_pipeline = None  # lazy init
        
        # Transport : HierarchicalMemoryManager pour L3/L5
        self._hmm = HierarchicalMemoryManager(nvme_dir=nvme_dir)
        
        # Politique
        self._hot_scores: dict[str, float] = {}
        self._policy = TieringPolicy()
    
    def promote(self, block_id: str, target: Tier = "L1"):
        """Monte un bloc vers le tier cible."""
        current = self._hmm.get_tier(block_id)
        tensor = self._tensor_registry[block_id]
        
        if current == "L2" and target == "L1":
            # Fast path : GpuPipeline 25 GB/s
            self._gpu_pipeline.transfer(src_ptr, dst_ptr, size)
        else:
            # Slow path : HierarchicalMemoryManager
            self._hmm.migrate(block, target, tensor)
    
    def promote_async(self, block_id: str, target: Tier = "L1"):
        """Lance une promotion asynchrone (GPU0 n'attend pas)."""
        # Utilise async_gpu_transfer → DMA sur stream dédié
        pass
    
    def demote(self, block_id: str):
        """Descend un bloc d'un niveau (L1→L2, L2→L3, etc.)."""
        pass

def get_tiering_engine() -> TieringEngine:
    """Singleton."""
    pass
```

---

## 6. Plan d'implémentation

| Étape | Fichier | Temps | Dépend de |
|---|---|---|---|
| 1. `MemoryBank` + `TieringEngine` squelette | `core/tiering_engine.py` | Session 1 | Rien |
| 2. Intégration `GpuPipeline` dans `promote()` L1↔L2 | `core/tiering_engine.py` | Session 1 | Étape 1 |
| 3. `promote_async()` avec `async_gpu_transfer` | `core/tiering_engine.py` | Session 2 | Étape 2 |
| 4. Scoring LFU avec decay + `should_promote()` | `core/tiering_engine.py` | Session 2 | Étape 1 |
| 5. `evict()` avec politique de banques | `core/tiering_engine.py` | Session 2 | Étape 1 |
| 6. Intégration dans `infer()` (le vrai forward) | `core/backends.py` | Session 3 | Étapes 2-5 |
| 7. Benchmark : 14B sur GPU0 seul vs GPU0+GPU1 RAMDisk | `benchmarks/` | Session 3 | Étape 6 |
| 8. Ajouter les niveaux L3 (RAM), L5 (NVMe) | `core/tiering_engine.py` | Session 4 | Étape 6 |
| 9. Ajouter L4 (réseau : laptop, Mac) via `GpuNetBridge` | `core/tiering_engine.py` | Session 5 | Étape 8 |

---

## 7. Le résultat final : ce que ça permettra

Avec les 3 GPUs + RAM + NVMe :

| Configuration | VRAM "effective" | Modèle max |
|---|---|---|
| GPU0 seul (24 GB) | 24 GB | Qwen2.5-14B BF16 (~28 GB → tient pas) |
| GPU0 + GPU1 RAMDisk | 40 GB | Qwen2.5-14B BF16 (OK, 28 GB) |
| GPU0 + GPU1 + RAM + NVMe | 24 + 16 + RAM + NVMe | **Qwen2.5-32B BF16 (~64 GB → tient !)** |
| + Laptop + Mac (réseau) | ~illimité | **70B+ modèles** |

Et tout ça avec un **seul GPU qui calcule**. Les autres sont des disques. C'est ça la vision originale de VRAMancer — et on a déjà 80% du code.

---

## 8. Décision : est-ce qu'on le fait ?

**Arguments POUR :**
- 80% du code existe déjà (`HierarchicalMemoryManager`, `GpuPipeline`, `FastNVMeTransfer`)
- Le `GpuPipeline` a été mesuré à 25.3 GB/s — la fondation est solide
- Avec la 7900 XT en plus, c'est 24+16+20 = 60 GB de "VRAM effective"
- Personne d'autre ne fait ça. La preuve de concept serait unique.

**Arguments CONTRE :**
- Complexité d'intégration dans `infer()` — toucher au forward est risqué
- Le bug A1 (Path 2) doit être corrigé d'abord
- La politique de tiering est difficile à tuner sans benchmark réel

**Recommandation : GO, mais en deux phases.**
1. **Phase A** : `TieringEngine` + L1↔L2 via `GpuPipeline` (sessions 1-3). Valider avec un benchmark 14B.
2. **Phase B** : Ajouter L3 (RAM) + L5 (NVMe) (sessions 4-5). Valider avec un 32B.

— DeepSeek
