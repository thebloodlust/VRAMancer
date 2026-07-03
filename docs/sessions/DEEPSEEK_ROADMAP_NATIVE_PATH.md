# DeepSeek → Opus : Roadmap VRAMancer Native Path

> Stratégie : posséder l'ORCHESTRATION, garder le forward standard.
> Date : 2026-06-15.

---

## Le constat

accelerate nous bride sur 4 points :
- `device_map="auto"` force le pipeline parallèle → pas de split de phase
- Pas d'offload vers GPU → on doit coder des hooks custom
- Forward opaque → bug A1 (`cache_position`), on ne peut pas le fixer
- Pas de contrôle du scheduling → pas de data-parallel natif

Mais écrire un moteur complet (forward, 50+ architectures, Flash Attention,
quantization...) = 50+ développeurs × plusieurs années. **On ne fait pas ça.**

---

## La stratégie : "Native Path" en 5 phases

On remplace UNIQUEMENT ce qui nous bloque. On garde le forward transformers
(fiabilisé par 500 contributeurs). On injecte notre intelligence dans la
couche qui décide **quoi mettre où et quand**.

```
┌─────────────────────────────────────────────────────────┐
│           VRAMANCER NATIVE PATH                         │
│                                                         │
│  🧠 NOTRE CODE (orchestration)                          │
│  ┌───────────────────────────────────────────────────┐ │
│  │ VRAMancerDispatch      → remplace device_map      │ │
│  │ GpuPipeline (25 GB/s)  → remplace .to() torch     │ │
│  │ PagedAttention kernel  → remplace DynamicCache HF │ │
│  │ VRAMancerScheduler     → remplace vLLM scheduler  │ │
│  │ ClusterGovernance      → data-parallel multi-nœud │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  🏗️ CODE STANDARD (forward, on garde)                   │
│  ┌───────────────────────────────────────────────────┐ │
│  │ transformers → embed/blocks/head forward           │ │
│  │ torch        → ops GPU, streams, CUDA             │ │
│  │ llama.cpp    → GGUF (optionnel)                    │ │
│  └───────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1 — VRAMancerDispatch (2-3 sessions)

**Remplace `device_map="auto"` d'accelerate par NOTRE dispatch.**

```python
class VRAMancerDispatch:
    """
    Place les couches selon NOTRE plan.
    Utilise GpuPipeline pour les transferts.
    Le forward reste transformers standard.
    accelerate = fallback si on échoue.
    """
    
    def __init__(self, model, gpus: list[int], profiler: LayerProfiler):
        self.model = model
        self.gpus = gpus
        self.profiler = profiler
        self._original_device_map = getattr(model, 'hf_device_map', None)
    
    def plan(self) -> PlacementPlan:
        """Calcule le placement optimal (DP)."""
        layers = self.profiler.profile_model(self.model)
        gpu_profiles = self.profiler.profile_gpus(self.gpus)
        return self.profiler.compute_optimal_placement(layers, gpu_profiles)
    
    def dispatch(self, plan: PlacementPlan):
        """Exécute le placement. GpuPipeline pour les transferts."""
        for layer_idx, gpu_id in plan.assignments:
            layer = self.model.model.layers[layer_idx]
            current_gpu = next(layer.parameters()).device.index
            if current_gpu != gpu_id:
                self._move_layer(layer, current_gpu, gpu_id)
    
    def _move_layer(self, layer, src, dst):
        """Déplace une couche GPU→GPU via GpuPipeline (25 GB/s)."""
        pipeline = GpuPipeline(src, dst, chunk_mb=4)
        for param in layer.parameters():
            # Allouer sur GPU destination
            dst_buf = torch.empty_like(param, device=f"cuda:{dst}")
            # Transférer via GpuPipeline
            pipeline.transfer(param.data_ptr(), dst_buf.data_ptr(),
                            param.numel() * param.element_size())
            param.data = dst_buf

# Usage :
dispatch = VRAMancerDispatch(model, gpus=[0, 1], profiler=profiler)
plan = dispatch.plan()
dispatch.dispatch(plan)
# → Le forward standard tourne, mais avec NOTRE placement
output = model.generate(...)
```

**Gain** : split de phase possible (GPU0=decode, GPU1=prefill), placement
asymétrique (FP4 sur un GPU, BF16 sur l'autre), GpuPipeline partout.

**Risque** : faible. accelerate = fallback. Si le nôtre échoue, on rollback.

---

## Phase 2 — KV Cache natif (1-2 sessions)

**Remplace `DynamicCache` HF par `PagedAttention` custom.**

On intercepte les appels au cache dans le forward. On utilise notre kernel
CUDA (`paged_attention_kernel.cu`, 8.8× vs PyTorch).

```python
class VRAMancerKVCache:
    """Notre KV cache paginé. Remplace DynamicCache HF."""
    
    def __init__(self, max_pages=1024, page_size=16):
        self.pages = PagedKVCache(max_pages, page_size)
    
    def update(self, key, value, layer_idx, cache_position):
        """Écrit dans les pages. Même interface que DynamicCache."""
        page_id = self.pages.allocate_page()
        self.pages.write(page_id, key, value)
    
    def get(self, layer_idx):
        """Lit les pages pour l'attention. Kernel CUDA PagedAttention."""
        return paged_attention_kernel(self.pages, layer_idx)

# Monkey-patch transformers pour utiliser NOTRE cache :
model.model.layers[i].self_attn._kv_cache = VRAMancerKVCache()
```

**Gain** : KV cache 8.8× plus rapide, paginé (pas de fragmentation), swappable
entre GPUs.

---

## Phase 3 — VRAMancerScheduler (2 sessions)

**Remplace le scheduler vLLM. Data-parallel + split de phase natif.**

```python
class VRAMancerScheduler:
    """
    Ordonnance les requêtes sur les GPUs.
    
    Modes :
    - data-parallel : requête entière → GPU le moins chargé (0 crossing)
    - split-phase   : prefill→GPU1, décode→GPU0 (multi-requêtes)
    - batch          : accumulate les requêtes avant forward
    """
    
    def schedule(self, requests: list[InferenceRequest]) -> Schedule:
        plan = Schedule()
        for req in requests:
            gpu = self._least_loaded_gpu()
            plan.assign(req, gpu)
        
        # Si 2+ requêtes sur le même GPU → batch
        plan.optimize_batches()
        
        # Si prefill lourd en attente → mode split-phase
        if plan.has_heavy_prefill():
            plan.enable_split_phase(prefill_gpu=1, decode_gpu=0)
        
        return plan
```

---

## Phase 4 — Forward natif (3-5 sessions, OPTIONNEL)

Notre propre boucle embed→blocks→head. Nécessaire UNIQUEMENT si on veut
supprimer la dépendance transformers. Pas prioritaire.

---

## Phase 5 — Moteur standalone (10+ sessions, LONG TERME)

Plus aucune dépendance. VRAMancer = un binaire. Pas prioritaire.

---

## Comparaison : notre moteur vs l'existant

| Brique | accelerate (actuel) | VRAMancer Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|---|
| Placement | `device_map="auto"` (subi) | `VRAMancerDispatch` (contrôlé) | ✅ | ✅ |
| Transfert | `.to()` (11 GB/s) | `GpuPipeline` (25 GB/s) | ✅ | ✅ |
| KV cache | `DynamicCache` HF | `DynamicCache` HF | `PagedAttention` (8.8×) | ✅ |
| Scheduling | Aucun | Aucun | Aucun | `VRAMancerScheduler` |
| Split phase | ❌ | ✅ | ✅ | ✅ |
| Data-parallel | ❌ | ❌ | ❌ | ✅ |

---

## Questions pour Opus

1. **D'accord avec la stratégie "Native Path"** — posséder l'orchestration,
   pas réécrire le forward ?

2. **Phase 1 d'abord** (VRAMancerDispatch, 2-3 sessions) ? C'est le plus gros
   gain (split de phase, GpuPipeline partout, placement contrôlé) pour l'effort
   le plus faible.

3. **Phases 4-5 = long terme.** On ne les commence pas tant que 1-3 ne sont
   pas prouvées. D'accord ?

4. **Le fallback accelerate** : chaque phase garde l'ancien chemin en fallback.
   Si notre dispatch échoue → rollback sur accelerate. Pas de régression. OK ?

— DeepSeek
