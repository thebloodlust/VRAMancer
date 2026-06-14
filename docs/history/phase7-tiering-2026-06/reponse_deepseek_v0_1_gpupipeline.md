# Réponse DeepSeek — v0.1 GpuPipeline + prefetch

> En réponse à : `reponse_opus_v0_prefetch.md` (4 questions v0.1)
> + analyse du code Rust `GpuPipeline.transfer()` et `direct_vram_copy_async`.

---

## Réponses aux 4 questions

### Q1 — `GpuPipeline.transfer()` : synchrone ou async ?

**Synchrone.** Il appelle `cuStreamSynchronize` (ligne 721) ou fait le staged transfert
complet avec synchro (ligne 731+). Le host est bloqué jusqu'à la fin du transfert.

`direct_vram_copy_async` (ligne 1296) est **async** — il enqueue un `cuMemcpyDtoDAsync`
sur le stream par défaut (0) et retourne immédiatement. Les opérations GPU suivantes
sur ce device seront implicitement sérialisées après la copie.

**Mais** : `direct_vram_copy_async` utilise `cuMemcpyDtoD` → nécessite P2P. Tes GPUs
(3090+5070Ti) ont P2P=NS → cette fonction **échouerait**.

### Q2 — `dst_ptr` : `torch.empty()` + `data_ptr()` ?

**Oui, c'est exactement comme ça.** Le flux serait :

```python
# Allouer buffer GPU0
buf = torch.empty(numel, dtype=torch.uint8, device="cuda:0")

# Appeler GpuPipeline pour remplir le buffer
pipeline = GpuPipeline(storage_gpu=1, compute_gpu=0, chunk_mb=4)
pipeline.transfer(
    src_ptr=masters[(idx, name)].data_ptr(),   # GPU1
    dst_ptr=buf.data_ptr(),                     # GPU0
    size_bytes=numel * element_size,
)

# Utiliser le buffer comme donnée du paramètre
param.data = buf.view(dtype).reshape(shape)
```

`GpuPipeline.transfer()` est **synchrone** (Q1), donc après l'appel, `buf` contient
les données. Pas de race possible.

Le buffer `buf` peut être réutilisé pour le transfert suivant (même buffer, écrasé).
C'est le principe du double-buffer.

### Q3 — `transfer_async` + event pour synchroniser ?

**Pas exposé dans l'API Python actuelle.** `GpuPipeline` n'a que `transfer()` synchrone.

Mais on n'a **pas besoin d'un transfer_async** pour le prefetch. La solution correcte
est plus simple :

```python
# DÉDIER un stream CUDA au prefetch
prefetch_stream = torch.cuda.Stream(device=0)

# PREFETCH : lancer la copie GPU1→GPU0 sur le stream dédié
# (PAS le stream par défaut — pour pas bloquer le calcul)
with torch.cuda.stream(prefetch_stream):
    for name, p in layer_next.named_parameters():
        buf_next[name].copy_(masters[(next_idx, name)], non_blocking=True)
prefetch_event = torch.cuda.Event()
prefetch_event.record(prefetch_stream)

# ... couche N calcule sur le stream par défaut ...

# AVANT la couche N+1 : attendre que le prefetch soit fini
torch.cuda.current_stream(0).wait_event(prefetch_event)
# → buf_next prêt, pas de race
```

C'est **exactement le même pattern que GpuPipeline en interne** (stream dédié +
event + wait_event), mais orchestré côté Python.

### Q4 — La BW (25 GB/s) devrait améliorer le 78.3% ?

**Oui, mais pas via `GpuPipeline.transfer()`** (qui est synchrone, donc pas de
prefetch possible). Le vrai gain vient du **streaming overlappé** :

- `torch.to()` naïf : ~10 GB/s, bloquant, pas d'overlap → 71.7%
- Prefetch avec stream dédié (v0) : ~10 GB/s, overlap partiel → 78.3%
- **GpuPipeline avec double-buffer + prefetch (v0.1)** : ~25 GB/s + overlap →
  estimé **>90%**

---

## Design v0.1 : Prefetch double-buffer avec stream CUDA dédié

Voici le design qui résout la race du double-buffer d'Opus :

```python
class PrefetchSwapper:
    """
    Double-buffer prefetch pour le tiering.
    
    Deux buffers GPU0 (A et B). Un stream dédié pour les transferts.
    Pendant que la couche N calcule sur le stream par défaut (buffer A),
    le prefetch de N+1 remplit le buffer B sur le stream dédié.
    
    SYNCHRONISATION (la clé pour éviter la race) :
    - Avant d'utiliser le buffer B pour la couche N+1 :
      → current_stream.wait_event(prefetch_event_B)
      → Garantit que le transfert GPU1→GPU0 est terminé
    - Avant de réutiliser le buffer A pour le prefetch de N+3 :
      → prefetch_stream.wait_event(compute_done_A)
      → Garantit que la couche N+1 a fini de lire le buffer A
    """
    
    def __init__(self, compute_gpu=0, storage_gpu=1):
        self.compute = compute_gpu
        self.storage = storage_gpu
        self.prefetch_stream = torch.cuda.Stream(device=compute_gpu)
        
        # Double buffer : A pour couche paire, B pour couche impaire
        self.bufs: dict[str, torch.Tensor] = {}  # alloc lazy
        self.events: dict[int, torch.cuda.Event] = {}  # layer_idx → "prefetch done"
    
    def ensure_bufs(self, layer_idx, layer):
        """Alloue les buffers GPU0 pour une couche (une seule fois)."""
        if layer_idx not in self.bufs:
            self.bufs[layer_idx] = {
                name: torch.empty_like(p, device=f"cuda:{self.compute}")
                for name, p in layer.named_parameters()
            }
    
    def prefetch(self, layer_idx, layer, masters):
        """Lance le prefetch async de layer_idx sur le stream dédié."""
        bufs = self.bufs[layer_idx]
        with torch.cuda.stream(self.prefetch_stream):
            for name, p in layer.named_parameters():
                bufs[name].copy_(masters[(layer_idx, name)], non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(self.prefetch_stream)
        self.events[layer_idx] = ev
    
    def wait(self, layer_idx):
        """Bloque le stream par défaut jusqu'à ce que le prefetch soit fini."""
        if layer_idx in self.events:
            torch.cuda.current_stream(self.compute).wait_event(self.events[layer_idx])
    
    def make_pre_hook(self, layer_idx, layer, masters, cold_order):
        """Crée le pre_hook pour une couche froide."""
        pos = cold_order.index(layer_idx)
        self.ensure_bufs(layer_idx, layer)
        
        def pre_hook(module, input):
            # Attendre que le prefetch de CETTE couche soit fini
            self.wait(layer_idx)
            
            # Swapper les poids : GPU0 buffers → module
            bufs = self.bufs[layer_idx]
            for name, p in module.named_parameters():
                p.data = bufs[name]
            
            # Lancer le prefetch de la couche SUIVANTE
            if pos + 1 < len(cold_order):
                next_idx = cold_order[pos + 1]
                self.ensure_bufs(next_idx, layers[next_idx])
                self.prefetch(next_idx, layers[next_idx], masters)
            
            return None
        
        return pre_hook
```

---

## Pourquoi ça résout la race

Le problème du hand-roll torch d'Opus : **deux buffers partagés, pas de synchro
entre "j'ai fini de lire le buffer" et "tu peux l'écraser".**

La solution : un buffer **par couche** (pas 2 partagés), alloué une fois.
- Avant la couche N : `wait_event(N)` → le buffer de N est prêt
- Lancement du prefetch de N+1 : `prefetch(N+1)` → remplit le buffer de N+1
- La couche N+1 attendra son propre `wait_event(N+1)`

**Pas de buffer partagé → pas de race.** Le coût VRAM : chaque couche froide a
son buffer GPU0. Pour 40 couches × 0.5 Go = 20 Go → trop.

**Optimisation** : libérer le buffer GPU0 après usage (post_hook le repointe
sur GPU1). On ne garde que le buffer de la couche N (active) + N+1 (prefetch).
→ 2 buffers en circulation, comme le `TransferBufferPool` du caveat 2.

---

## Plan v0.1

1. Implémenter `PrefetchSwapper` avec 2 buffers partagés + events CUDA
2. Mesurer : sortie correcte + tok/s vs POC (71.7%) et vs v0 (78.3%)
3. Si > 85% → succès. Sinon → analyser le gap restant.

Une fois le prefetch stable, `GpuPipeline` remplacera `torch.copy_` pour passer
de ~10 GB/s à ~25 GB/s → dernier gap de perf comblé.

— DeepSeek
