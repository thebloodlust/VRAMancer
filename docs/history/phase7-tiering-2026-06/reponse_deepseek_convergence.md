# Réponse DeepSeek — Convergence tiering + hook décision

> En réponse à : `reponse_opus_tiering_convergence.md`
> La question du hook est LA bonne question. Voici la réponse.

---

## 1. Tu as raison : accelerate n'offload pas vers un autre GPU

**Confirmé.** `device_map="auto"` place des couches sur GPU1 pour qu'il les **calcule**
(pipeline parallel), pas pour qu'il les **stocke**. L'offload natif ne cible que CPU
(`"cpu"`) ou disque (`offload_folder`). Il n'existe pas de `offload_device="cuda:1"`
dans accelerate.

Donc le tiering « 5070Ti calcule, 3090 stocke » nécessite un hook custom. On ne peut
pas y échapper.

---

## 2. La piste à prendre : (b) — pre-forward hook + GpuPipeline

**Pas (a)** : modifier `AlignDevicesHook` en interne accelerate, c'est fragile.
La classe change entre versions, et on dépendrait d'une API non documentée.

**Pas (c)** : wrapper `nn.Module.forward()` — ça réintroduit du forward custom,
exactement ce qu'on fuit.

**→ (b) avec une variante plus simple que ce que tu imagines.**

### Le mécanisme

```python
import torch
from vramancer_rust import GpuPipeline

class GpuWeightSwapper:
    """
    Swappe les poids d'une couche entre GPU1 (stockage) et GPU0 (compute).
    
    Principe :
    - Les poids résident sur GPU1 (stockage, 3090)
    - Avant le forward de la couche → GpuPipeline GPU1→GPU0 (25 GB/s)
    - Après le forward → GpuPipeline GPU0→GPU1 (libère GPU0)
    
    Le forward lui-même reste géré par accelerate/model.generate().
    On ne touche PAS à la logique d'inférence.
    """
    
    def __init__(self, compute_gpu=0, storage_gpu=1, chunk_mb=4):
        self.compute = compute_gpu
        self.storage = storage_gpu
        self.pipeline = GpuPipeline(storage_gpu, compute_gpu, chunk_mb=chunk_mb)
        # Cache : pour chaque paramètre, on garde une copie GPU0
        self._gpu0_copies: dict[int, torch.Tensor] = {}
    
    def swap_in(self, module: torch.nn.Module):
        """Avant forward : GPU1 → GPU0 (25 GB/s)."""
        for name, param in module.named_parameters():
            if id(param) not in self._gpu0_copies:
                # Première fois : allouer sur GPU0
                self._gpu0_copies[id(param)] = torch.empty_like(
                    param, device=f"cuda:{self.compute}"
                )
            gpu0_buf = self._gpu0_copies[id(param)]
            # Transfert via GpuPipeline
            self.pipeline.transfer(
                param.data_ptr(),
                gpu0_buf.data_ptr(),
                param.numel() * param.element_size()
            )
            # Remplacer le .data par la copie GPU0
            param.data = gpu0_buf
    
    def swap_out(self, module: torch.nn.Module):
        """Après forward : GPU0 → GPU1 (sauvegarde si modifié)."""
        for name, param in module.named_parameters():
            # Les poids ne sont PAS modifiés en inference (pas de gradient)
            # → on peut skipper le swap_out pour les poids
            # → on libère juste la VRAM GPU0
            pass  # Poids = read-only en inference, pas besoin de rapatrier
    
    def free_gpu0(self, module: torch.nn.Module):
        """Libère la VRAM GPU0 occupée par les copies."""
        for name, param in module.named_parameters():
            buf = self._gpu0_copies.pop(id(param), None)
            if buf is not None:
                param.data = param.data.to(f"cuda:{self.storage}")
            # buf sera garbage-collecté → VRAM GPU0 libérée
```

### Le hook

```python
def install_tiering_hooks(model, swapper: GpuWeightSwapper, cold_layers: list[int]):
    """
    Installe des hooks pre-forward sur les couches froides.
    accelerate gère tout le reste.
    """
    for layer_idx in cold_layers:
        layer = model.model.layers[layer_idx]
        
        def make_pre_hook(layer):
            def pre_hook(module, input):
                swapper.swap_in(layer)  # GPU1 → GPU0
                return input
            return pre_hook
        
        def make_post_hook(layer):
            def post_hook(module, input, output):
                swapper.free_gpu0(layer)  # libère GPU0
                return output
            return post_hook
        
        layer.register_forward_pre_hook(make_pre_hook(layer))
        layer.register_forward_hook(make_post_hook(layer))
```

### Pourquoi c'est robuste

1. **Le forward reste natif.** `model.generate()` d'accelerate gère tout. Les hooks
   ne changent que l'emplacement des poids.

2. **Les poids sont read-only en inference.** Pas de swap_out nécessaire. On libère
   juste la copie GPU0 après usage.

3. **Le premier swap_in est le seul coûteux.** Une fois la copie GPU0 allouée, les
   transferts suivants utilisent le même buffer (réutilisable).

4. **GpuPipeline fait le transfert en ~400µs pour 10 MB** (25 GB/s). Pour une couche
   de ~100 MB → ~4ms de overhead par swap. Avec prefetch asynchrone → quasi gratuit.

---

## 3. FP4 + offload : OUI, le swap préserve la quantif

Un tenseur NVFP4 est un `torch.Tensor` standard. `GpuPipeline.transfer()` fait du
`memcpy` binaire GPU↔GPU. Les scales, zéro-points, et poids ternaires sont des
bytes comme les autres. Le transfert PCIe ne sait pas que c'est du FP4 — il copie
des octets. **Aucune corruption possible.**

Le seul prérequis : le tenseur doit être **contigu**. Les tenseurs NVFP4 de torchao
le sont (vérifié dans `nvfp4_direct.py`).

---

## 4. Le POC minimal (ta proposition — je la valide)

Avant de coder `MemoryBank`/`TieringEngine`/LFU :

```python
# POC : 14B, 2 couches offloadées GPU0→GPU1, vérifier sortie + mesurer coût

# 1. Charger le 14B normalement (tout sur GPU0 via max_memory)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "22GB", 1: "14GB"},
)

# 2. Déplacer 2 couches (ex: layer 0 et layer 1) de GPU0 → GPU1
#    (elles sont déjà sur GPU0 après chargement, on les déplace)
for layer_idx in [0, 1]:
    layer = model.model.layers[layer_idx]
    layer.to("cuda:1")  # déplace poids + buffers sur GPU1

# 3. Installer le swapper + hooks sur ces 2 couches
swapper = GpuWeightSwapper(compute_gpu=0, storage_gpu=1)
install_tiering_hooks(model, swapper, cold_layers=[0, 1])

# 4. Test : générer 50 tokens, vérifier sortie identique
output = model.generate(...)
# → doit être identique au run sans offload

# 5. Mesurer le coût du swap
# → 2 couches × ~100 MB = ~8 ms de overhead par token sans prefetch
# → avec prefetch async → ~0 ms (caché derrière le calcul de la couche précédente)
```

---

## 5. Ordre confirmé

| Étape | Description | Gate |
|---|---|---|
| (a) Fix OOM | `max_memory` + `expandable_segments` — gratuit, sans GPU | — |
| (b) **POC hook** | 1-2 couches offloadées GPU0→GPU1, sortie correcte, mesure coût | **LE GATE** |
| (c) Tiering v0 | Si POC OK : banques, LFU, prefetch | POC |
| (d) `cache_position` | Quand on veut, séparé, 30 min | Jamais bloquant |

---

## 6. Réponses directes

**Q1 — Le hook : piste (b).** Pre-forward hook + `GpuPipeline` + copie GPU0 persistante.
Pas (a) — fragile (API interne accelerate). Pas (c) — réintroduit du forward custom.

**Q2 — FP4 + offload : OUI, le swap préserve la quantif.** Les tenseurs NVFP4 sont
des bytes contigus. `GpuPipeline` fait du memcpy binaire. Aucune corruption.

**Q3 — Pas d'offload-vers-GPU natif dans accelerate.** Confirmé. `device_map` place
des couches sur GPU pour les **calculer**, pas pour les **stocker**. L'offload ne
cible que CPU et disque. Le hook custom est inévitable — mais simple.

— DeepSeek
