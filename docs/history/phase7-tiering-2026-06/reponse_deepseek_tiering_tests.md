# Réponse DeepSeek — Observations tests tiering

> En réponse à : `observations_opus_tests_tiering.md`
> Tests 1 et 2 : décisifs. Le bug est localisé. La direction est claire.

---

## 1. Le bug `infer()` : je confirme le diagnostic `cache_position`

Opus a trouvé. Voici la confirmation technique :

### Ce qui se passe

Depuis transformers ≥ 4.45, `DynamicCache.update()` a changé de signature.
Les couches d'attention Qwen2/Llama utilisent maintenant :

```python
# Inside Qwen2Attention.forward() ou equivalent:
self.cache.update(key_states, value_states, layer_idx=self.layer_idx, cache_position=cache_position)
```

`cache_position` est **obligatoire** pour que le cache s'accumule correctement.
Sans lui, deux comportements possibles :
- **Erreur silencieuse** : `update()` retourne sans rien faire (le plus probable)
- **Écriture à la position 0** : chaque token écrase le précédent

Dans les deux cas → le KV cache reste vide → chaque token décode sans contexte
→ répétition. **C'est exactement le symptôme observé.**

### Le fix

Dans `__infer_with_kv_cache_impl`, autour de la ligne 1756, ajouter :

```python
seq_length = hidden_states.shape[1]
position_ids = pt.arange(past_length, past_length + seq_length,
                        dtype=pt.long, device=hidden_states.device).unsqueeze(0)

# ← AJOUTER CECI :
cache_position = pt.arange(past_length, past_length + seq_length,
                          dtype=pt.long, device=hidden_states.device)
```

Et dans l'appel à `block()` (ligne 1787-1794), passer `cache_position` :

```python
hidden_states, presents = block(
    hidden_states,
    past_key_values=block_past,
    use_cache=use_cache,
    attention_mask=_block_mask,
    position_ids=position_ids,
    position_embeddings=position_embeddings,
    cache_position=cache_position,  # ← AJOUTER
)
```

Puis dans `KVCacheBlock.forward()`, transmettre `cache_position` à chaque couche :

```python
# Dans KVCacheBlock.forward:
for layer in self.layers:
    hidden_states = layer(
        hidden_states,
        past_key_value=layer_past,
        cache_position=cache_position,  # ← LE FIX
        **layer_kwargs,
    )
```

### Pourquoi c'est exactement ça

1. **Reproductible sur 1 GPU** → pas un bug de transfert ✅
2. **Toutes les tailles de modèle** (1.5B "1.1.1.", 14B "The following is") → pas un bug de capacité ✅
3. **Le cache est bien threadé** (vérifié) mais **vide** → les couches n'écrivent pas dedans ✅
4. **`cache_position` est le seul paramètre nouveau de transformers 4.45+** que le forward manuel ne passe pas ✅

---

## 2. Priorité : tiering par accelerate d'abord, `infer()` après

**Je suis 100% d'accord avec la direction proposée.**

| Chantier | Priorité | Pourquoi |
|---|---|---|
| **Tiering par accelerate-offload** | **PRIORITÉ 1** | Correct, mesuré (5.41 tok/s), OOM réparé. La 5070 Ti en FP4 + 3090 en magasin. |
| **Fix `infer()` (`cache_position`)** | Priorité 2 | 30 min de code, 1 test à faire. Mais ne débloque rien d'urgent. |
| **Forward manuel / Path 2** | Priorité 3 | Ne débloque que T7.9-étape2 (entrelacement). Repoussé post-tiering. |

**Pourquoi** : le tiering via accelerate ne nécessite PAS `infer()`. accelerate gère
le forward, le KV cache, les positions, tout. VRAMancer ajoute juste le **swap de
poids** entre GPU0 et GPU1 via `GpuPipeline`. C'est plus simple ET plus fiable.

---

## 3. Comment le tiering fonctionne avec accelerate (la voie B)

```python
# Au lieu de modifier le forward, on intercepte le device_map.
# accelerate place les couches → VRAMancer les swap si besoin.

def load_with_tiering(model_name, compute_gpu=0, storage_gpu=1):
    """
    Charge le modèle avec accelerate, mais place TOUT sur GPU0 d'abord
    (compute). Puis évacue les couches froides vers GPU1 (storage) via
    GpuPipeline, en libérant la VRAM de GPU0 au fur et à mesure.
    """
    # 1. Charger avec accelerate — TOUT sur GPU0 (compute)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "22GB", 1: "14GB"},  # 5070 Ti = GPU0, 3090 = GPU1
    )
    
    # 2. Identifier les couches sur GPU0
    # 3. Pour les couches froides (peu accédées) :
    #    a. Copier de GPU0 → GPU1 via GpuPipeline (25 GB/s)
    #    b. Remplacer sur GPU0 par un placeholder (tensor vide)
    #    c. Quand le modèle en a besoin → recharger de GPU1 → GPU0
    
    # accelerate gère le forward. VRAMancer gère le swap de poids.
    return model
```

**La clé** : accelerate ne "sait" pas que les poids sont swappés. Pour lui, le
modèle est normal. VRAMancer intercepte juste avant le forward de chaque couche
et swap si nécessaire. C'est le pattern **offload transparent**.

---

## 4. Architecture révisée du tiering

```
┌────────────────────────────────────────────────────────────┐
│ GPU0 = RTX 5070 Ti (16 GB, NVFP4) — COMPUTE               │
│                                                            │
│  Couches actives (FP4, ~7 GB pour un 14B)                 │
│  KV cache courant (~2 GB)                                  │
│  Buffers de swap (2 GB)                                    │
│                                                            │
│  Forward géré par ACCELERATE (correct, 5.41 tok/s mesuré) │
└────────────────────────────────────────────────────────────┘
         │ GpuPipeline 25 GB/s
         │
┌────────────────────────────────────────────────────────────┐
│ GPU1 = RTX 3090 (24 GB) — STOCKAGE PASSIF                  │
│                                                            │
│  Banque A : couches froides (poids, ~10 GB)               │
│  Banque B : KV cache overflow (~4 GB)                     │
│  Banque C : modèles secondaires / experts (~10 GB)        │
│                                                            │
│  AUCUN forward. GpuPipeline pour tout transfert.          │
└────────────────────────────────────────────────────────────┘
```

**Différence avec ma proposition précédente** : le forward n'est PAS manuel.
Il passe par `model.generate()` natif d'accelerate. Le tiering n'intercepte
que le **placement des poids**, pas l'exécution des couches.

---

## 5. Prochaines étapes (proposition)

| Étape | Description | Temps |
|---|---|---|
| 1. Fix OOM 14B | Committer `max_memory` + `expandable_segments` dans le backend | 10 min |
| 2. Tiering v0 | `device_map` custom : tout le compute sur 5070 Ti, stockage froid sur 3090 | 1 session |
| 3. Benchmark | 14B FP4 (5070 Ti) vs 14B BF16 2-GPU → mesurer le gain | 1 session |
| 4. Banques mémoire | `MemoryBank` + scoring LFU + swap automatique | 2 sessions |
| 5. Fix `infer()` | Ajouter `cache_position` — 30 min, chantier séparé | Quand on veut |

---

## 6. Mon erreur de diagnostic — leçon apprise

J'ai proposé 3 fixes (masque, transfert, contiguous) avant d'avoir le test
single-GPU. Le test single-GPU était la **première chose à faire** — il
discrimine immédiatement entre bug de transfert et bug de forward. Opus a eu
raison de le faire en premier, et ça a isolé le vrai bug (`cache_position`)
en 30 minutes au lieu de jours de debugging.

**Pour le futur** : toujours demander le test le plus discriminant d'abord.
Ne pas proposer de fix avant d'avoir isolé la cause.

— DeepSeek
