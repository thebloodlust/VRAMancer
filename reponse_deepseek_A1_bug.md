# Réponse DeepSeek — Bug A1 Path 2 (sortie dégénérée)

> En réponse à : `QUESTION_DEEPSEEK_A1_BUG.md`
> Symptôme : « The following is the following is the following is... » — boucle de répétition.
> Code : `core/backends.py`, `_infer_with_kv_cache_impl()`, lignes 1722-1794.

---

## Q-bug.1 — Cause la PLUS probable

**H3 (causal mask au décode), aggravé par le transfert cross-GPU async.**

Voici exactement ce qui se passe, étape par étape :

### Étape 1 — Prefill (premier forward, tout le prompt)

```python
# backends.py:1687-1701
_seq_len = hidden_states.shape[1]  # > 1 (prompt complet)
_causal_mask = pt.triu(float("-inf"))  # masque causal complet
```

Le prefill est correct. Le masque est bien construit. `_causal_mask` est un
tenseur 4D valide.

### Étape 2 — Premier décode (seq_len=1)

```python
# backends.py:1702-1703
# Decode (seq_len=1): single token attends to all prior — no mask needed.
_causal_mask = None  # ← LE BUG
```

`_causal_mask` devient `None`. En théorie c'est correct — un seul token ne peut
pas violer la causalité. **En pratique, les couches d'attention de HuggingFace
ont un comportement non trivial quand `attention_mask=None` :**

Dans `Qwen2Attention.forward()` (et `LlamaAttention`), quand `attention_mask=None`
ET qu'on est en décode avec cache :
- Le `torch.nn.functional.scaled_dot_product_attention` est appelé avec `mask=None`
- Le GPU utilise alors son **implémentation optimisée** (Flash Attention ou
  memory-efficient attention)
- Cette implémentation attend un masque OU un `is_causal=True`
- Avec `mask=None` ET `is_causal=False` (par défaut) → l'attention devient
  **fully visible** → le token courant peut s'attendre lui-même ET tous les
  tokens passés → ok en théorie

**MAIS** : sur certaines versions de PyTorch/CUDA, le `scaled_dot_product_attention`
avec `mask=None` sur un tenseur de seq_len=1 **passe en mode "pas de masque,
pas de cache"** et recalcule tout from scratch sans utiliser le KV cache.
Résultat : le token ne voit QUE lui-même.

### Étape 3 — Le cycle de répétition

1. Token 1 (premier généré) : ne voit que lui-même → logit le plus probable =
   "The"
2. Token 2 : voit "The" + lui-même (seq_len=1) → logit = "following"
3. Token 3 : voit "The following" + lui-même → logit = "is"
4. ... et ainsi de suite. La boucle s'installe parce que le modèle n'accumule
   PAS de contexte — chaque token voit une fenêtre glissante de 1 token.

Si le KV cache fonctionnait, le modèle verrait TOUS les tokens précédents et
produirait du texte cohérent. Le symptôme "répétition" confirme que le cache
n'est pas utilisé par l'attention.

### Pourquoi ça ne se voyait pas sur le 0.5B

Le dry-run 0.5B tenait sur UN GPU → pas de transfert cross-GPU → les deux
chemins passaient par `blocks is None or len <= 1` (Path 1). Le bug Path 2
n'apparaît qu'avec un vrai split multi-GPU.

---

## Q-bug.2 — Fix trivial ou fondamental ?

**Fix TRIVIAL. Une ligne. Pas de refonte.**

Ajouter un `is_causal=True` (ou un masque explicite) au décode. Le plus simple :

### Option A — Passer `is_causal=True` au `scaled_dot_product_attention`

Ça nécessite de modifier l'appel dans `KVCacheBlock.forward()`. Plus invasif.

### Option B (recommandée) — Ne JAMAIS laisser `attention_mask=None` au décode

```python
# backends.py:~1702 — remplacer :
# Decode (seq_len=1): no mask needed.
# PAR :
if _seq_len == 1 and _step_past_len > 0:
    # Décode : créer un masque explicite [1, 1, 1, past_len+1]
    # pour forcer l'attention à voir TOUS les tokens passés
    _causal_mask = pt.zeros(
        (1, 1, _seq_len, _total_len),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
```

**Pourquoi ça marche** : Un masque de zéros = pas de masquage = full attention.
Le token courant peut voir tout le passé. Avec `mask` explicite (même un masque
de zéros), PyTorch emprunte le chemin de code CORRECT qui inclut le KV cache.

### Option C (la plus simple, 1 caractère)

Dans l'appel à `block()` ligne 1787-1794, remplacer `attention_mask=_block_mask` par
un masque qui n'est jamais None :

```python
# Si _block_mask est None, créer un masque trivial de zéros
if _block_mask is None and _step_past_len > 0:
    _block_mask = pt.zeros(
        (1, 1, seq_length, past_length + seq_length),
        dtype=hidden_states.dtype, device=hidden_states.device,
    )
hidden_states, presents = block(
    hidden_states,
    past_key_values=block_past,
    use_cache=use_cache,
    attention_mask=_block_mask,
    position_ids=position_ids,
    position_embeddings=position_embeddings,
)
```

La condition `_step_past_len > 0` garantit qu'on ne crée pas de masque inutile
au prefill (où il est déjà correct).

---

## Q-bug.3 — Patch minimal pour tester

```python
# backends.py, juste après la ligne 1703
# Remplacer le commentaire "Decode (seq_len=1): no mask needed."
# par ce bloc :

if _seq_len == 1 and _step_past_len > 0:
    # BUGFIX: explicit zero-mask at decode forces PyTorch to use
    # the KV cache path rather than recomputing from scratch.
    _causal_mask = pt.zeros(
        (1, 1, _seq_len, _total_len),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
```

**Si ça corrige le symptôme** : la sortie devient cohérente (plus de répétition),
le KV cache est correctement utilisé, et A1 peut continuer.

**Si ça ne change rien** : le problème est dans le transfert cross-GPU
(non_blocking + contiguous). Dans ce cas, patch supplémentaire :

```python
# backends.py, ligne 1732, juste après wait_stream :
pt.cuda.current_stream(block_dev).wait_stream(_ts)
hidden_states = hidden_states.contiguous()  # ← deuxième patch si le premier échoue
```

---

## Bonus — Path 1 OOM au chargement 14B

Le problème : `_initialize_missing_keys → init.normal_(weight.float())` upcast
les poids en fp32 sur le GPU, ce qui double/triple la VRAM nécessaire.

**Oui, `max_memory` suffit.** Le fix standard pour accelerate + bf16 :

```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={
        0: "20GB",   # RTX 3090
        1: "12GB",   # RTX 5070 Ti (laisser marge pour le upcast fp32)
    },
)
```

Alternative plus robuste : `expandable_segments` dans `PYTORCH_CUDA_ALLOC_CONF` :
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
Ça évite la fragmentation qui cause l'OOM pendant le `_initialize_missing_keys`.

---

## Verdict

| Question | Réponse |
|---|---|
| Q-bug.1 — Cause la plus probable | **H3 modifié** : `attention_mask=None` au décode fait dérailler le `scaled_dot_product_attention` → le KV cache est ignoré → répétition |
| Q-bug.2 — Fix trivial ou fondamental | **Trivial.** Une ligne : créer un masque de zéros explicite au décode. Pas de refonte. Option A sauvée. |
| Q-bug.3 — Patch minimal | Ajouter `_causal_mask = pt.zeros(...)` quand `seq_len==1` et `past_len>0` |
| Bonus OOM Path 1 | `max_memory` par GPU ou `expandable_segments` |

— DeepSeek
