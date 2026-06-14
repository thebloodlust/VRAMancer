# Réponse DeepSeek — Questions A1 Path 2 (benchmark vs accelerate)

> En réponse à : `QUESTION_DEEPSEEK_A1_PATH2.md`
> Contexte : palier A1, décision architecte §3 — prouver la parité Path 2 vs Path 1
> (accelerate) avant bascule. Code analysé : `core/backends.py`, `generate()`,
> `infer()`, `_infer_with_kv_cache_impl()`.

---

## Q-A1.1 — Comment forcer Path 2 proprement pour un 14B bf16 ?

### Le mécanisme de dispatch (confirmé dans le code)

```python
# backends.py:1415 — LE point de décision
if hasattr(self.model, 'hf_device_map') and self.model.hf_device_map:
    self.blocks = None       # → Path 1 (accelerate)
    return []
# Sinon → Path 2 (split manuel VRAMancer)
```

**Si `hf_device_map` est présent, Path 2 est IMPOSSIBLE.** Le `split_model()` se
termine avant d'avoir extrait les couches. Donc la condition nécessaire et
suffisante pour Path 2 est : charger le modèle **sans** `device_map="auto"`.

### La bonne invocation

```python
from transformers import AutoModelForCausalLM
import torch

# Étape 1 — Charger en CPU, SANS device_map
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B",
    torch_dtype=torch.bfloat16,
    device_map=None,          # ← PAS de device_map="auto"
    low_cpu_mem_usage=True,   # ← réduit le pic mémoire CPU
)

# Étape 2 — Attacher au backend
backend.model = model
backend.tokenizer = tokenizer

# Étape 3 — Split manuel VRAMancer
backend.split_model(num_gpus=2)
# → N'entre PAS dans le if hf_device_map (ligne 1415)
# → Retire les hooks accelerate (ligne 1433)
# → Extrait les couches, peuple self.blocks, self.block_devices
```

### Est-ce que `split_model` déplace réellement les blocs CPU→GPU ?

Oui, via `assign_blocks_to_gpus()` (appelé à la fin de `split_model`, ligne ~1478).
Cette fonction fait le `.to(f"cuda:{gpu_id}")` sur chaque bloc.

### Le pic mémoire CPU est-il acceptable ?

28 Go bf16 en CPU, c'est beaucoup. Avec `low_cpu_mem_usage=True`, le modèle est
chargé par morceaux (poids → CPU, puis immédiatement `.to(cuda)` pour le bloc en
cours). Le pic RAM réel est ~15-20 Go, pas 28.

Alternative (si le pic est quand même trop haut) :
```python
# Charger directement avec un device_map custom qui place tout sur GPU:0
# puis split_model rééquilibrera
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-14B",
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"},   # tout sur GPU 0 d'abord
    max_memory={0: "20GB"},      # limite pour pas OOM
)
# Puis split_model(2) va rééquilibrer vers GPU 0 et GPU 1
```

---

## Q-A1.2 — La comparaison est-elle apples-to-apples pour la parité de sortie ?

### Ce que Path 2 fait (vérifié dans `_infer_with_kv_cache_impl`)

| Composant | Path 1 (HF generate) | Path 2 (VRAMancer infer) | Équivalent ? |
|---|---|---|---|
| Embedding | Interne HF | `comp["embed"](inputs)` (l.1629) | ✅ Même couche |
| Positional encoding | Interne HF (rotary dans les couches d'attention) | `comp["pos_embed"](pos)` + `hidden_states + pos_emb` (l.1652-1655) | ⚠️ **Différence potentielle** |
| Causal mask | Interne HF | Manuel `pt.triu(float("-inf"))` (l.1693-1701) | ⚠️ **Différence potentielle** |
| Dropout | Interne HF | `comp["drop"](hidden_states)` (l.1657-1658) | ✅ Si dropout=0 en eval |
| Blocs transformer | Internes HF | `block(hidden_states, ...)` manuel (l.1713+) | ✅ Mêmes couches |
| KV cache | StaticCache/DynamicCache HF | `DynamicCache` (l.1665-1669) | ✅ Même classe HF |
| Final norm | Interne HF | `comp["final_norm"]` | ✅ Même couche |
| LM head | Interne HF | `comp["lm_head"]` | ✅ Même couche |

### Les 2 points de divergence numérique

**1. Positional encoding (l.1652-1655)**

Path 2 applique `pos_embed` séparément AVANT les blocs. Dans HF, le rotary
embedding est appliqué À L'INTÉRIEUR de chaque couche d'attention (dans
`LlamaRotaryEmbedding` ou `Qwen2RotaryEmbedding`).

Si `comp["pos_embed"]` est le même module rotary que HF utilise en interne,
et si HF applique aussi rotary séparément → **double application** = sorties
différentes.

**Vérification nécessaire** : Print `comp["pos_embed"]` et vérifier si Qwen2.5
applique le rotary dans la couche d'attention OU via un module externe. Si
c'est interne (ce qui est le cas pour la plupart des modèles récents), alors
Path 2 applique le rotary **deux fois** → les tokens seront différents.

**Solution** : Si `comp["pos_embed"]` est le rotary embedding (pas le sinusoidal
classique), il faut soit :
- Le retirer de `_components` (laisser les couches d'attention le faire)
- Ou le passer à chaque couche d'attention, pas l'appliquer globalement

```python
# Patch rapide pour le bench A1 :
# Vérifier si pos_embed est déjà intégré dans les couches
if comp["pos_embed"] is not None:
    # Qwen2.5 applique le rotary DANS les couches → ne pas l'appliquer ici
    pass  # skip pos_embed, les couches s'en chargent
```

**2. Causal mask (l.1693-1701)**

Path 2 construit `float("-inf")` manuellement. HF utilise `torch.finfo(dtype).min`
ou `-inf`. Normalement équivalent, mais la forme du masque peut différer :
- HF : [batch, 1, seq_len, total_len] avec broadcast
- Path 2 : [batch, 1, seq_len, total_len] (même shape, OK)

Pour le décode (seq_len=1), Path 2 ne crée **pas** de masque (l.1702-1703 :
"no mask needed"). C'est correct — un seul token ne peut pas s'attendre
lui-même dans un masque causal. HF fait pareil.

### Verdict sur la parité

**Risque MODÉRÉ pour le rotary embedding.** C'est le point le plus probable
de divergence token-à-token. À vérifier AVANT de lancer le bench : faire un
test de 10 tokens greedy sans pos_embed, comparer les sorties.

**Risque FAIBLE pour le causal mask.** La logique est correcte. Pour le prefill
(1er token, tout le prompt), le masque est bien causal. Pour le décode, pas de
masque (correct pour seq_len=1).

---

## Q-A1.3 — Mesure tok/s honnête

### Recommandations

**1. Subprocess par chemin : OUI.** Pas de thread, pas de même processus.
Deux processus Python séparés garantissent :
- Pas de contention VRAM (le 1er processus libère tout avant le 2e)
- Pas de pollution de cache GPU
- Pas de state leak entre les deux runs

```python
# Structure du harness
for path_name, env_var in [("accelerate", "VRM_PATH=1"), ("vramancer", "VRM_PATH=2")]:
    subprocess.run([
        sys.executable, "-c", BENCH_SCRIPT,
        "--model", "Qwen/Qwen2.5-14B",
        "--prompt", PROMPT,  # prompt fixe, identique pour les deux
        "--max-tokens", "256",
        "--env", env_var,
    ], capture_output=True, text=True)
```

**2. Warmup : OUI, 50 tokens.** Les premiers tokens paient le chargement du
modèle, le warmup CUDA, la compilation JIT. 50 tokens de warmup (non mesurés)
puis 256 tokens mesurés.

**3. Métrique : decode tok/s uniquement.** Le prefill (TTFT) dépend surtout
de la taille du prompt et du caching. Le decode est le régime établi qui
domine pour les longues générations. Mesurer le **temps de décode hors prefill** :

```python
# Après prefill
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(256):
    next_token = generate_one_step(...)
    torch.cuda.synchronize()
t1 = time.perf_counter()
decode_tok_s = 256 / (t1 - t0)
```

**4. Itérations : 3 runs, prendre la médiane.** Pas la moyenne (sensible aux
outliers). 3 runs×256 tokens = assez pour une mesure stable.

---

## Q-A1.4 — Pièges connus de `infer()`/Path 2

### Piège 1 : Stream de transfert non synchronisé (l.1726-1727)

```python
# backends.py:1726-1727
if _dst_idx not in self._transfer_streams:
    self._transfer_streams[_dst_idx] = pt.cuda.Stream(device=_dst_idx)
```

Le stream de transfert est créé mais la synchronisation après `.to(block_dev)`
n'est **pas garantie** avant la mesure. Si le transfert est asynchrone sur un
stream dédié, la fin de `block(hidden_states)` ne garantit pas que les données
sont arrivées → le bloc suivant peut lire des données partielles.

**Fix** : Ajouter `self._transfer_streams[_dst_idx].synchronize()` après chaque
transfert inter-GPU avant de passer le tenseur au bloc suivant.

### Piège 2 : `.to()` bloquant dans `_transfer_to_device` (l.1539)

```python
# backends.py:1539
return tensor.to(f"cuda:{dst_gpu}")
```

Ce `.to()` est **bloquant** (stream par défaut). Il force une synchronisation
implicite avec tout ce qui tourne sur le stream par défaut du GPU source.
Pour le bench, ça garantit un timing juste, mais ça pénalise Path 2 par
rapport à accelerate qui fait du transfert asynchrone.

**Pour le bench A1** : c'est acceptable — on veut comparer la perf réelle
actuelle, pas la perf théorique après optimisation. Mais le rapport final
devra mentionner "Path 2 transferts synchrones (`.to()`), Path 1 transferts
accelerate (potentiellement async)".

### Piège 3 : Double application de `.to(embed_dev)` (l.1630-1634)

```python
hidden_states = comp["embed"](inputs)
if hasattr(hidden_states, "device") and hidden_states.device != embed_dev:
    hidden_states = hidden_states.to(embed_dev)
if hasattr(hidden_states, "device") and hidden_states.device != embed_dev:  # DOUBLON
    hidden_states = hidden_states.to(embed_dev)
```

C'est un doublon inoffensif (le 2e `.to()` est un no-op si le device est déjà
bon), mais il montre que ce code a été patché sans être nettoyé.

### Piège 4 : DynamicCache vs past_key_values

Path 2 utilise `DynamicCache` (l.1665), la classe standard de transformers 4.45+.
C'est le même format que HF generate() utilise en interne → compatibilité OK.

MAIS : le passage de `past_key_values` entre les appels successifs de `infer()`
dans la boucle `generate()` (l.1933-1934) doit préserver le `DynamicCache` intact.
Si `past_key_values` est mal propagé (ex: copié au lieu d'être référencé), le
cache est vide à chaque étape → le décode recalcule tout le prompt à chaque token
→ tok/s catastrophique.

**Vérifier** : dans la boucle `generate()` (l.1925-1982), `past_blocks` est bien
réinjecté à chaque appel (l.1933: `past_key_values=past_blocks`). C'est correct
si `_infer_with_kv_cache_impl` retourne bien le `DynamicCache` modifié. OK.

### Piège 5 : Pas de `torch.cuda.synchronize()` avant `time.perf_counter()`

Le code de `generate()` ne fait **aucun** `torch.cuda.synchronize()` avant de
mesurer. Les opérations GPU sont asynchrones — le CPU peut avoir fini la boucle
Python alors que le GPU n'a pas fini le dernier token.

**Pour le bench A1** : ajouter `torch.cuda.synchronize()` avant et après la
boucle de décode. Sans ça, les chiffres sont faux (le CPU mesure son temps,
pas celui du GPU).

---

## Résumé pour le harness A1

```python
# Structure recommandée
import subprocess, time, torch

PROMPT = "The capital of France is"  # fixe, ~6 tokens
MODEL = "Qwen/Qwen2.5-14B"
MAX_TOKENS = 256
WARMUP_TOKENS = 50
RUNS = 3

for path in ["accelerate", "vramancer"]:
    for run in range(RUNS):
        # 1. Charger le modèle selon le path
        # 2. Warmup: 50 tokens (non mesurés)
        # 3. torch.cuda.synchronize()
        # 4. t0 = time.perf_counter()
        # 5. Générer 256 tokens
        # 6. torch.cuda.synchronize()
        # 7. t1 = time.perf_counter()
        # 8. decode_tok_s = 256 / (t1 - t0)
        # 9. Vérifier que les 256 tokens sont identiques entre les deux paths
```

### Points de vigilance

1. **Rotary embedding** — vérifier que Path 2 ne l'applique pas en double
2. **Stream synchronisation** — ajouter `synchronize()` après transferts inter-GPU
3. **CUDA sync avant mesure** — obligatoire pour des chiffres honnêtes
4. **Subprocess** — un process par path, pas de thread
5. **Prompt fixe** — le même pour les deux paths, court (~6 tokens)
6. **Greedy uniquement** — temperature=0, pas de sampling (pour la parité exacte)

— DeepSeek
