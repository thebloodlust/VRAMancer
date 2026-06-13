# Note à l'architecte — Le cluster « frontière inter-GPU » (T7.2 / T7.3 / T7.4 / T7.9) cible un chemin de code inactif en production

**Date** : 2026-06-12 · **Auteur** : agent Phase 7 · **Statut** : DÉCISION REQUISE avant de poursuivre T7.2/T7.3/T7.4 et l'étape 2 de T7.9.

---

## TL;DR

Quatre tâches de la Phase 7 — **T7.2** (FP8 frontière), **T7.3** (rotation Hadamard),
**T7.4** (transport adaptatif) et **l'étape 2 de T7.9** (entrelacement 2 requêtes) —
agissent toutes sur **le même point** : le transfert d'activations `hidden_states`
d'un GPU au suivant dans le pipeline manuel VRAMancer.

Or **ce point n'est jamais atteint dans le chemin d'inférence HF multi-GPU de
production**. La production bf16 multi-GPU passe par `device_map="auto"`
(accelerate), où VRAMancer met `self.blocks = None` et **délègue entièrement les
transferts inter-device à accelerate**, sans aucun point d'interception VRAMancer.

Tant que cette question n'est pas tranchée, implémenter T7.2/T7.3/T7.4 reviendrait
à optimiser un chemin (« Path 2 ») qui ne sert aucune requête aujourd'hui.

---

## Le fait, prouvé par le code

`HuggingFaceBackend.generate()` a 3 chemins (commentaire d'origine,
[core/backends.py:1853-1859](../../core/backends.py#L1853)) :

| Chemin | Condition | Transfert inter-GPU | Atteint en prod ? |
|---|---|---|---|
| **Path 1** | `self.blocks is None or len<=1` → `model.generate()` natif | géré par **accelerate** (hooks internes) | **OUI** (bf16 multi-GPU) |
| **Path 2** | `self.blocks` len>1, `KVCacheBlock` | **`hidden_states.to(block_dev)` VRAMancer** | **NON** |
| **Path 3** | fallback boucle sans KV-cache | idem Path 2 | NON |

### Pourquoi la prod prend Path 1 (et jamais Path 2)

1. **Chargement bf16 multi-GPU** → `kwargs["device_map"] = "auto"`
   ([core/backends.py:1310](../../core/backends.py#L1310)).
2. `split_model()` voit `model.hf_device_map` non vide →
   **`self.blocks = None ; return []`** et log
   *« Using native accelerate inference »*
   ([core/backends.py:1421-1433](../../core/backends.py#L1421)).
3. `generate()` teste `if self.blocks is None ... :` → **Path 1**
   ([core/backends.py:1874](../../core/backends.py#L1874)).

→ Le point de transfert que T7.2 veut instrumenter,
**`hidden_states = hidden_states.to(block_dev, non_blocking=True)`**
([core/backends.py:1736](../../core/backends.py#L1736), dans
`__infer_with_kv_cache_impl`), **n'est jamais exécuté**.

### Et le cas quantifié (NF4/INT8) ?

Pire pour ce cluster : le multi-GPU BnB est **forcé en mono-GPU**
(`device_map={"": best_gpu}`, [core/backends.py:1302](../../core/backends.py#L1302),
commentaire *« multi-GPU BnB broken upstream »*). Donc **aucune frontière
inter-GPU du tout** dans ce mode.

### Le serveur de prod actuel

Tourne en **llama.cpp** (Qwen3.6-35B-A3B GGUF), qui gère son propre découpage
multi-GPU en interne — encore une fois, le transfert frontière VRAMancer n'est
pas dans la boucle.

**Conclusion factuelle** : le transfert frontière manuel VRAMancer (Path 2,
[backends.py:1736](../../core/backends.py#L1736)) n'est atteint que si on charge
explicitement avec `device_map=None` — ce que font **uniquement les scripts de
mesure** que j'ai écrits pour T7.5 et T7.9, jamais le serveur.

---

## Ce que les mesures de Phase 7 ont déjà établi sur ce chemin

Même si Path 2 n'est pas en prod, je l'ai mesuré (c'est le seul endroit où la
frontière existe), et les résultats restent informatifs :

- **T7.5 (FERMÉ)** : sur Path 2, frontière couche 12 de Qwen2.5-7B, l'activation
  change de **~88 % de sa norme par token** (médiane r_t=0.879). → L'encodage
  différentiel **temporel** est mort. *Mais une compression par-tenseur (T7.2)
  ne dépend pas de la stabilité temporelle — elle reste a priori viable.*
- **T7.9 étape 1 (BULLE_SIGNIFICATIVE)** : sur Path 2, oisiveté GPU **68.7 % /
  47.8 %** → la bulle d'entrelacement est réelle… mais **sur Path 2 uniquement**.
  Accelerate (Path 1) a sa propre stratégie de placement et la « bulle » n'y est
  pas forcément exploitable de la même manière.

---

## Impact tâche par tâche

| Tâche | Cible | Bloquée par le mismatch ? |
|---|---|---|
| **T7.2** FP8 frontière | `hidden_states.to()` Path 2 | **OUI** — l'encode/decode FP8 s'insère exactement là où le code n'est pas exécuté en prod |
| **T7.3** rotation Hadamard | poids adjacents à la coupure Path 2 | **OUI** (et dépend du SUCCÈS de T7.2) |
| **T7.4** transport adaptatif | `TransferManager` entre blocs Path 2 | **OUI** — le `TransferManager` n'est sollicité que sur Path 2 |
| **T7.9 étape 2** scheduler 2-slots | boucle bloc-par-bloc Path 2 | **OUI** — l'entrelacement suppose le contrôle explicite des forwards, qu'accelerate ne donne pas |

→ **Tout le sous-thème « frontière / transport inter-GPU » de la Phase 7 dépend
de la même décision d'architecture.** C'est pourquoi je remonte le point une
seule fois pour les 4 tâches, plutôt que de le redécouvrir à chaque tâche.

---

## Décision demandée à l'architecte

**Option A — Faire de Path 2 le chemin de prod multi-GPU bf16.**
Remplacer `device_map="auto"` par le découpage manuel VRAMancer pour le bf16
multi-GPU. Bénéfice : T7.2/T7.3/T7.4/T7.9 deviennent immédiatement pertinentes et
mesurables sur le vrai chemin. Coût/risque : il faut d'abord prouver que Path 2
(KV-cache maison, masque causal, rotary, lm_head manuels) atteint la **parité de
qualité et de débit** avec accelerate sur 7B/14B — non garanti, à benchmarker
avant tout le reste.

**Option B — Ajouter un point d'interception sur Path 1 (accelerate).**
Garder accelerate, mais insérer l'encode/decode FP8 (T7.2) via un hook sur le
module-frontière (`register_forward_pre_hook`/`forward_hook` sur la première
couche du GPU aval — exactement la technique du probe T7.5). Bénéfice : on
optimise le chemin réellement servi, sans réécrire l'inférence. Coût : T7.9
(entrelacement) reste **infaisable** sous accelerate sans contrôle explicite des
forwards → T7.9 étape 2 serait à fermer. T7.4 (`TransferManager`) à repenser car
accelerate fait ses propres `.to()`.

**Option C — Sortir le cluster « frontière » de la Phase 7.**
Acter que l'architecture de prod (accelerate Path 1, ou llama.cpp) ne se prête
pas à l'optimisation manuelle de la frontière sans refonte, et clore
T7.2/T7.3/T7.4/T7.9-étape2 comme **résultats négatifs documentés** (valide per
règle Phase 7 #4). Conserver les acquis non concernés (T7.1 prompt-lookup, T7.6
auto-heal, T7.10 CUDA graphs, T7.11 cache experts MoE — qui, eux, ne dépendent
pas de la frontière).

### Ma recommandation

**Option B en premier**, ciblée sur **T7.2 seul**, comme test à faible coût :
un hook FP8 sur le module-frontière accelerate est ~30 lignes, n'altère pas
l'inférence, et donne un vrai chiffre (octets/token, perplexité wikitext-2) sur
**le chemin de prod**. Si T7.2 passe le seuil de qualité là, on saura que
l'idée a de la valeur et on pourra arbitrer A vs B pour la suite (T7.3/T7.4).
T7.9 étape 2, elle, attend une décision explicite A (sinon fermeture).

À défaut de réponse, je **mets T7.2/T7.3/T7.4 + T7.9-étape2 en pause** et je
continue la Phase 7 par les tâches **indépendantes de la frontière** :
**T7.10** (CUDA Graphs) puis **T7.11** (cache experts MoE, mesure d'abord).

---

## Référence rapide (fichiers)

- Sélection de chemin : [core/backends.py:1874](../../core/backends.py#L1874)
- Mise à `blocks=None` sous accelerate : [core/backends.py:1421-1433](../../core/backends.py#L1421)
- `device_map="auto"` bf16 multi-GPU : [core/backends.py:1310](../../core/backends.py#L1310)
- BnB forcé mono-GPU : [core/backends.py:1302](../../core/backends.py#L1302)
- **Le transfert frontière à instrumenter (Path 2)** : [core/backends.py:1736](../../core/backends.py#L1736)
- Mesures de la frontière déjà faites : [T7.5_report.md](T7.5_report.md), [T7.9_report.md](T7.9_report.md)
