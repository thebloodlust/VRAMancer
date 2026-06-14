# Question DeepSeek — bug de correction du forward manuel (A1, Path 2)

> Contexte : le run A1 14B a tourné. **Path 2 (split manuel VRAMancer, `infer()`)
> a chargé et généré, mais produit une sortie DÉGÉNÉRÉE.** On a besoin de ton
> diagnostic : cause la plus probable + **fix trivial ou problème fondamental ?**
> (ça décide : option A salvageable, ou repli B définitif).

## Le symptôme (mesuré)
- Modèle : Qwen2.5-14B-Instruct, bf16, **2 GPU** (3090 + 5070 Ti).
- Prompt : « Write a Python function that parses a CSV file and returns a dict. »
- Greedy, 256 tokens. Sortie Path 2 :
  > « The following is the following is the following is the following is … »
  (boucle infinie, ~aucun rapport avec le prompt). tok/s = 6.05.
- `path_used=manual_split`, `n_blocks=2`, **`block_devices=["1","0"]`** (← ordre
  inversé : bloc 0 sur GPU1, bloc 1 sur GPU0).
- Le 0.5B en dry-run ne révélait RIEN (les 2 chemins y étaient `manual_split`,
  ils se comparaient à eux-mêmes). Le bug n'apparaît qu'au 14B / vrai split 2-GPU.

## Ce que fait `infer()` (core/backends.py, déjà partagé)
- embed → (pas de pos_embed additif pour Qwen, vérifié) → boucle blocs → norm → head.
- **Transfert cross-GPU** entre blocs (l.1719+) : si `hidden_states.device != block_dev`,
  copie async sur un `_transfer_streams[dst]` puis
  `pt.cuda.current_stream(block_dev).wait_stream(_ts)` (l.~1738).
- **Rotary** : `position_embeddings` calculé une fois via `comp["rotary_emb"]` et
  passé à chaque bloc (vérifié : appliqué 1×).
- **Causal mask** : construit manuellement en prefill (`triu(-inf)`, l.1693-1701) ;
  **pas de mask au décode** (seq_len=1).
- **KV cache** : `DynamicCache` (l.1665), `past_key_values=block_past` réinjecté.

## Nos hypothèses (à confirmer/infirmer par toi)
1. **Sync de transfert insuffisante** (ton piège Q-A1.4 n°1) : `wait_stream` ne
   garantit-il pas réellement que `hidden_states` est arrivé avant le calcul du
   bloc suivant ? Corruption silencieuse → logits faux → répétition.
2. **Ordre de blocs inversé** (`block_devices=[1,0]`) : embed/lm_head sur quel
   device ? Un mismatch embed↔bloc0 ou head↔dernier-bloc casserait la cohérence.
3. **Causal mask / position au décode** : pas de mask en décode + `position_ids`
   recalculés par bloc — risque que l'attention voie mal le passé (→ répétition).
4. **DynamicCache** : le cache est-il bien le MÊME objet propagé à travers les 2
   GPU, ou réinitialisé par bloc (→ le modèle ré-attend mal) ?

## Les questions
- **Q-bug.1** : d'après le symptôme (répétition) + le cross-GPU 2-blocs, quelle est
  la cause la PLUS probable parmi 1-4 (ou autre) ?
- **Q-bug.2** : **fix trivial** (ex. `synchronize()` au lieu de `wait_stream`, ou
  forcer l'ordre des blocs, ou un `.contiguous()` après transfert) **ou
  fondamental** (le forward manuel ne peut pas être correct cross-GPU sans
  refonte) ? C'est LA question qui décide A vs B.
- **Q-bug.3** : un patch minimal que tu proposerais pour tester l'hypothèse n°1
  (remplacer le `wait_stream` par une vraie synchro) ?

Bonus (séparé) : Path 1 (accelerate) OOM au chargement 14B via
`_initialize_missing_keys → init.normal_(weight.float())` (upcast fp32 sur GPU1
plein). C'est l'issue #6 généralisée au bf16. Un `max_memory` par GPU ou
`expandable_segments` suffit-il, selon toi ?

— Pour croisement avec l'analyse d'Opus (qui investigue `infer()` en parallèle).
