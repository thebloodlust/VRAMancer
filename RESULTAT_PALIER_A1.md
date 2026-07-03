# Résultat du palier A1 — parité Path 2 (split manuel) vs Path 1 (accelerate)

> Pour l'architecte (Fable). `decision_architecte_7.md §3`, palier A1.
> Verdict mesuré, sans hype. Date : 2026-07-03.

## Critères A1 (rappel)
- **(a)** sorties greedy **identiques au token près** (256 tokens, prompt fixe) entre
  Path 2 (split manuel VRAMancer `infer()` + GpuPipeline) et Path 1 (accelerate `device_map`).
- **(b)** tok/s Path 2 ≥ tok/s accelerate − 5 %.

## VERDICT : ❌ A1 NE PASSE PAS — Path 2 est cassé (bug de forward, pas multi-GPU)

Le critère (a) échoue : Path 2 produit une sortie **dégénérée** (répétitions, ex. « 1. 1. 1. »),
pas identique à accelerate. Le (b) est sans objet tant que (a) échoue.

### Cause racine (isolée par la mesure)
`benchmarks/test_a1_single_gpu.py` fait le split manuel VRAMancer **mais force tout sur
`cuda:0`** (zéro transfert cross-GPU). La sortie manuelle reste **dégénérée** alors que la
génération native HF sur le même GPU est correcte. → **Le bug est dans le forward LOGIQUE
lui-même, pas dans le multi-GPU** (ni transfert, ni ordre des blocs).

Racine précise : le forward manuel (`core/backends.py`, `_infer_with_kv_cache_impl` /
`KVCacheBlock.forward`) **ne passe pas `cache_position` aux couches**. Depuis transformers
≥ 4.45, sans `cache_position` le `DynamicCache` est bien tissé mais **jamais peuplé** → le
modèle ré-attend à vide → sortie dégénérée.

### Hypothèses testées puis RÉFUTÉES par la mesure
- **Patch du masque d'attention** (proposé par DeepSeek) → **no-op**, sortie inchangée.
- **Transfert synchrone** (hypothèse « race de transfert ») → **no-op**, sortie inchangée.
- Le test single-GPU tranche : ni le masque ni le transfert ne sont en cause → c'est
  `cache_position`.

## Path 1 (accelerate) — la voie qui MARCHE
`benchmarks/test_a1_accelerate_baseline.py` : le 14B (Qwen2.5-14B) sur 2 GPU via
`device_map="auto"` :
1. **charge sans OOM** (fix : `max_memory` par GPU + `expandable_segments` — l'OOM venait de
   `_initialize_missing_keys` → `init.normal_(weight.float())` upcast fp32 sur GPU plein) ;
2. **sortie correcte** (contrairement au forward manuel) ;
3. **baseline ≈ 5,41 tok/s**.

## Conclusion & recommandation
- **La voie de production multi-GPU EST accelerate (Path 1)**, qui marche. Le « split manuel »
  Path 2 n'atteint pas la parité car il est cassé au niveau logique.
- **`core/backends.py` a été laissé INTACT** (les patches cache_position/mask/transfert ont été
  testés puis annulés) : puisque accelerate fait déjà le travail correctement, on n'a pas
  introduit de risque dans le chemin de prod.
- **Deux options pour A2** (bascule prod), à ton arbitrage :
  - **Option recommandée** : acter accelerate comme chemin de prod (c'est déjà le cas de fait),
    et cadrer VRAMancer en **orchestrateur au-dessus d'accelerate** (le README a été mis à jour
    en ce sens). Le forward manuel devient `experimental/` ou est déprécié.
  - **Option « réparer Path 2 »** : passer `cache_position` dans `_infer_with_kv_cache_impl` +
    `KVCacheBlock.forward`, re-tester A1. Mais même réparé, la mesure de session montre que le
    split manuel **ne bat pas** accelerate (transfert-bound, P2P indisponible/217, décode-dominé) —
    donc peu d'intérêt vs le laisser à accelerate.

## Fichiers
- `benchmarks/test_a1_single_gpu.py` — prouve le bug LOGIQUE (single-GPU).
- `benchmarks/test_a1_accelerate_baseline.py` — Path 1 OK, 5,41 tok/s.
- `benchmarks/bench_a1_path2_vs_accelerate.py` — harnais de parité (critères a/b).
