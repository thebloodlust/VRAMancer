# A1 — Parité Path 2 (split manuel) vs Path 1 (accelerate)

Modèle : `Qwen/Qwen2.5-14B-Instruct` · bf16 2-GPU · greedy · 48 tokens ·
prompt fixe : « Write a Python function that parses a CSV file and returns a dict. »

> ⚠️ Généré par un scaffold non encore validé sur matériel. Invocation
> Path 2 à confirmer (QUESTION_DEEPSEEK_A1_PATH2.md).

| Chemin | Emprunté (vérifié) | tok/s | sortie |
|---|---|---|---|
| Path 1 accelerate | ? | ÉCHEC | - car |
| Path 2 manuel | manual_split (2 blocs) | 4.16 tok/s | 272 car |

## Critères A1
- (a) sorties identiques (token-proxy) : **None**
- (b) tok/s Path 2 ≥ accelerate − 5 % : **None**
- **VERDICT : FAIL/À REVOIR**

Notes : Un worker a échoué — voir stderr_tail dans le JSON.

## Méthodo (après revue DeepSeek Q-A1.3/.4)
- tok/s = **médiane de 3 runs**, warmup 50 tokens, `cuda.synchronize()`
  avant/après mesure. Prompt court => prefill négligeable => end-to-end ≈ decode.
- Invocation Path 2 (load CPU + `split_model`) **confirmée** (Q-A1.1).
- Rotary : crainte de double-application **vérifiée fausse** sur Qwen2.5
  (`pos_embed` None ; rotary appliqué une fois). Voir commentaire du worker.

## Limites restantes
- Comparaison de sortie = chaînes décodées exactes (proxy token-pour-token) ;
  pour une vraie égalité d'IDs, exposer `out_ids` côté backend.
- Pièges Path 2 à surveiller au run (DeepSeek Q-A1.4) : sync des streams de
  transfert inter-GPU (le code fait `wait_stream` à 1738), propagation du
  `DynamicCache` entre étapes (sinon decode recalcule tout => tok/s effondré).
