# Plan V5 (MASTER) — Suite de V4 : fixes ciblés, gap vLLM, hot-path honesty

> **POUR L'AGENT EXÉCUTANT — LIRE ATTENTIVEMENT AVANT DE COMMENCER :**
>
> Ce plan est **exhaustif et autonome**. Tu n'as PAS besoin de revenir demander un autre plan.
> Il continue strictement le travail amorcé en V4 (`docs/reports/PLAN_ACTION_V4.md` +
> `resultat_v4.md`). V5 se concentre sur des fixes mesurables, courts, à faible risque,
> qui découlent directement des trous laissés par V4.
>
> **Règles d'or (identiques à V4) :**
> 1. Lis chaque section dans l'ordre. Ne saute jamais de validation.
> 2. Après chaque tâche numérotée (P&lt;x&gt;.&lt;y&gt;), fais un commit ATOMIQUE préfixé `[P<x>.<y>]`.
> 3. Si une tâche échoue 2 fois → **STOP**, écris dans `resultat_v5.md` une section
>    `[BLOCKED@P<x>.<y>]` avec la cause, passe à la phase suivante. Ne brute-force pas.
> 4. Si une mesure ne montre pas le gain attendu → **REVERT honnêtement** (`git revert HEAD`)
>    et documente l'échec sous `[NEGATIVE@P<x>.<y>]`. Ne jamais arrondir vers le haut.
>    Ne jamais cacher un échec.
> 5. Aucune modif dans :
>    - `_deprecated/`
>    - `core/security/__init__.py`
>    - `core/security/startup_checks.py`
>    - `tests/test_chaos_concurrency.py`
>    - `csrc/paged_attention_kernel.cu`
>    - `core/paged_attention.py`
>    - `core/paged_attention_cuda.py`
>    - `rust_core/src/` (sauf P4 explicite, voir restrictions)
> 6. Aucune désactivation de tests existants pour faire passer la suite.
> 7. Aucun `git push`, `git push --force`, merge vers `main`, ouverture de PR.
>    L'utilisateur s'en chargera.
> 8. Toujours `source .venv/bin/activate` avant `python`/`pytest`, sinon `ModuleNotFoundError`.
> 9. Tous les commits préfixés `[P<x>.<y>]` (ex: `[P2.3] ...`).
> 10. Honnêteté > marketing. Si un bench est bruité, écris `INDÉTERMINÉ`, pas un beau chiffre.
> 11. Pas de nouveau fichier markdown hors `resultat_v5.md` et `docs/reports/*_V5_*.md`
>     sauf si le plan le demande explicitement.

**Version :** v5.0-master
**Date :** 2026-05-04
**Branche source :** `chore/sonnet-plan-v4` (HEAD = `5343039`)
**Branche cible :** `chore/sonnet-plan-v5`
**Auteur :** Architecte Claude Opus 4.7
**Exécutant attendu :** Claude Sonnet 4.6 (ou agent équivalent suivant pas-à-pas)
**Pré-requis :** V4 terminé et mergé/disponible (la branche `chore/sonnet-plan-v4` doit être
checkout-able). Tests baseline V4 = 1 failed (pre-existing) + 1074 passed + 42 skipped.

---

## Table des phases

| Phase | Titre                                                         | Effort  | Risque | Bloquant si échec ? |
|-------|---------------------------------------------------------------|---------|--------|---------------------|
| P0    | Préparation + baseline V5                                     | 5 min   | Faible | OUI                 |
| P1    | ContinuousBatcher : auto-start dans `generate()`              | 1h      | Moyen  | NON (revert OK)     |
| P2    | TransferManager : labels honnêtes (RUST_P2P)                  | 30 min  | Faible | NON                 |
| P3    | vLLM gap : `use_cache=True` explicite + audit HF flags        | 45 min  | Faible | NON                 |
| P4    | PyO3 `transfer_async` — exposer variant non-bloquant          | 2-3h    | Élevé  | NON (revert OK)     |
| P5    | Silent exceptions sweep — hot paths (3 modules ciblés)        | 1h      | Faible | NON                 |
| P6    | Hetero advantage : bench reproductible 14B 5070Ti+3090        | 1h      | Faible | NON (skip si OOM)   |
| P7    | Example `usb4_distributed_vram.py` — fix ou déprécier         | 30 min  | Faible | NON                 |
| P8    | Repo root cleanup — déplacer `bench_*.{json,log,txt}`         | 20 min  | Faible | NON                 |
| P9    | TODO markers : adresser les 3 ouverts ou documenter           | 30 min  | Faible | NON                 |
| P10   | Tests coverage : auto-start batcher + label consistency       | 45 min  | Faible | NON                 |
| P11   | Documentation : TECHNICAL_DEBT + CHANGELOG → 1.6.0            | 30 min  | Faible | NON                 |
| P12   | HF Browser : choix de n'importe quel modèle HuggingFace       | 1h30    | Moyen  | NON                 |
| P13   | DeepSeek V4 Flash dual-GPU + engram KV offload (jusqu'à 200GB)| 2-3h    | Élevé  | NON (skip si OOM)   |
| P14   | Validation finale (tests + smoke + sanity)                    | 15 min  | Faible | OUI                 |
| P15   | `resultat_v5.md` SUMMARY + verdict                            | 30 min  | Faible | OUI                 |

**Total estimé :** ~13-16h équivalent humain, sur 2-3 sessions.
**Ordre recommandé :** strictement séquentiel (P0 → P15). P4 peut être déplacé en dernière
position si P1-P3 ont consommé la session. P13 dépend de P12 (utilise le browser pour
choisir le modèle DeepSeek).

---

# P0 — Préparation et baseline V5

## P0.1 — Setup branche

```bash
cd /home/jeremie/VRAMancer/VRAMancer
source .venv/bin/activate
git status                                # doit être propre
git --no-pager log --oneline -1           # doit être 5343039 ou descendant
git checkout chore/sonnet-plan-v4         # base V5 = tip de V4
git --no-pager log --oneline -3
git checkout -b chore/sonnet-plan-v5
```

**Si la branche `chore/sonnet-plan-v5` existe déjà** (session reprise) :

```bash
git checkout chore/sonnet-plan-v5
git --no-pager log --oneline -10          # voir où on en est
```

Identifie la dernière phase complétée par les messages de commit (`[P<x>.<y>]`) et reprends à
la phase suivante.

## P0.2 — Crée `resultat_v5.md`

```bash
cat > resultat_v5.md << 'EOF'
# Résultat Plan V5 (MASTER)

**Date début :** YYYY-MM-DD HH:MM
**Branche :** chore/sonnet-plan-v5
**Plan :** docs/reports/PLAN_ACTION_V5.md
**Base :** chore/sonnet-plan-v4 @ 5343039

## [BASELINE]
(à remplir P0.3)

## [P1] — ContinuousBatcher auto-start
## [P2] — TransferManager labels honnêtes
## [P3] — vLLM gap : use_cache + audit
## [P4] — PyO3 transfer_async
## [P5] — Silent exceptions sweep
## [P6] — Hetero advantage bench
## [P7] — usb4_distributed_vram example
## [P8] — Repo root cleanup
## [P9] — TODO markers
## [P10] — Tests coverage
## [P11] — Documentation 1.6.0
## [P12] — HF browser load
## [P13] — DeepSeek + engram
## [P14] — Validation finale
## [SUMMARY]
EOF
git add resultat_v5.md
git commit -m "[P0.2] init resultat_v5.md skeleton"
```

## P0.3 — Baseline tests + smoke

```bash
source .venv/bin/activate
pytest tests/ -q --tb=line 2>&1 | tail -5
python tests/smoke.py 2>&1 | tail -5
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('cuda:', torch.cuda.is_available()); \
  [print(f'GPU{i}', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
```

Reporte dans `resultat_v5.md` :

```markdown
## [BASELINE]

**Tests :** N failed, M passed, K skipped
**Smoke :** pytest via tests/smoke.py → exit 0
**GPU mapping :**
- `nvidia-smi` ordre PCI_BUS_ID : ...
- `torch.cuda` ordre FAST_FIRST : ...
**HEAD :** &lt;sha actuel&gt;
```

```bash
git add resultat_v5.md
git commit -m "[P0.3] baseline V5: <résumé>"
```

**Bail-out :** si la baseline a plus que 1 failure (autre que `test_health_imports_fault_manager`
pré-existant identifié en V4), STOP et documente — quelque chose a régressé entre V4 et V5.

---

# P1 — ContinuousBatcher : auto-start dans `generate()`

> **CRITICAL FIX.** V4 P4 a diagnostiqué le bug : `VRM_CONTINUOUS_BATCHING=1` crée le batcher
> mais ne le démarre pas. `generate()` vérifie `self.continuous_batcher._running` (False) →
> route vers `_protected_generate`. Conséquence : N=4 requêtes concurrentes ne batchent pas.
>
> Fix simple : si `VRM_CONTINUOUS_BATCHING=1` et batcher existe et `_running=False`, démarrer
> le batcher avant le check de routage.

## P1.1 — Audit ciblé (5 min)

Lis ces fichiers/lignes pour comprendre le contexte avant d'éditer :

- `core/inference_pipeline.py:528-541` : check `_running` actuel
- `core/inference_pipeline.py:1386-1403` : init du batcher (commentaire `# Don't auto-start`)
- `core/continuous_batcher.py:149` : `self._running = False`
- `core/continuous_batcher.py:216-232` : `start()` lance le thread
- `core/continuous_batcher.py:292` : `while self._running:` boucle principale

Confirme aussi qu'il existe une méthode `stop()` ou `shutdown()` symétrique. Si oui, note le
numéro de ligne pour P1.4 (cleanup).

```bash
grep -n "def stop\|def shutdown\|def close" core/continuous_batcher.py
```

## P1.2 — Implémente l'auto-start derrière flag explicite

**Fichier :** `core/inference_pipeline.py`

**Modification 1 — Patch dans `generate()` autour des lignes 528-541** :

Avant (state actuel) :

```python
elif (self.continuous_batcher is not None
      and self.continuous_batcher._running):
    # Route through continuous batcher for automatic
    # request batching when multiple requests are in flight
    future = self.continuous_batcher.submit(
        prompt,
        ...
    )
```

Après :

```python
elif self.continuous_batcher is not None and (
    self.continuous_batcher._running
    or os.environ.get("VRM_CONTINUOUS_BATCHING", "0") == "1"
):
    # Auto-start the batcher on first generate() call when the env flag is set.
    # Without this, the batcher remains idle and generate() falls through to
    # the protected (single-request) path. See V4 P4 diagnosis.
    if not self.continuous_batcher._running:
        try:
            self.continuous_batcher.start()
            _logger.info("ContinuousBatcher auto-started via VRM_CONTINUOUS_BATCHING=1")
        except Exception as e:
            _logger.warning(
                "ContinuousBatcher auto-start failed (%s) — falling back to "
                "single-request path", e,
            )
            # Fall through to else branch below
            result = self._protected_generate(prompt, gen_kwargs)
        else:
            future = self.continuous_batcher.submit(
                prompt,
                max_new_tokens=gen_kwargs.get("max_new_tokens", max_new_tokens),
                temperature=gen_kwargs.get("temperature", temperature),
                top_k=gen_kwargs.get("top_k", top_k),
                top_p=gen_kwargs.get("top_p", top_p),
            )
            result = future.result(
                timeout=(_flags.GENERATE_TIMEOUT if _flags else float(os.environ.get("VRM_GENERATE_TIMEOUT", "300")))
            )
    else:
        # Already running — same path as before
        future = self.continuous_batcher.submit(
            prompt,
            max_new_tokens=gen_kwargs.get("max_new_tokens", max_new_tokens),
            temperature=gen_kwargs.get("temperature", temperature),
            top_k=gen_kwargs.get("top_k", top_k),
            top_p=gen_kwargs.get("top_p", top_p),
        )
        result = future.result(
            timeout=(_flags.GENERATE_TIMEOUT if _flags else float(os.environ.get("VRM_GENERATE_TIMEOUT", "300")))
        )
```

**Note importante :** garde la lecture d'`os.environ` car `_flags` peut ne pas exposer ce flag
selon les versions de `core/env_flags.py`. Si `_flags.CONTINUOUS_BATCHING` existe, préfère
l'utiliser ; sinon fallback `os.environ`.

**Modification 2 — Mets à jour le log d'init lignes 1399-1400** :

Avant :

```python
# Don't auto-start — only start on first submit or explicit call
_logger.info("ContinuousBatcher ready (call pipeline.submit() to use)")
```

Après :

```python
# Auto-start happens lazily in generate() when VRM_CONTINUOUS_BATCHING=1.
# Pre-V5 behavior preserved when the flag is unset.
_logger.info(
    "ContinuousBatcher ready (auto-start on generate() if "
    "VRM_CONTINUOUS_BATCHING=1, else manual via pipeline.submit())"
)
```

## P1.3 — Test fonctionnel manuel

```bash
source .venv/bin/activate
VRM_CONTINUOUS_BATCHING=1 VRM_DEBUG=1 python -c "
from core.inference_pipeline import InferencePipeline
import os
os.environ['VRM_BACKEND_ALLOW_STUB'] = '1'
# Use the smallest backend that has a real generate() if available, sinon stub.
# Note : si aucun pipeline réel ne se monte sans modèle, ce test peut juste
# vérifier que start() est invoqué — voir P10 pour un test pytest propre.
print('OK — manual integration smoke')
"
```

Si tu n'as pas de modèle local pour tester end-to-end, **passe directement à P1.4** (les vrais
tests sont en P10) — ne perds pas de temps sur une intégration manuelle bruitée.

## P1.4 — Re-bench concurrent (compare V4 P4.2)

```bash
# V4 P4.2 a produit bench_stress_concurrent_v4.py — réutilise-le.
ls benchmarks/bench_stress_concurrent_v4.py 2>&1 || echo "ABSENT — skip P1.4"
```

Si présent, lance avec et sans flag :

```bash
# OFF (baseline = même que V4 P4.2 colonne "OFF")
unset VRM_CONTINUOUS_BATCHING
python benchmarks/bench_stress_concurrent_v4.py --concurrency 1,4,8 \
  > /tmp/bench_v5_p1_off.log 2>&1
tail -30 /tmp/bench_v5_p1_off.log

# ON (V5 fix actif)
VRM_CONTINUOUS_BATCHING=1 python benchmarks/bench_stress_concurrent_v4.py \
  --concurrency 1,4,8 > /tmp/bench_v5_p1_on.log 2>&1
tail -30 /tmp/bench_v5_p1_on.log
```

Reporte dans `resultat_v5.md` la table comparée :

```markdown
**P1.4 Bench résultats (V5 fix actif) :**

| Mode      | N=1       | N=4       | N=8       |
|-----------|-----------|-----------|-----------|
| OFF       | XX tok/s  | XX tok/s  | XX tok/s  |
| ON V4     | XX tok/s  | XX tok/s  | XX tok/s  |  ← copié de resultat_v4.md
| ON V5     | XX tok/s  | XX tok/s  | XX tok/s  |
```

**Critère de succès :** ON V5 ≥ 1.5× ON V4 sur N=4 ou N=8 (le batching doit faire effet).
**Si ON V5 ≈ ON V4** → le fix n'a rien changé, écris `[NEGATIVE@P1.4]`, fais `git revert` du
commit P1.2, et passe à P2. **Ne masque pas l'échec.**

## P1.5 — Commit

```bash
git add core/inference_pipeline.py
git diff --cached --stat
git commit -m "[P1.2] auto-start ContinuousBatcher on generate() when VRM_CONTINUOUS_BATCHING=1

Fixes V4 P4 diagnosis: batcher was created but never started, so generate()
fell through to single-request _protected_generate. Now auto-starts on first
generate() when the flag is set.

Bench (Qwen-7B, N=4 concurrent): <fill avec résultats P1.4>"
```

Si P1.4 a montré gain → ajoute le résultat. Si bench skip (pas de modèle) → écris
"bench skipped (no local model)" dans le commit body.

---

# P2 — TransferManager : labels honnêtes (`RUST_P2P`)

> V4 P1.4 a documenté `TRANSFER_MANAGER_LABEL_INCORRECT` : `_get_method_for()` retourne la
> chaîne `"CPU_STAGED"` alors que les transferts réels passent par le bypass Rust P2P
> (172-190 Gbps). Label cosmétique, mais source de confusion future. V5 : on corrige.
>
> Bonus : V4 P1.4 a aussi noté incohérence type-retour (string vs `TransportMethod` enum
> selon les call sites). On harmonise.

## P2.1 — Audit ciblé (5 min)

Lis :

- `core/transfer_manager.py:74` : enum `CPU_STAGED = auto()`
- `core/transfer_manager.py:325-331` : `_get_gpu_pipeline()` (création Rust GpuPipeline)
- `core/transfer_manager.py:378-466` : `send_activation()` (entry point)
- `core/transfer_manager.py:504-570` : `_execute_transfer()` (où le Rust path est sélectionné)
- `core/transfer_manager.py:559-562` : `direct_vram_copy()` Rust call
- `core/transfer_manager.py:631, 746` : usage enum `TransportMethod.CPU_STAGED`
- `core/transfer_manager.py:849` : log via `_get_method_for()` string
- `core/transfer_manager.py:956-966` : `_get_method_for()` (return string)

Note les call sites de `_get_method_for()` :

```bash
grep -n "_get_method_for" core/transfer_manager.py
grep -rn "_get_method_for" core/ tests/ | grep -v __pycache__
```

## P2.2 — Ajoute le label `RUST_P2P`

**Fichier :** `core/transfer_manager.py`

**Modification 1 — Ajoute la valeur à l'enum (autour de la ligne 74)** :

```python
class TransportMethod(IntEnum):
    CUDA_P2P = auto()
    NCCL = auto()
    CPU_STAGED = auto()
    RUST_P2P = auto()        # NEW V5 — Rust GpuPipeline (cuMemcpyPeerAsync triple-buffered)
    # ... autres si présents
```

(Adapte selon la définition réelle — `IntEnum`, `Enum`, etc. Garde les valeurs
existantes inchangées en ajoutant `RUST_P2P` à la fin.)

**Modification 2 — Patch `_get_method_for()` ligne 956-966** :

Avant :

```python
def _get_method_for(self, src: int, dst: int) -> str:
    if self._can_p2p(src, dst):
        return "CUDA_P2P"
    if self._nccl_initialized:
        return "NCCL"
    return "CPU_STAGED"
```

Après :

```python
def _get_method_for(self, src: int, dst: int) -> str:
    """Return the *advertised* transport label for (src, dst).

    Notes:
        When the Rust GpuPipeline is active for this pair (see
        ``_get_gpu_pipeline``), actual transfers may use cuMemcpyPeerAsync
        directly even when this label says ``CUDA_P2P``. The honest label
        ``RUST_P2P`` is used when the Rust fast-path is the selected
        execution path (decided in ``_execute_transfer``).
    """
    # Prefer Rust P2P label if a GpuPipeline is already cached for this pair —
    # it means _execute_transfer will route through the Rust bypass.
    pair = (src, dst)
    if pair in self._gpu_pipelines:
        return "RUST_P2P"
    if self._can_p2p(src, dst):
        return "CUDA_P2P"
    if self._nccl_initialized:
        return "NCCL"
    return "CPU_STAGED"
```

**Note :** ajuste `self._gpu_pipelines` au nom réel de l'attribut (vérifie autour de
ligne 325-331).

**Modification 3 — Si call site ligne 849 utilise la string pour journaliser, garde la
string** mais documente l'invariant. Si un autre call site convertit string → enum, ajoute
le mapping `"RUST_P2P" → TransportMethod.RUST_P2P` (le plus simple : `getattr(TransportMethod, label, TransportMethod.CPU_STAGED)`).

## P2.3 — Mise à jour TECHNICAL_DEBT.md

**Fichier :** `docs/reports/TECHNICAL_DEBT.md`

Édite la ligne sur `_get_method_for()` (autour de la ligne 36 du fichier actuel) :

Avant :

```markdown
| `_get_method_for()` retourne `CPU_STAGED` alors que `send_activation()` utilise le bypass Rust P2P (172-190 Gbps) | Label cosmétique uniquement... |
```

Après :

```markdown
| ~~`_get_method_for()` retourne `CPU_STAGED` alors que `send_activation()` utilise le bypass Rust P2P (172-190 Gbps)~~ ✅ **Résolu V5 P2** : nouveau label `RUST_P2P` retourné quand un `GpuPipeline` est actif pour la paire (src, dst). Voir `core/transfer_manager.py:_get_method_for`. | — |
```

## P2.4 — Tests

```bash
source .venv/bin/activate
pytest tests/test_transfer_manager_basic.py tests/test_transfer_manager_unit.py -v --tb=line
pytest tests/ -q --tb=line 2>&1 | tail -3
```

**Critère de succès :** aucune régression vs baseline P0.3. Si un test échoue car il assertait
sur `"CPU_STAGED"` pour un cas qui retourne maintenant `"RUST_P2P"`, mets à jour le test
(le nouveau label est plus juste).

## P2.5 — Commit

```bash
git add core/transfer_manager.py docs/reports/TECHNICAL_DEBT.md tests/
git diff --cached --stat
git commit -m "[P2.2] honest TransportMethod label: add RUST_P2P for Rust GpuPipeline path

V4 P1.4 documented that _get_method_for() returned 'CPU_STAGED' even when
the actual transport went through the Rust GpuPipeline bypass at 172-190 Gbps.
V5: when a cached GpuPipeline exists for (src,dst), report RUST_P2P instead.

No behavior change — labels in logs and metrics now match reality.
TECHNICAL_DEBT.md entry struck through with V5 P2 reference."
```

---

# P3 — vLLM gap : `use_cache=True` explicite + audit HF flags

> V4 P5 a benché VRAMancer 27.5 tok/s vs vLLM 51.5 tok/s sur Qwen2.5-7B 1-GPU (vLLM +87%).
> L'audit V5 a révélé que le path single-GPU dans `core/backends.py:1799-1835` appelle
> `model.generate()` HF natif **sans `use_cache=True` explicite** — on dépend du défaut
> du modèle. C'est probablement déjà True pour les LLM standards, mais on le rend explicite
> pour éviter les surprises et pour pouvoir mesurer.

## P3.1 — Audit confirmé (5 min)

```bash
sed -n '1795,1840p' core/backends.py
grep -n "use_cache" core/backends.py
```

Note tous les call sites de `model.generate()` dans `backends.py` (il peut y en avoir
plusieurs : path 1 single-GPU, path 2 multi-GPU pipeline, path TurboEngine).

Liste les flags HF explicites passés actuellement et ceux **manquants critiques** :

| Flag                  | Présent ? | Importance perf | À ajouter en V5 ? |
|-----------------------|-----------|-----------------|-------------------|
| `use_cache=True`      | ?         | ÉLEVÉE          | OUI si absent     |
| `pad_token_id`        | OUI       | —               | déjà OK           |
| `do_sample`           | conditionnel | moyen        | déjà OK           |
| `num_beams`           | ?         | basse           | non               |
| `early_stopping`      | ?         | basse           | non               |

Reporte ce tableau dans `resultat_v5.md` section P3.

## P3.2 — Patch : `use_cache=True` explicite

**Fichier :** `core/backends.py` (ligne ≈ 1825-1832, dans la branche `if self.blocks is None or len(self.blocks) <= 1`)

Avant :

```python
out_ids = self.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    pad_token_id=self.tokenizer.pad_token_id,
    **gen_kwargs,
)
```

Après :

```python
# V5 P3.2 — explicit use_cache=True. Without this, we depended on the
# model's default which is True for most LLMs but unspecified for some
# fine-tunes / custom heads. Explicit > implicit on the hot path.
gen_kwargs.setdefault("use_cache", True)
out_ids = self.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=max_new_tokens,
    pad_token_id=self.tokenizer.pad_token_id,
    **gen_kwargs,
)
```

Applique le **même patch** à TOUS les autres call sites `model.generate()` dans
`backends.py` (hors path multi-GPU pipeline qui a sa propre boucle décode et utilise déjà
le KV cache via PagedKV).

## P3.3 — Re-bench Qwen2.5-7B 1-GPU vs V4 P5.2

> Si tu n'as pas accès au modèle Qwen2.5-7B (téléchargé pendant V4), skip ce sous-point et
> documente "bench skipped (model not present locally)".

```bash
source .venv/bin/activate
ls ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct 2>&1 | head -3 || \
  echo "MODEL ABSENT — skip P3.3"
```

Si présent :

```bash
# Reprend EXACTEMENT le script bench V4 P5.2 (cherche-le)
ls benchmarks/ | grep -i vllm
ls benchmarks/ | grep -i qwen
```

Lance le bench VRAMancer (BF16, 100 tokens, 3 runs) et compare au V4 P5.2 (27.53 tok/s
médiane). Reporte dans `resultat_v5.md` :

```markdown
**P3.3 Bench Qwen2.5-7B 1-GPU avec use_cache explicite :**

| Run    | tok/s |
|--------|-------|
| V4 baseline (use_cache implicite) | 27.53 médiane |
| V5 (use_cache=True explicite)     | XX médiane    |
| Δ                                 | +X %          |
```

**Critère :** si Δ ≥ +5% → c'est un vrai gain, on garde. Si Δ ∈ [-2%, +5%] → bruit, on garde
quand même (correction d'honnêteté). Si Δ ≤ -2% → REVERT P3.2 et `[NEGATIVE@P3.3]`.

## P3.4 — Audit honnête : que reste-t-il du gap vLLM ?

Reporte dans `resultat_v5.md` une analyse honnête en 5-10 lignes des facteurs qui restent
non-adressés :

```markdown
**P3.4 Gap vLLM restant après V5 P3 :**

vLLM avantages structurels qui ne seront pas comblés sans refonte :
- Paged attention matérialisée + CUDA Graphs baked single-GPU
- Worker loop C++ (vs Python GIL côté VRAMancer)
- Préallocation 90% VRAM pour KV blocks (vs allocation à la demande VRAMancer)
- Chunked prefill agressif

VRAMancer avantages structurels (gardés en V5) :
- Multi-GPU hétérogène (vLLM OOM, démontré V4 P5.3 — Qwen2.5-14B 5070Ti+3090)
- Allocation VRAM-proportionnelle
- Hot-plug GPU
- VRAM lending cross-machine

Conclusion : VRAMancer ne battra pas vLLM single-GPU homogène, mais est **seul** sur
multi-GPU hétérogène. C'est le positionnement honnête à pousser.
```

## P3.5 — Commit

```bash
git add core/backends.py
git diff --cached --stat
git commit -m "[P3.2] explicit use_cache=True in HF generate() single-GPU paths

V4 P5 measured a 1.87x gap vs vLLM on Qwen2.5-7B single-GPU. V5 audit found
HF generate() did not pass use_cache=True explicitly — relying on model
default (True for most LLMs but not guaranteed). Explicit > implicit on the
hot path.

Bench: <fill avec P3.3>"
```

---

# P4 — PyO3 `transfer_async` — exposer variant non-bloquant

> **Phase ÉLEVÉE risque.** V4 P2.5 a noté : `VRM_TRANSFER_OVERLAP=1` ne donne ~0% de gain
> car la Strategy 1.5 Rust ignore le `torch.cuda.Stream` Python passé. Pour avoir un vrai
> gain de stream overlap, il faut exposer côté Rust une variante async qui accepte un
> stream handle (cuStream) et ne bloque pas.
>
> **Cette phase touche `rust_core/src/`** (zone normalement interdite). C'est l'EXCEPTION
> autorisée par ce plan. **Garde les changements minimaux et entièrement encapsulés derrière
> le flag `VRM_TRANSFER_ASYNC=1`.**
>
> Si tu n'as pas confiance dans le build Rust ou si `cargo build --release` échoue à la
> baseline → **skip P4 entièrement**, documente `[SKIPPED@P4]` et passe à P5.

## P4.1 — Pré-flight Rust

```bash
cd /home/jeremie/VRAMancer/VRAMancer
cargo --version
rustc --version
ls rust_core/src/
cargo build --release --manifest-path rust_core/Cargo.toml 2>&1 | tail -10
```

Si `cargo build --release` échoue → **STOP P4**, écris `[BLOCKED@P4.1]` dans `resultat_v5.md`,
passe à P5.

Si OK, importe Python :

```bash
source .venv/bin/activate
python -c "import vramancer_rust; print(dir(vramancer_rust))" | tr ',' '\n' | grep -i transfer
```

Note les fonctions exposées actuelles (probablement `direct_vram_copy`, `GpuPipeline`).

## P4.2 — Conception minimale

**Objectif :** ajouter une fonction Rust `direct_vram_copy_async(src, dst, ptr, nbytes, stream_handle) -> Future-like` qui :

1. Accepte un `stream_handle: u64` (cast de `cudaStream_t`).
2. Lance `cuMemcpyPeerAsync` sur ce stream.
3. **Ne fait PAS de `cudaStreamSynchronize`** côté Rust.
4. Retourne immédiatement (Python sync via `torch.cuda.Stream.synchronize()` plus tard).

**Code Rust à ajouter dans `rust_core/src/lib.rs` (ou le module pertinent — vérifie la structure)** :

```rust
/// V5 P4 — async variant of direct_vram_copy.
///
/// Caller MUST synchronize the stream from Python side after calling this
/// (e.g. `torch.cuda.Stream.synchronize()`).
///
/// # Safety
/// - `tensor_ptr` must be a valid CUDA device pointer on `src_gpu`.
/// - `stream_handle` must be a valid `cudaStream_t` (cast to u64).
/// - The caller MUST keep the source tensor alive until the stream completes.
#[pyfunction]
pub fn direct_vram_copy_async(
    src_gpu: i32,
    dst_gpu: i32,
    tensor_ptr: u64,
    nbytes: u64,
    stream_handle: u64,
) -> PyResult<bool> {
    use cuda_driver_sys::*;
    unsafe {
        let stream = stream_handle as cudaStream_t;
        let src_ptr = tensor_ptr as cudaPtr;
        // Allocate dst on dst_gpu — caller responsible for pre-allocation.
        // For V5 prototype, assume dst is also pre-allocated and we receive
        // its pointer in a future overload. For now, route to existing sync
        // path if dst pointer not provided.
        let result = cuMemcpyPeerAsync(
            /* dst */ ..., /* dst_ctx */ ...,
            /* src */ src_ptr, /* src_ctx */ ...,
            nbytes as usize,
            stream,
        );
        Ok(result == CUDA_SUCCESS)
    }
}
```

> **CAVEAT IMPLEMENTATEUR :** la signature ci-dessus est un **squelette indicatif**, pas du
> code prêt à compiler. La vraie implémentation doit s'aligner sur l'API actuelle de
> `direct_vram_copy` (regarde le code Rust existant et copie sa structure). Le but est de
> n'introduire qu'**une seule** nouvelle fonction `*_async` qui fait exactement comme la sync
> mais sans synchroniser à la fin.

**Si la signature actuelle ne le permet pas trivialement** (par ex. dst_ptr est alloué côté
Rust pendant le call) → **STOP P4**, écris `[BLOCKED@P4.2 — needs deeper Rust refactor]` et
passe à P5. Ne te lance pas dans une refonte Rust.

Enregistre la nouvelle fonction dans le `#[pymodule]` :

```rust
m.add_function(wrap_pyfunction!(direct_vram_copy_async, m)?)?;
```

## P4.3 — Build + import smoke

```bash
cargo build --release --manifest-path rust_core/Cargo.toml 2>&1 | tail -20
maturin develop --release --manifest-path rust_core/Cargo.toml 2>&1 | tail -10 || \
  cd rust_core && python -m maturin develop --release && cd ..
source .venv/bin/activate
python -c "import vramancer_rust; print('async fn:', hasattr(vramancer_rust, 'direct_vram_copy_async'))"
```

Si `direct_vram_copy_async` n'apparaît pas → bug PyO3 / build, debug ou bail.

## P4.4 — Wire côté Python derrière flag

**Fichier :** `core/transfer_manager.py`

Dans `_execute_transfer()` (lignes ≈ 504-570), ajoute un branch derrière le flag
`VRM_TRANSFER_ASYNC=1` :

```python
if (os.environ.get("VRM_TRANSFER_ASYNC", "0") == "1"
        and stream is not None
        and hasattr(vramancer_rust, "direct_vram_copy_async")):
    # V5 P4 — async path. Caller (this method) is responsible for
    # synchronizing the stream below.
    ok = vramancer_rust.direct_vram_copy_async(
        src_gpu, dst_gpu, tensor_ptr, nbytes,
        stream.cuda_stream,   # u64 CUDA stream handle
    )
    if ok:
        # Return without sync — caller's stream barrier provides ordering.
        return TransportMethod.RUST_P2P, None
    # Fall through to sync path on failure
```

## P4.5 — Bench

```bash
ls benchmarks/ | grep -i overlap
ls benchmarks/bench_p2p_impact.py 2>&1 || echo "ABSENT"
```

Si `bench_p2p_impact.py` existe (V3), réutilise-le :

```bash
unset VRM_TRANSFER_ASYNC VRM_TRANSFER_OVERLAP
python benchmarks/bench_p2p_impact.py > /tmp/bench_v5_p4_off.log 2>&1

VRM_TRANSFER_OVERLAP=1 VRM_TRANSFER_ASYNC=1 \
  python benchmarks/bench_p2p_impact.py > /tmp/bench_v5_p4_on.log 2>&1
```

Compare les latences moyennes. Reporte dans `resultat_v5.md`.

**Critère :** gain ≥ 8% sur transferts > 1 MiB → garde. Sinon → `git revert` du commit P4 et
`[NEGATIVE@P4]`. Ne masque pas un gain nul.

## P4.6 — Commit

```bash
git add rust_core/ core/transfer_manager.py
git diff --cached --stat
git commit -m "[P4.2] expose Rust direct_vram_copy_async behind VRM_TRANSFER_ASYNC=1

V4 P2 found that the existing GpuPipeline.transfer() ignored the Python
torch.cuda.Stream — overlap gain was ~0%. V5 exposes a non-blocking PyO3
variant that accepts a CUDA stream handle and lets the Python caller own
the synchronization.

Default OFF. Enable with VRM_TRANSFER_OVERLAP=1 + VRM_TRANSFER_ASYNC=1.
Bench: <fill avec P4.5>"
```

---

# P5 — Silent exceptions sweep — hot paths (3 modules ciblés)

> Audit V5 a confirmé : 226 `except Exception:` au total dans `core/`. Hot paths critiques :
> `transfer_manager.py` (10 occurrences), `continuous_batcher.py` (11), `hierarchical_memory.py`
> (12). Cible V5 : ces 3 modules. Pas de big bang sur les 226 ailleurs — ça c'est V6.

## P5.1 — Audit ciblé

```bash
grep -n "except Exception" core/transfer_manager.py
grep -n "except Exception" core/continuous_batcher.py
grep -n "except Exception" core/hierarchical_memory.py
```

Note **chaque ligne** dans un fichier scratch. Pour chaque `except`, classe :

- **A** = vraiment silencieux (`pass`, `continue`, retour bool/None sans log)
- **B** = log mais sans `exc_info=True` (perd la stack)
- **C** = déjà bien (log avec `exc_info` ou re-raise)

V5 cible : **A** → migrer vers B avec `exc_info=True` ; **B** → ajouter `exc_info=True`
si manquant ; **C** → laisser.

## P5.2 — Patch `transfer_manager.py` (10 occurrences)

Pour chaque ligne identifiée :

Avant (exemple) :

```python
try:
    self._gpu_pipelines[(src, dst)].transfer(...)
except Exception:
    pass  # silently fall back
```

Après :

```python
try:
    self._gpu_pipelines[(src, dst)].transfer(...)
except Exception as e:
    _logger.debug("Rust GpuPipeline.transfer failed (%s) — falling back", e, exc_info=True)
```

> **RÈGLE D'OR P5 :** ne change JAMAIS le control flow. Si l'`except` faisait `pass`, le
> nouveau bloc fait aussi un fall-through silencieux côté logique — on AJOUTE seulement le
> log. Si l'`except` faisait `return None`, on garde le `return None` après le log.
>
> Le but est l'observabilité, pas la correction de bug. Les vrais bugs cachés sortiront via
> les logs en production, et seront fixés en V6.

Niveau de log :

- `_logger.debug` si l'erreur est attendue (fallback de fonctionnalité optionnelle)
- `_logger.warning` si l'erreur dégrade la perf
- `_logger.error` si l'erreur compromet une opération critique

Si `_logger` n'existe pas dans le module, vérifie en haut du fichier qu'il est défini
(`_logger = logging.getLogger(__name__)`). Si absent, ajoute-le.

## P5.3 — Patch `continuous_batcher.py` (11 occurrences)

Même approche que P5.2. Lignes 307, 380, 390, 430, 444, 488, 489, 570, 601, 682, 803.

## P5.4 — Patch `hierarchical_memory.py` (12 occurrences)

Même approche. Audit ciblé d'abord :

```bash
grep -n "except Exception" core/hierarchical_memory.py
```

## P5.5 — Tests

```bash
source .venv/bin/activate
pytest tests/ -q --tb=line 2>&1 | tail -3
```

**Critère :** aucune régression. Si un test échoue car il assertait sur l'absence de log,
le mettre à jour (le log est plus utile que l'assertion).

## P5.6 — Commit

```bash
git add core/transfer_manager.py core/continuous_batcher.py core/hierarchical_memory.py
git diff --cached --stat
git commit -m "[P5.2-P5.4] add exc_info=True to silent excepts in 3 hot-path modules

Audit V5 confirmed 226 bare 'except Exception:' in core/. V5 sweep targets
the 3 hottest paths: transfer_manager (10), continuous_batcher (11),
hierarchical_memory (12). Control flow unchanged — only observability added.

Remaining ~193 instances deferred to V6."
```

---

# P6 — Hetero advantage : bench reproductible 14B 5070Ti+3090

> V4 P5.3 a montré que vLLM OOM sur Qwen2.5-14B 2-GPU hétérogène (5070 Ti 16GB + 3090 24GB),
> mais VRAMancer y arrive (split VRAM-proportionnel). C'est le **positionnement unique** du
> projet. V5 : on rend ce bench reproductible et on génère un graphique.

## P6.1 — Vérifie le bench V4

```bash
ls benchmarks/ | grep -i hetero
ls benchmarks/ | grep -i 14b
ls bench_*14b* 2>&1
```

S'il existe déjà un script V4 pour ce bench, réutilise-le. Sinon, crée
`benchmarks/bench_hetero_advantage.py` :

```python
"""V5 P6 — reproducible hetero-GPU benchmark.

Runs Qwen2.5-14B-Instruct (or fallback to a smaller model if absent) on:
1. VRAMancer with 5070 Ti (16GB) + 3090 (24GB) — VRAM-proportional split
2. (skipped) vLLM tensor_parallel_size=2 — known to OOM (V4 P5.3)

Outputs:
- benchmarks/bench_hetero_advantage_v5.json (raw)
- benchmarks/bench_hetero_advantage_v5.md (markdown table)
"""
import json
import os
import time
from pathlib import Path

MODEL = os.environ.get("VRM_BENCH_MODEL", "Qwen/Qwen2.5-14B-Instruct")
PROMPT = "Explain the second law of thermodynamics in detail."
MAX_TOKENS = 100
RUNS = 3
OUT_JSON = Path("benchmarks/bench_hetero_advantage_v5.json")
OUT_MD = Path("benchmarks/bench_hetero_advantage_v5.md")


def bench_vramancer():
    from core.inference_pipeline import InferencePipeline
    pipeline = InferencePipeline(model_name=MODEL)  # adapte selon API réelle
    runs = []
    for i in range(RUNS):
        t0 = time.perf_counter()
        out = pipeline.generate(PROMPT, max_new_tokens=MAX_TOKENS)
        dt = time.perf_counter() - t0
        tok_s = MAX_TOKENS / dt
        runs.append({"run": i, "tok_s": tok_s, "dt_s": dt})
        print(f"  run {i}: {tok_s:.2f} tok/s")
    return runs


if __name__ == "__main__":
    print(f"Bench {MODEL} on hetero 2-GPU...")
    runs = bench_vramancer()
    median = sorted(r["tok_s"] for r in runs)[len(runs) // 2]
    result = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "runs": runs,
        "median_tok_s": median,
        "vllm_status": "OOM (V4 P5.3 — tensor_parallel_size=2 on hetero refused)",
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    md = f"""# Hetero advantage bench (V5 P6)

**Model:** `{MODEL}`
**GPUs:** RTX 5070 Ti (16GB) + RTX 3090 (24GB)
**Max tokens:** {MAX_TOKENS} | **Runs:** {RUNS}

| Tool | Median tok/s | Status |
|------|-------------|--------|
| VRAMancer (split VRAM-proportional) | {median:.2f} | OK |
| vLLM 0.20.1 (TP=2) | — | OOM (V4 P5.3) |

**Conclusion:** VRAMancer is the only tool tested that loads {MODEL} on this
hetero 2-GPU setup. vLLM TP assumes homogeneous GPUs and saturates the
smaller GPU.
"""
    OUT_MD.write_text(md)
    print(f"\nWrote {OUT_JSON} and {OUT_MD}")
    print(f"Median: {median:.2f} tok/s")
```

Adapte `InferencePipeline(model_name=MODEL)` à l'API réelle (vérifie `core/inference_pipeline.py`
constructor signature).

## P6.2 — Lance le bench

```bash
source .venv/bin/activate
python benchmarks/bench_hetero_advantage.py 2>&1 | tee /tmp/bench_v5_p6.log
```

Si OOM ou modèle absent → documente `[SKIPPED@P6 — model absent / OOM on dev rig]` et passe.
Le seul critère ici est : **le script doit tourner sans erreur OU échouer proprement avec
un message clair**.

## P6.3 — Commit

```bash
git add benchmarks/bench_hetero_advantage.py benchmarks/bench_hetero_advantage_v5.{json,md}
git diff --cached --stat
git commit -m "[P6.1-P6.2] reproducible hetero-GPU bench (Qwen2.5-14B 5070Ti+3090)

V4 P5.3 showed vLLM OOM on hetero TP. This script makes the VRAMancer-only
result reproducible: bench_hetero_advantage_v5.{json,md}.

Result: <fill avec P6.2 median>"
```

---

# P7 — Example `usb4_distributed_vram.py` — fix ou déprécier

> V4 P11 a ajouté un header `Requires:` à cet example, mais il reste cassé : il importe
> `core.network.packets` qui a été déplacé dans `_deprecated/`. Audit V5 a confirmé que la
> ligne 4 importe `core.network.transmission` (probable remplaçant). Décision : on tente
> une migration **light** ; si ça ne marche pas en 20 min, on déprécie.

## P7.1 — Audit

```bash
cat examples/usb4_distributed_vram.py
ls core/network/
grep -l "send_block\|llm_transport" core/network/
```

Vérifie quel module fournit aujourd'hui l'API utilisée par l'example.

## P7.2 — Décide : fix ou déprécier

**Option A — Fix (si remplaçant trivial)** : remplace les imports cassés par les nouveaux,
lance `python examples/usb4_distributed_vram.py --help` (ou équivalent), confirme que ça
charge sans `ImportError`. Pas besoin que ça tourne end-to-end (le hardware USB4 manque),
juste que l'import passe.

**Option B — Déprécier (si refactor non trivial)** :

```bash
mkdir -p _deprecated/examples
git mv examples/usb4_distributed_vram.py _deprecated/examples/
```

Puis ajoute une note dans `examples/README.md` (ou `EXAMPLES.md` à la racine) :

```markdown
## Deprecated examples

- `usb4_distributed_vram.py` — moved to `_deprecated/examples/`.
  Used `core.network.packets` (now in `_deprecated/`). To revive, port
  to the current network stack in `core/network/` (see `transmission.py`,
  `llm_transport.py`).
```

## P7.3 — Commit

Option A :

```bash
git add examples/usb4_distributed_vram.py
git commit -m "[P7.2] fix usb4_distributed_vram example: migrate to core.network.transmission"
```

Option B :

```bash
git add _deprecated/examples/ examples/ EXAMPLES.md
git commit -m "[P7.2] deprecate usb4_distributed_vram.py — port to current net stack required

V4 P11 marked as broken. V5 audit: import chain requires non-trivial port.
Moved to _deprecated/examples/ with porting notes in EXAMPLES.md."
```

---

# P8 — Repo root cleanup — déplacer `bench_*.{json,log,txt}`

> Audit V5 a compté ~26 fichiers `bench_*.json` + 5 logs/txt à la racine. V4 P14.3 a créé
> `benchmarks/RESULTS_INDEX.md` mais les fichiers sont restés à la racine. V5 : on déplace.

## P8.1 — Liste

```bash
ls -la /home/jeremie/VRAMancer/VRAMancer/bench_*.{json,log,txt} 2>/dev/null | wc -l
ls /home/jeremie/VRAMancer/VRAMancer/bench_*.json 2>/dev/null
ls /home/jeremie/VRAMancer/VRAMancer/bench_*.log 2>/dev/null
ls /home/jeremie/VRAMancer/VRAMancer/bench_*.txt 2>/dev/null
ls /home/jeremie/VRAMancer/VRAMancer/{cache_base,test_results,test_output,sc_source,sl_source}.{txt,log} 2>/dev/null
```

## P8.2 — Déplace

```bash
mkdir -p benchmarks/results
git mv bench_*.json benchmarks/results/ 2>&1 | tail -5
git mv bench_*.log benchmarks/results/ 2>&1 | tail -5
git mv bench_*.txt benchmarks/results/ 2>&1 | tail -5
```

Pour les fichiers misc (logs hors-bench) :

```bash
# Fichiers à supprimer (artefacts de tests jamais commités utilement) :
for f in cache_base.txt test_results.txt test_output.log sc_source.txt sl_source.txt; do
  if [ -f "$f" ]; then
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      git rm "$f"
    else
      rm "$f"
    fi
  fi
done
```

> **ATTENTION :** vérifie d'abord que ces fichiers ne sont pas référencés dans la doc ou les
> tests :
>
> ```bash
> grep -rn "test_results.txt\|test_output.log\|cache_base.txt\|sc_source.txt\|sl_source.txt" \
>   --include="*.py" --include="*.md" --include="*.yml" .
> ```
>
> S'ils sont référencés → ne supprime pas, juste documente.

## P8.3 — Mets à jour `.gitignore`

V4 P14.4 a déjà durci `.gitignore`. Ajoute juste la pattern pour empêcher la récidive :

```bash
grep -q "^bench_.*\.json$" .gitignore || echo "
# V5 P8 — block bench artifacts at repo root (use benchmarks/results/ instead)
/bench_*.json
/bench_*.log
/bench_*.txt
/test_results.txt
/test_output.log
/cache_base.txt
" >> .gitignore
```

## P8.4 — Mets à jour `benchmarks/RESULTS_INDEX.md`

Édite le fichier pour pointer vers `benchmarks/results/` au lieu de la racine :

```bash
sed -i.bak 's|^| - \[`|; s|\.json$|.json`](results/&)|' /tmp/_index_lines.txt 2>/dev/null
# Plus simple : édite manuellement avec ton tool d'édition
```

Vérifie que les chemins sont valides après déplacement.

## P8.5 — Commit

```bash
git status
git add .gitignore benchmarks/
git commit -m "[P8.2-P8.4] move bench_*.{json,log,txt} from repo root to benchmarks/results/

26 bench JSON/log/txt files cluttering repo root, moved to benchmarks/results/.
.gitignore hardened to prevent recidive. Cleaned 5 misc artifact files."
```

---

# P9 — TODO markers : adresser les 3 ouverts ou documenter

> Audit V5 a trouvé 3 markers TODO/FIXME ouverts dans `core/`. Pas urgent, mais V5 les passe
> en revue : soit fix trivial, soit migration vers TECHNICAL_DEBT.md.

## P9.1 — Liste

```bash
grep -rn "TODO\|FIXME\|XXX" core/ rust_core/ csrc/ \
  --include="*.py" --include="*.rs" --include="*.cpp" --include="*.c" 2>/dev/null \
  | grep -v __pycache__
```

Confirme les 3 markers d'audit V5 :

1. `core/turbo_engine.py:202` — "Phase 2 (TODO): StaticKVCache + CUDA Graph capture."
2. `core/cross_vendor_bridge.py:166` — drm_render_node placeholder
3. `core/cross_vendor_bridge.py:356` — DRM device check

## P9.2 — Décide cas par cas

**TODO 1 (turbo_engine StaticKVCache + CUDA Graph) :** trop gros pour V5. Migrer vers
TECHNICAL_DEBT.md comme nouvel item :

```markdown
| TURBO_KV_CUDAGRAPH | core/turbo_engine.py:202 | Phase 2 non implémenté: StaticKVCache + CUDA Graph capture | Gros | Moyenne — gain perf single-GPU significatif si fait |
```

Remplace dans le code :

```python
# Phase 2 (TODO): StaticKVCache + CUDA Graph capture.
```

par :

```python
# Phase 2 (deferred): StaticKVCache + CUDA Graph capture.
# See TECHNICAL_DEBT.md → TURBO_KV_CUDAGRAPH.
```

**TODO 2 et 3 (DRM render node) :** documentation, pas un vrai TODO. Convertir en commentaire
narratif sans le tag TODO :

```python
# drm_render_node: e.g. "/dev/dri/renderD128"  — populated by detect_drm_node()
```

## P9.3 — Commit

```bash
git add core/turbo_engine.py core/cross_vendor_bridge.py docs/reports/TECHNICAL_DEBT.md
git diff --cached --stat
git commit -m "[P9.2] resolve open TODO markers: 1 migrated to TECHNICAL_DEBT, 2 reworded"
```

---

# P10 — Tests coverage : auto-start batcher + label consistency

## P10.1 — Test auto-start ContinuousBatcher

**Fichier nouveau :** `tests/test_continuous_batcher_autostart.py`

```python
"""V5 P10.1 — verify ContinuousBatcher auto-starts when VRM_CONTINUOUS_BATCHING=1.

Regression guard for V4 P4 diagnosis (auto-start was missing).
"""
import os
import pytest


@pytest.fixture
def env_continuous_batching(monkeypatch):
    monkeypatch.setenv("VRM_CONTINUOUS_BATCHING", "1")
    monkeypatch.setenv("VRM_BACKEND_ALLOW_STUB", "1")


def test_batcher_autostarts_in_generate(env_continuous_batching):
    """Once VRM_CONTINUOUS_BATCHING=1, generate() must start the batcher."""
    pytest.importorskip("torch")
    from core.inference_pipeline import InferencePipeline

    # Use stub backend — we don't need a real model, only to check the batcher
    # gets started.
    try:
        pipe = InferencePipeline(backend_name="stub")  # adapte à l'API réelle
    except Exception as e:
        pytest.skip(f"InferencePipeline init failed (stub backend missing?): {e}")

    if pipe.continuous_batcher is None:
        pytest.skip("Batcher not initialized (probably no torch CUDA in test env)")

    assert pipe.continuous_batcher._running is False, "should be idle pre-generate"

    try:
        pipe.generate("hello", max_new_tokens=4)
    except Exception:
        # We don't care if generation succeeds — only that start() was called.
        pass

    assert pipe.continuous_batcher._running, \
        "VRM_CONTINUOUS_BATCHING=1 should auto-start batcher (V5 P1 fix)"

    # Cleanup
    if hasattr(pipe.continuous_batcher, "stop"):
        pipe.continuous_batcher.stop()


def test_batcher_no_autostart_without_flag(monkeypatch):
    """Without the flag, generate() must NOT auto-start the batcher (V4 behavior)."""
    monkeypatch.delenv("VRM_CONTINUOUS_BATCHING", raising=False)
    monkeypatch.setenv("VRM_BACKEND_ALLOW_STUB", "1")
    pytest.importorskip("torch")
    from core.inference_pipeline import InferencePipeline

    try:
        pipe = InferencePipeline(backend_name="stub")
    except Exception as e:
        pytest.skip(f"InferencePipeline init failed: {e}")

    if pipe.continuous_batcher is None:
        pytest.skip("Batcher not initialized")

    try:
        pipe.generate("hello", max_new_tokens=4)
    except Exception:
        pass

    assert pipe.continuous_batcher._running is False, \
        "without VRM_CONTINUOUS_BATCHING=1, batcher must stay idle"
```

> Adapte `backend_name="stub"` au mécanisme réel (peut-être `model_name="stub"` ou un mock).
> Si tu ne trouves pas comment instancier sans modèle réel, **mets un `pytest.skip`** propre
> en début de test plutôt que de fabriquer un faux qui ne teste rien.

## P10.2 — Test label `RUST_P2P`

**Fichier nouveau :** `tests/test_transfer_method_label.py`

```python
"""V5 P10.2 — _get_method_for() returns RUST_P2P when GpuPipeline cached."""
import pytest


def test_method_for_returns_rust_p2p_when_gpu_pipeline_cached():
    pytest.importorskip("torch")
    from core.transfer_manager import TransferManager

    tm = TransferManager()
    if not hasattr(tm, "_gpu_pipelines"):
        pytest.skip("TransferManager has no _gpu_pipelines attribute")

    # Inject a sentinel into the pipeline cache to simulate a cached pair.
    tm._gpu_pipelines[(0, 1)] = object()
    assert tm._get_method_for(0, 1) == "RUST_P2P"


def test_method_for_falls_back_to_existing_labels():
    pytest.importorskip("torch")
    from core.transfer_manager import TransferManager

    tm = TransferManager()
    # No pipeline cached for this pair
    label = tm._get_method_for(2, 3)
    assert label in {"CUDA_P2P", "NCCL", "CPU_STAGED"}
```

## P10.3 — Lance la suite complète

```bash
source .venv/bin/activate
pytest tests/test_continuous_batcher_autostart.py tests/test_transfer_method_label.py -v
pytest tests/ -q --tb=line 2>&1 | tail -3
```

**Critère :** la baseline doit passer de 1074 → ≥ 1076 passed (2 nouveaux tests). Si les
nouveaux tests skippent en CI minimal, c'est OK.

## P10.4 — Commit

```bash
git add tests/test_continuous_batcher_autostart.py tests/test_transfer_method_label.py
git commit -m "[P10.1+P10.2] tests: ContinuousBatcher auto-start + RUST_P2P label

Regression guards for V5 P1 (auto-start fix) and V5 P2 (honest label)."
```

---

# P11 — Documentation : `TECHNICAL_DEBT` + `CHANGELOG` → 1.6.0

## P11.1 — Refresh `TECHNICAL_DEBT.md`

Édite `docs/reports/TECHNICAL_DEBT.md` :

1. Section "Limitations connues" : marque `CONTINUOUS_BATCHER_GENERATE_BYPASS` comme
   ✅ résolu V5 P1, déplace vers "Stubs résolus".
2. Section "Limitations connues" : marque le label `_get_method_for() CPU_STAGED` comme
   ✅ résolu V5 P2 (déjà fait en P2.3, vérifie cohérence).
3. Ajoute si P9.2 t'a fait ajouter `TURBO_KV_CUDAGRAPH`.
4. Met à jour la date "Dernière mise à jour" en haut → `2026-05 (V5 plan execution)`.

## P11.2 — Bump version 1.5.0 → 1.6.0

**Fichier :** `core/__init__.py` (ou `vramancer/__init__.py` selon ce qui définit `__version__`).

```bash
grep -rn "__version__" core/ vramancer/ pyproject.toml | head -5
```

Édite la ligne `__version__ = "1.5.0"` → `__version__ = "1.6.0"`.

## P11.3 — `CHANGELOG.md` : Unreleased → 1.6.0

Édite `CHANGELOG.md` :

Avant :

```markdown
## [Unreleased] — V4 plan (sera 1.5.1 ou 1.6.0)
```

Après :

```markdown
## [Unreleased] — V5 plan suite

(à remplir au fur et à mesure de V5)

## [1.6.0] — 2026-05-XX — V4 + V5 release

### Added (V5)
- Auto-start `ContinuousBatcher` quand `VRM_CONTINUOUS_BATCHING=1` (P1)
- Label `RUST_P2P` honnête dans `TransferManager._get_method_for()` (P2)
- `use_cache=True` explicite dans `HuggingFaceBackend.generate()` single-GPU (P3)
- Flag `VRM_TRANSFER_ASYNC=1` + PyO3 `direct_vram_copy_async` (P4) — si P4 livré
- `benchmarks/bench_hetero_advantage.py` reproducible (P6)
- Tests régression : auto-start batcher + label `RUST_P2P` (P10)
- `/api/models/load` endpoint + bouton "Charger" dans browser HF (P12)
- Flag `VRM_KV_OFFLOAD_ENGRAM=1` + `VRM_KV_DRAM_LIMIT_GB` pour KV offload massif (P13)
- `benchmarks/bench_deepseek_engram.py` (P13)

### Fixed (V5)
- 33 silent excepts en hot path migrés vers `_logger.* exc_info=True` (P5)
- `usb4_distributed_vram` example : migré ou déprécié proprement (P7)
- 26 `bench_*.{json,log,txt}` à la racine déplacés vers `benchmarks/results/` (P8)
- 3 TODO markers ouverts résolus ou migrés (P9)

### Changed (V5)
- `TECHNICAL_DEBT.md` rafraîchi (V5 résolutions)
- Version 1.5.0 → 1.6.0

### Added (V4 — récap, voir resultat_v4.md)
- `VRM_TRANSFER_OVERLAP=1` flag (P2)
- `VRM_DEBUG_SAMPLING` flag (P3.1)
- vLLM 0.20.1 installation + comparaison (P5)
- Tests transfer_manager + triton_sampling (P8)
- CI: `VRM_BACKEND_ALLOW_STUB=1` + lint job (P9)
- `/api/dashboard/gpus` alias (P10)
- `benchmarks/RESULTS_INDEX.md` (P14.3)

### Fixed (V4)
- 4 corrections d'honnêteté V3 (GPU ordering, attribution +382%, méthodologie, label) (P1)
- ContinuousBatcher diagnostic (P4) — bug confirmé, fix livré en V5 P1
- Stubs formalisés : VTP_L3, DMABUF, NAT (P6)

[autres entrées V4 si pertinentes]
```

> Ne supprime pas les entrées V4 de la section Unreleased — elles sont juste promues dans la
> nouvelle section `[1.6.0]`. Adapte selon ce qui existe déjà.

## P11.4 — Commit

```bash
git add docs/reports/TECHNICAL_DEBT.md CHANGELOG.md core/__init__.py
git diff --cached --stat
git commit -m "[P11] docs: TECHNICAL_DEBT V5 refresh + CHANGELOG 1.6.0 + version bump"
```

---

# P12 — HF Browser : choix de n'importe quel modèle HuggingFace

> **Constat audit V5 :** un endpoint `/browser` existe déjà dans
> `dashboard/dashboard_web.py:215-216` avec une page `templates/browser.html` qui appelle
> `/api/models/search` (recherche HuggingFace + Ollama). **MAIS** : il n'y a pas (ou pas
> encore) d'action "Charger ce modèle" qui pousse l'ID HF vers le pipeline VRAMancer pour
> chargement effectif sur les GPUs.
>
> **Objectif V5 P12 :** câbler le bouton "Load" du browser pour qu'un user puisse, depuis
> son navigateur, taper "deepseek-v4-flash" (ou n'importe quel ID HF), cliquer Charger, et
> voir le modèle apparaître dans `/api/dashboard/gpus` après split sur les GPUs disponibles.
>
> Cette phase prépare P13 : on aura besoin du browser pour découvrir l'ID HF exact de
> DeepSeek V4 Flash.

## P12.1 — Audit de l'existant (10 min)

```bash
sed -n '210,260p' dashboard/dashboard_web.py
cat dashboard/templates/browser.html | head -80
grep -n "load_model\|load.*hf\|load_hf\|/api/models/load\|/api/pipeline/load" \
  dashboard/dashboard_web.py core/inference_pipeline.py | head -10
```

Identifie :

1. La route actuelle qui RECHERCHE (`/api/models/search`).
2. Si une route LOAD existe déjà (`/api/models/load` ou similaire).
3. La méthode du `InferencePipeline` qui charge un modèle par ID HF (cherche
   `def load`, `def load_model`, `from_pretrained`, `model_name=`).

Reporte la liste des routes existantes dans `resultat_v5.md` section P12.

## P12.2 — Ajoute la route `/api/models/load` côté backend

**Fichier :** `dashboard/dashboard_web.py`

Ajoute (à côté de `/api/models/search`) :

```python
@app.route("/api/models/load", methods=["POST"])
def api_models_load():
    """Load a HuggingFace model into the active InferencePipeline.

    POST body: {"model_id": "deepseek-ai/deepseek-v4-flash", "source": "hf"}
    Returns: {"ok": bool, "msg": str, "model_id": str, "device_map": dict}
    """
    from flask import request, jsonify
    payload = request.get_json(silent=True) or {}
    model_id = payload.get("model_id", "").strip()
    source = payload.get("source", "hf").strip().lower()
    if not model_id:
        return jsonify({"ok": False, "msg": "model_id required"}), 400
    if source != "hf":
        return jsonify({"ok": False, "msg": f"source '{source}' not supported yet"}), 400

    # Acquire pipeline singleton — adapte selon l'API réelle.
    try:
        from core.inference_pipeline import InferencePipeline
        # Cherche un singleton existant ou instancie. Si l'app Flask garde une
        # référence (par ex. dans `app.config["pipeline"]`), utilise-la pour
        # éviter de spawner un nouveau pipeline et doubler la VRAM.
        pipeline = app.config.get("pipeline")
        if pipeline is None:
            pipeline = InferencePipeline(model_name=model_id)
            app.config["pipeline"] = pipeline
        else:
            # Reload model into existing pipeline if API supports it
            if hasattr(pipeline, "load_model"):
                pipeline.load_model(model_id)
            else:
                # Fallback: replace the instance
                pipeline = InferencePipeline(model_name=model_id)
                app.config["pipeline"] = pipeline
    except Exception as e:
        logger.error("Failed to load model %s: %s", model_id, e, exc_info=True)
        return jsonify({"ok": False, "msg": f"load failed: {e}"}), 500

    device_map = getattr(getattr(pipeline, "backend", None), "hf_device_map", None) or {}
    return jsonify({
        "ok": True,
        "msg": f"Loaded {model_id}",
        "model_id": model_id,
        "device_map": dict(device_map) if hasattr(device_map, 'items') else {},
    })
```

> **CAVEAT :** l'API exacte d'`InferencePipeline` peut différer (constructeur, nom du kwarg,
> méthode `load_model`). Adapte au code réel — vérifie `core/inference_pipeline.py`
> `__init__` et `load_*` methods.

## P12.3 — Ajoute le bouton "Charger" dans `browser.html`

**Fichier :** `dashboard/templates/browser.html`

Trouve le `<tbody id="resultsBody">` et la fonction JS qui peuple les lignes. Ajoute pour
chaque ligne un bouton dans la colonne "Action" :

```html
<button onclick="loadModel('${model.id}', '${model.source}')">Charger</button>
```

Et la fonction JS associée (dans le `<script>` en bas) :

```javascript
async function loadModel(modelId, source) {
    const status = document.getElementById('status');
    status.textContent = `Chargement de ${modelId}...`;
    status.style.color = '#ffcc00';
    try {
        const resp = await fetch('/api/models/load', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({model_id: modelId, source: source})
        });
        const data = await resp.json();
        if (data.ok) {
            status.textContent = `✅ ${data.msg} — device_map: ${JSON.stringify(data.device_map)}`;
            status.style.color = '#00ff88';
        } else {
            status.textContent = `❌ ${data.msg}`;
            status.style.color = '#ff5555';
        }
    } catch (e) {
        status.textContent = `❌ Erreur réseau: ${e}`;
        status.style.color = '#ff5555';
    }
}
```

> Si la fonction de rendu de table existante utilise `innerHTML += "<tr>..."`, adapte le
> template HTML là où la balise `<button>Charger</button>` doit apparaître. Si la table est
> rendue par une autre méthode (DOM API), insère le bouton via `createElement('button')`.

## P12.4 — Smoke test bout en bout

```bash
source .venv/bin/activate

# Lance le dashboard en background
VRM_BACKEND_ALLOW_STUB=1 python dashboard/dashboard_web.py > /tmp/dashboard_v5.log 2>&1 &
DASHBOARD_PID=$!
sleep 3

# Test search
curl -s "http://localhost:5002/api/models/search?q=tinyllama" | head -c 500
echo

# Test load avec un PETIT modèle d'abord
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"model_id":"sshleifer/tiny-gpt2","source":"hf"}' \
  http://localhost:5002/api/models/load

# Vérifie /api/dashboard/gpus reflète le chargement
curl -s http://localhost:5002/api/dashboard/gpus | head -c 500
echo

# Cleanup
kill $DASHBOARD_PID 2>/dev/null
```

> Si le port 5002 n'est pas le bon, vérifie dans `dashboard/dashboard_web.py` la valeur de
> `app.run(port=...)` ou la variable d'environnement `VRM_DASHBOARD_PORT`.

Reporte dans `resultat_v5.md` :

```markdown
**P12.4 Smoke browser HF loader :**
- /api/models/search?q=tinyllama → status 200, N résultats HF
- /api/models/load tiny-gpt2 → status 200, ok=true
- /api/dashboard/gpus après load → device_count=N, model visible
```

## P12.5 — Visual smoke (optionnel mais recommandé)

Si tu as un environnement où `xdg-open` ou un navigateur fonctionne :

```bash
# Lance le dashboard
VRM_BACKEND_ALLOW_STUB=1 python dashboard/dashboard_web.py &
sleep 2
xdg-open http://localhost:5002/browser 2>/dev/null || \
  echo "Open manually: http://localhost:5002/browser"
```

Vérifie visuellement :
1. La barre de recherche fonctionne (tape "qwen", appuie Entrée).
2. Les résultats HF apparaissent.
3. Le bouton "Charger" est cliquable et déclenche le status-bar avec un message.

Si tu ne peux pas tester visuellement (CI, headless), c'est OK — le test curl P12.4 suffit.

## P12.6 — Commit

```bash
git add dashboard/dashboard_web.py dashboard/templates/browser.html
git diff --cached --stat
git commit -m "[P12] HF browser model loader: end-to-end /api/models/load + UI button

Audit V5: /browser route existed since V3 with HF/Ollama search but no
'Load' action wired. V5 adds POST /api/models/load that instantiates (or
hot-swaps) the active InferencePipeline with the chosen model_id, and a
'Charger' button per result row.

Smoke: tiny-gpt2 loaded successfully, device_map populated."
```

---

# P13 — DeepSeek V4 Flash dual-GPU + engram KV offload (jusqu'à 200GB RAM)

> **Démonstration cible :** charger DeepSeek V4 Flash sur le rig dual-GPU (RTX 5070 Ti 16GB +
> RTX 3090 24GB), avec contexte long qui dépasse la VRAM disponible. Le KV cache excédentaire
> est offloadé vers la RAM serveur (jusqu'à 200 GB) via le mécanisme `engram` (parity_memory)
> + `PagedAttentionOffloader`.
>
> C'est le cas d'usage **flagship** de VRAMancer : faire tourner un modèle + contexte qui ne
> tiennent JAMAIS dans la VRAM, en utilisant la hiérarchie mémoire VRAM → DRAM avec
> protection engram.

## P13.1 — Découvre l'ID HF exact de DeepSeek V4 Flash

> **L'utilisateur a dit "DeepSeek V4 Flash". Le nom exact peut être**
> `deepseek-ai/DeepSeek-V3` , `deepseek-ai/DeepSeek-V2.5-1210`, `deepseek-ai/DeepSeek-V3-Flash`,
> ou un autre. **Ne devine pas. Découvre via le browser HF de P12.**

Lance le dashboard et utilise le browser pour chercher "deepseek" :

```bash
source .venv/bin/activate
VRM_BACKEND_ALLOW_STUB=1 python dashboard/dashboard_web.py > /tmp/dashboard_v5_p13.log 2>&1 &
DASHBOARD_PID=$!
sleep 3

# Recherche tous les DeepSeek
curl -s "http://localhost:5002/api/models/search?q=deepseek" | python -m json.tool | head -80

kill $DASHBOARD_PID 2>/dev/null
```

Identifie le modèle qui correspond à "DeepSeek V4 Flash" (le plus récent / contenant "flash"
dans le nom, sinon le V3 le plus récent). Note son ID HF exact dans `resultat_v5.md` section
P13 :

```markdown
**P13.1 Modèle cible identifié :**
- ID HF : <ID exact, par ex. deepseek-ai/DeepSeek-V3>
- Taille : XXB params
- Quantization disponible : <BF16/FP8/Q4/etc>
- VRAM minimale doc : <X GB>
```

> **Si le modèle est trop gros pour le rig (e.g. 671B params)** : skip ce point avec
> `[SKIPPED@P13.1 — model too large for 16+24=40 GB VRAM rig even with offload]` et
> propose une alternative dans le markdown : DeepSeek-Coder-V2-Lite, DeepSeek-V2-Lite, ou
> Qwen2.5-72B-Instruct (déjà testé en V4 P5.3 sans hetero).
>
> Le but pédagogique de P13 est de démontrer le mécanisme engram, pas de battre un record.
> Adapte le modèle pour qu'il soit chargeable.

## P13.2 — Configure l'offload engram

**Fichier :** `core/inference_pipeline.py` (ou un nouveau fichier de config selon
l'architecture)

Vérifie que le wiring entre `PagedAttentionOffloader` et `HierarchicalMemoryManager` existe
réellement :

```bash
grep -n "PagedAttentionOffloader\|paged_attention_offload" core/ -r | head -10
grep -n "store_engram\|heal_engram\|active_engrams" core/ -r | head -10
```

Si l'offloader n'est pas câblé automatiquement quand le KV cache est plein, ajoute (ou
expose) un flag :

```python
# Dans le code d'init du pipeline (probablement vers ligne 1380-1400 où PagedKV s'initialise)
if os.environ.get("VRM_KV_OFFLOAD_ENGRAM", "0") == "1":
    from core.paged_attention_offload import PagedAttentionOffloader
    from core.hierarchical_memory import HierarchicalMemoryManager
    from core.parity_memory import ParityMemoryHealer  # ajuste au nom réel
    hmm = HierarchicalMemoryManager(
        max_dram_gb=int(os.environ.get("VRM_KV_DRAM_LIMIT_GB", "200")),
        # autres kwargs selon API
    )
    healer = ParityMemoryHealer()  # engram parity protection
    self.kv_offloader = PagedAttentionOffloader(self.paged_kv, hmm)
    self.kv_engram = healer
    _logger.info(
        "KV engram offload enabled: DRAM limit=%d GB, parity protection ON",
        int(os.environ.get("VRM_KV_DRAM_LIMIT_GB", "200")),
    )
```

> Le code ci-dessus est un **squelette indicatif**. Vérifie les vrais constructeurs de
> `HierarchicalMemoryManager` et `ParityMemoryHealer`/`ParityMemory` (le nom exact est dans
> `core/parity_memory.py`). Si l'API existante demande des kwargs différents, adapte. Si
> l'intégration est trop complexe pour V5, **skip P13.2 et passe à P13.3 avec ce qui existe
> déjà** — l'objectif est de démontrer le mécanisme, même partiellement.

## P13.3 — Crée le bench `bench_deepseek_engram.py`

**Fichier nouveau :** `benchmarks/bench_deepseek_engram.py`

```python
"""V5 P13 — DeepSeek + engram KV offload bench.

Loads the chosen DeepSeek model on hetero dual-GPU and runs increasing
context lengths. Measures:
- Tokens/sec per context size
- Peak VRAM per GPU
- DRAM used by engram offload
- Threshold context size where offload kicks in
"""
import json
import os
import time
from pathlib import Path

MODEL = os.environ.get("VRM_BENCH_MODEL", "deepseek-ai/DeepSeek-V3")  # adapte P13.1
CONTEXT_SIZES = [1024, 4096, 16384, 65536, 131072]  # tokens
MAX_NEW = 64
OUT_JSON = Path("benchmarks/bench_deepseek_engram_v5.json")
OUT_MD = Path("benchmarks/bench_deepseek_engram_v5.md")


def measure_vram_per_gpu():
    import torch
    out = {}
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        out[f"gpu{i}"] = {
            "used_mb": (total - free) // (1024 * 1024),
            "total_mb": total // (1024 * 1024),
        }
    return out


def measure_dram_used():
    """Best-effort DRAM measurement via psutil if available."""
    try:
        import psutil
        proc = psutil.Process()
        return {"rss_mb": proc.memory_info().rss // (1024 * 1024)}
    except Exception:
        return {"rss_mb": -1}


def make_prompt(num_tokens, tokenizer):
    """Generate a synthetic long prompt of ~num_tokens tokens."""
    base = "The quick brown fox jumps over the lazy dog. " * 50
    text = base
    while len(tokenizer.encode(text)) < num_tokens:
        text += base
    return text


def run_bench():
    from core.inference_pipeline import InferencePipeline
    print(f"Loading {MODEL} with engram KV offload (VRM_KV_OFFLOAD_ENGRAM=1)...")
    os.environ["VRM_KV_OFFLOAD_ENGRAM"] = "1"
    os.environ.setdefault("VRM_KV_DRAM_LIMIT_GB", "200")
    pipeline = InferencePipeline(model_name=MODEL)  # adapte API
    tokenizer = pipeline.backend.tokenizer

    results = []
    for ctx_size in CONTEXT_SIZES:
        prompt = make_prompt(ctx_size, tokenizer)
        actual_input_tokens = len(tokenizer.encode(prompt))
        vram_before = measure_vram_per_gpu()
        dram_before = measure_dram_used()
        t0 = time.perf_counter()
        try:
            out = pipeline.generate(prompt, max_new_tokens=MAX_NEW)
        except Exception as e:
            results.append({
                "ctx_target": ctx_size,
                "ctx_actual": actual_input_tokens,
                "error": str(e)[:200],
            })
            print(f"  ctx={ctx_size}: ERROR {e}")
            continue
        dt = time.perf_counter() - t0
        vram_after = measure_vram_per_gpu()
        dram_after = measure_dram_used()
        tok_s = MAX_NEW / dt if dt > 0 else 0
        results.append({
            "ctx_target": ctx_size,
            "ctx_actual": actual_input_tokens,
            "max_new": MAX_NEW,
            "dt_s": dt,
            "tok_s": tok_s,
            "vram_before": vram_before,
            "vram_after": vram_after,
            "dram_before_mb": dram_before["rss_mb"],
            "dram_after_mb": dram_after["rss_mb"],
            "dram_delta_mb": dram_after["rss_mb"] - dram_before["rss_mb"],
        })
        print(f"  ctx={ctx_size}: {tok_s:.2f} tok/s, "
              f"VRAM delta={[(k, v['used_mb']-vram_before[k]['used_mb']) for k,v in vram_after.items()]}, "
              f"DRAM delta={results[-1]['dram_delta_mb']} MB")

    OUT_JSON.write_text(json.dumps({
        "model": MODEL,
        "context_sizes": CONTEXT_SIZES,
        "max_new": MAX_NEW,
        "results": results,
    }, indent=2))

    md = ["# DeepSeek + engram KV offload bench (V5 P13)", ""]
    md.append(f"**Model:** `{MODEL}`")
    md.append(f"**GPUs:** RTX 5070 Ti (16GB) + RTX 3090 (24GB)")
    md.append(f"**KV offload:** engram parity → DRAM (cap 200 GB)")
    md.append("")
    md.append("| Context | Actual tok | tok/s | VRAM Δ GPU0 | VRAM Δ GPU1 | DRAM Δ MB |")
    md.append("|---------|-----------|-------|-------------|-------------|-----------|")
    for r in results:
        if "error" in r:
            md.append(f"| {r['ctx_target']} | {r['ctx_actual']} | ERROR | — | — | — |")
            continue
        v0 = r['vram_after'].get('gpu0', {}).get('used_mb', 0) - r['vram_before'].get('gpu0', {}).get('used_mb', 0)
        v1 = r['vram_after'].get('gpu1', {}).get('used_mb', 0) - r['vram_before'].get('gpu1', {}).get('used_mb', 0)
        md.append(f"| {r['ctx_target']} | {r['ctx_actual']} | {r['tok_s']:.2f} | {v0} MB | {v1} MB | {r['dram_delta_mb']} |")
    md.append("")
    md.append("**Engram threshold:** context size where DRAM Δ becomes > 0 (cold pages "
              "evicted from KV cache to RAM). Look at the table for the inflection point.")
    OUT_MD.write_text("\n".join(md))
    print(f"\nWrote {OUT_JSON}, {OUT_MD}")


if __name__ == "__main__":
    run_bench()
```

## P13.4 — Lance le bench

```bash
source .venv/bin/activate

# Vérifie d'abord la RAM disponible
free -h | head -3

# Lance avec timeout généreux (peut prendre 30+ min selon contexte)
VRM_KV_OFFLOAD_ENGRAM=1 \
VRM_KV_DRAM_LIMIT_GB=200 \
VRM_BENCH_MODEL="<ID HF de P13.1>" \
  timeout 3600 python benchmarks/bench_deepseek_engram.py 2>&1 | tee /tmp/bench_v5_p13.log
```

Surveille en parallèle dans un autre terminal :

```bash
watch -n 2 'nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; \
  echo; free -h | head -2'
```

## P13.5 — Documente honnêtement

Reporte dans `resultat_v5.md` la table complète + analyse :

```markdown
**P13 Résultats DeepSeek + engram KV offload :**

Modèle : <ID HF>
GPUs : 5070 Ti (16GB) + 3090 (24GB) — total 40 GB VRAM
RAM serveur : <X> GB disponible, cap 200 GB pour engram

| Context | Actual tok | tok/s | VRAM Δ G0 | VRAM Δ G1 | DRAM Δ MB | Notes              |
|---------|-----------|-------|-----------|-----------|-----------|--------------------|
| 1024    | XXX       | XX    | XX        | XX        | 0         | tout en VRAM       |
| 4096    | XXX       | XX    | XX        | XX        | 0         | tout en VRAM       |
| 16384   | XXX       | XX    | XX        | XX        | XXX       | engram kick-in     |
| 65536   | XXX       | XX    | XX        | XX        | XXXX      | engram dominant    |
| 131072  | XXX       | XX    | XX        | XX        | XXXXX     | engram saturation  |

**Inflection point engram :** ~XXk tokens (premier ctx où DRAM Δ > 0)
**Dégradation tok/s :** XX → XX (−Y%) entre 1k et 128k contexte
**Conclusion :** <honnête — gain de fonctionnalité (peut tourner) vs perf (ralenti)>
```

**Critères de "succès" P13 :**
- Le modèle se charge sans OOM ➜ ✅
- Au moins UN context size > taille VRAM totale produit du throughput non-nul ➜ ✅
- DRAM delta > 0 sur au moins un context size (preuve que l'engram offload est actif) ➜ ✅

**Si AUCUN des 3 critères n'est rempli** (ex: modèle ne charge pas, OOM dès 1k contexte) :
écris `[PARTIAL@P13]` ou `[BLOCKED@P13]` selon la cause, document précisément ce qui a
manqué (modèle absent, RAM insuffisante, engram pas câblé, etc.). Ne masque pas l'échec.

## P13.6 — Commit

```bash
git add benchmarks/bench_deepseek_engram.py \
        benchmarks/bench_deepseek_engram_v5.{json,md} \
        core/inference_pipeline.py
git diff --cached --stat
git commit -m "[P13] DeepSeek dual-GPU + engram KV offload bench (up to 200GB DRAM)

Demonstrates the flagship VRAMancer use case: model + context that does not
fit in 40 GB combined VRAM (5070 Ti + 3090), with overflow KV pages evicted
to host DRAM via PagedAttentionOffloader + parity_memory engram protection.

Flag: VRM_KV_OFFLOAD_ENGRAM=1, VRM_KV_DRAM_LIMIT_GB=200.
Model: <ID HF P13.1>
Engram inflection: ~XXk tokens (see bench_deepseek_engram_v5.md)"
```

---

# P14 — Validation finale (tests + smoke + sanity)

```bash
source .venv/bin/activate

# Suite complète
pytest tests/ -q --tb=line 2>&1 | tee /tmp/v5_final_pytest.log | tail -10

# Smoke
python tests/smoke.py 2>&1 | tee /tmp/v5_final_smoke.log | tail -5

# Sanity import
python -c "
import core
print('version:', core.__version__)
from core.inference_pipeline import InferencePipeline
from core.transfer_manager import TransferManager, TransportMethod
print('TransportMethod members:', [m.name for m in TransportMethod])
assert 'RUST_P2P' in [m.name for m in TransportMethod], 'P2 regression'
print('OK')
"
```

**Critères de succès :**

- Tests : aucune régression vs baseline P0.3 (≥ 1074 passed, le test pré-existant
  `test_health_imports_fault_manager` peut continuer à failer).
- Smoke : exit 0.
- Sanity : `RUST_P2P` présent dans l'enum, version = 1.6.0.

Reporte dans `resultat_v5.md` :

```markdown
## [P12] — Validation finale

**Tests :** N failed, M passed, K skipped (vs baseline N=1, M=1074, K=42)
**Smoke :** OK exit 0
**Sanity :** TransportMethod includes RUST_P2P ✅, version=1.6.0 ✅
**Régression :** AUCUNE
```

```bash
git commit --allow-empty -m "[P14] validation finale V5: <résumé>"
```

---

# P15 — `resultat_v5.md` SUMMARY + verdict

Édite la dernière section `## [SUMMARY]` :

```markdown
## [SUMMARY]

**Date fin :** YYYY-MM-DD
**Branche :** chore/sonnet-plan-v5
**Commits V5 :** N commits sur chore/sonnet-plan-v5

```
<copier git --no-pager log --oneline chore/sonnet-plan-v4..HEAD>
```

**Tests :**
- Baseline : 1 failed, 1074 passed, 42 skipped
- Final :    1 failed, M passed, K skipped
- Régression : AUCUNE

**Performance :**
- ContinuousBatcher auto-start (P1) : N=4 concurrent ≈ X tok/s (vs Y tok/s V4 = +Z%)
- vLLM gap (P3) : Qwen-7B 1-GPU 27.5 → X tok/s (Δ +Y%)
- PyO3 transfer_async (P4) : <gain ou skip>
- DeepSeek + engram (P13) : ctx 128k → X tok/s, DRAM peak Y GB

**Fixes structurels :**
- TransportMethod label honnête `RUST_P2P` (P2)
- 33 silent excepts en hot paths migrés vers logs informatifs (P5)
- usb4_distributed_vram : migré ou déprécié (P7)

**Nouvelles capacités UX :**
- Browser HF de modèles entièrement fonctionnel (recherche + chargement) (P12)
- KV cache offload massif RAM (200 GB cap) avec engram parity protection (P13)

**Hygiène :**
- 26 bench_*.json/log/txt déplacés vers benchmarks/results/ (P8)
- 3 TODO markers résolus ou documentés (P9)

**Documentation :**
- TECHNICAL_DEBT V5 refresh
- CHANGELOG 1.6.0 release section
- Version bump 1.5.0 → 1.6.0

**Verdict global V5 : SUCCESS / PARTIAL / FAILED** (choisis honnêtement)

**Reste à faire (V6 candidat) :**
- ~193 silent excepts restants hors hot paths
- TURBO_KV_CUDAGRAPH (Phase 2 turbo_engine)
- vLLM gap structurel : worker loop C++ ou refonte décode CUDA Graphs
- Refactor `core/transfer_manager.py` (>1000 lignes — split en sous-modules)
- VRAM Lending : papier OSDI à rédiger (cf. ANALYSE_ARCHITECTE_V5 Sprint D)
```

```bash
git add resultat_v5.md
git commit -m "[P15] resultat_v5.md SUMMARY + verdict V5"
```

---

# Annexe A — Quick reference : règles de bail-out

| Situation                                          | Action                                  |
|----------------------------------------------------|-----------------------------------------|
| Test échoue 2 fois après debug                     | `[BLOCKED@P<x>.<y>]` + skip phase       |
| Bench montre régression (gain négatif)             | `git revert HEAD` + `[NEGATIVE@P<x>]`   |
| Build Rust échoue (P4)                             | `[SKIPPED@P4]` + passe                  |
| Modèle absent localement                           | `[SKIPPED@P<x>.<y> — model absent]`     |
| Permission refusée (sudo, hardware, etc.)          | `[BLOCKED@P<x>.<y> — perm]` + passe     |
| Tu hésites entre 2 approches > 30 min              | Pick the simpler, document l'autre dans `resultat_v5.md` "Reste à faire" |

# Annexe B — Quick reference : commandes fréquentes

```bash
# Activate venv (TOUJOURS au début de session)
source .venv/bin/activate

# Tests rapides (un seul fichier)
pytest tests/test_<fichier>.py -v --tb=short

# Tests complets
pytest tests/ -q --tb=line 2>&1 | tail -10

# Smoke
python tests/smoke.py

# Diff staged
git diff --cached --stat
git diff --cached

# Commit atomique
git add <fichiers spécifiques>      # JAMAIS git add -A ou git add .
git commit -m "[P<x>.<y>] <message>"

# Voir l'historique V5
git --no-pager log --oneline chore/sonnet-plan-v4..HEAD

# Revert le dernier commit
git revert HEAD --no-edit
```

# Annexe C — Numéros de ligne références (audit V5, peut bouger)

| Référence                                             | Fichier                          | Ligne(s)  |
|-------------------------------------------------------|----------------------------------|-----------|
| `_running` init                                       | `core/continuous_batcher.py`     | 149       |
| `start()` definition                                  | `core/continuous_batcher.py`     | 216-232   |
| `_loop()` while                                       | `core/continuous_batcher.py`     | 292       |
| `generate()` batcher check                            | `core/inference_pipeline.py`     | 528-541   |
| Batcher init in `_setup_*`                            | `core/inference_pipeline.py`     | 1386-1403 |
| `TransportMethod` enum                                | `core/transfer_manager.py`       | 74        |
| `_get_gpu_pipeline()`                                 | `core/transfer_manager.py`       | 325-331   |
| `send_activation()`                                   | `core/transfer_manager.py`       | 378-466   |
| `_execute_transfer()`                                 | `core/transfer_manager.py`       | 504-570   |
| `direct_vram_copy()` Rust call                        | `core/transfer_manager.py`       | 559-562   |
| `_get_method_for()`                                   | `core/transfer_manager.py`       | 956-966   |
| `model.generate()` single-GPU path                    | `core/backends.py`               | 1799-1835 |
| `use_cache` (probablement absent ligne 1827)          | `core/backends.py`               | (audit)   |

> **Vérifie ces numéros au moment où tu y arrives** : les fichiers évoluent. Utilise `grep`
> ou `sed -n` pour confirmer avant d'éditer.

---

**Fin du Plan V5.**

Bonne route. Si tu as un doute → écris-le dans `resultat_v5.md` plutôt que de deviner.
L'honnêteté > la fierté du score.
