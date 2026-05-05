# Plan V4 (MASTER) — Tout ce qui reste à corriger / améliorer dans VRAMancer

> **POUR L'AGENT EXÉCUTANT — LIRE ATTENTIVEMENT AVANT DE COMMENCER :**
>
> Ce plan est **exhaustif et autonome**. Il couvre TOUT ce qui reste à faire sur VRAMancer
> (perf, dette technique, stubs, doc, tests, CI, polish secondaire). Tu ne dois PAS revenir
> demander un autre plan : tu enchaînes les phases dans l'ordre.
>
> **Règles d'or :**
> 1. Lis chaque section dans l'ordre. Ne saute jamais de validation.
> 2. Après chaque tâche numérotée (P<x>.<y>), fais un commit ATOMIQUE.
> 3. Si une tâche échoue 2 fois → **STOP**, écris dans `resultat_v4.md` une section `[BLOCKED@P<x>.<y>]`,
>    passe à la phase suivante. Ne brute-force pas.
> 4. Si une mesure ne montre pas le gain attendu → REVERT honnêtement et documente.
>    **Ne jamais arrondir vers le haut, ne jamais cacher un échec.**
> 5. Aucune modif dans `_deprecated/`, `core/security/__init__.py`,
>    `core/security/startup_checks.py`, `tests/test_chaos_concurrency.py`,
>    `csrc/paged_attention_kernel.cu`, `core/paged_attention.py`,
>    `core/paged_attention_cuda.py`, `rust_core/src/`.
> 6. Aucune désactivation de tests existants pour faire passer la suite.
> 7. Aucun `git push`, `git push --force`, merge vers main, ouverture de PR.
>    L'utilisateur s'en chargera.
> 8. Toujours `source .venv/bin/activate` avant python/pytest, sinon `ModuleNotFoundError`.
> 9. Tous les commits préfixés `[P<x>.<y>]` (ex: `[P2.3] ...`).
> 10. Honnêteté > marketing. Si un bench est bruité, écris "INDÉTERMINÉ", pas un beau chiffre.

**Version :** v4.0-master
**Date :** 5 mai 2026
**Branche source :** `main` après merge V3 (HEAD = `9bd9327`)
**Branche cible :** `chore/sonnet-plan-v4`
**Auteur :** Architecte Claude Opus 4.7
**Exécutant attendu :** Claude Sonnet 4.6 (ou agent équivalent suivant pas-à-pas)

---

## Table des phases

| Phase | Titre | Effort | Risque | Bloquant si échec ? |
|-------|-------|--------|--------|---------------------|
| P0    | Préparation + baseline | 5 min | Faible | OUI |
| P1    | Polish honnêteté V3 (4 fixes) | 30 min | Faible | NON |
| P2    | Performance — CUDA Stream Overlap | 1-2h | Moyen | NON (revert OK) |
| P3    | Performance — Triton sampling top-k fuser | 1-2h | Élevé | NON (revert OK) |
| P4    | Diagnostic ContinuousBatcher concurrent | 45 min | Faible | NON |
| P5    | Comparaison externe vLLM | 30 min – 2h | Moyen | NON (skip OK) |
| P6    | Stubs réels — formaliser ou fixer | 1h | Faible | NON |
| P7    | Dead code cleanup | 30 min | Faible | NON |
| P8    | Tests — combler les gaps de couverture | 1h | Faible | NON |
| P9    | CI/CD — vérifier et durcir | 30 min | Faible | NON |
| P10   | Dashboard polish (hardcoded data) | 45 min | Faible | NON |
| P11   | Examples reality check | 30 min | Faible | NON |
| P12   | Requirements files audit | 30 min | Faible | NON |
| P13   | Documentation harmonisation | 1h | Faible | NON |
| P14   | Hygiène repo (fichiers root) | 20 min | Faible | NON |
| P15   | Mise à jour `TECHNICAL_DEBT.md` | 15 min | Faible | NON |
| P16   | Validation finale (tests + smoke + sanity) | 15 min | Faible | OUI |
| P17   | Documentation finale `resultat_v4.md` | 30 min | Faible | OUI |

**Total estimé :** ~10-12h équivalent humain. Sur plusieurs sessions si besoin.
**Ordre recommandé :** strictement séquentiel (P0 → P17). Certaines phases internes peuvent être réordonnées localement.

---

# P0 — Préparation et baseline

## P0.1 — Setup branche

```bash
cd /home/jeremie/VRAMancer/VRAMancer
source .venv/bin/activate
git status                           # doit être propre
git --no-pager log --oneline -1      # doit être 9bd9327 ou plus récent
git checkout main
git checkout -b chore/sonnet-plan-v4
```

**Si la branche existe déjà** (par exemple session reprise) :

```bash
git checkout chore/sonnet-plan-v4
git --no-pager log --oneline -5      # voir où on en est
```

Identifie la dernière phase complétée par les messages de commit (`[P<x>.<y>]`) et reprends à la phase suivante.

## P0.2 — Crée `resultat_v4.md`

```bash
cat > resultat_v4.md << 'EOF'
# Résultat Plan V4 (MASTER)

**Date début :** YYYY-MM-DD HH:MM
**Branche :** chore/sonnet-plan-v4
**Plan :** docs/reports/PLAN_ACTION_V4.md
**Base :** main @ 9bd9327

## [BASELINE]
(à remplir P0.3)

## [P1] — Polish honnêteté
## [P2] — CUDA Stream Overlap
## [P3] — Triton sampling top-k
## [P4] — Diagnostic batcher
## [P5] — vLLM benchmark
## [P6] — Stubs formalisés
## [P7] — Dead code cleanup
## [P8] — Tests coverage
## [P9] — CI/CD
## [P10] — Dashboard polish
## [P11] — Examples
## [P12] — Requirements
## [P13] — Doc harmonisation
## [P14] — Hygiène repo
## [P15] — TECHNICAL_DEBT update
## [P16] — Validation finale
## [SUMMARY]
EOF
git add resultat_v4.md
git commit -m "[P0.2] init resultat_v4.md skeleton"
```

## P0.3 — Mesurer la baseline

```bash
source .venv/bin/activate

# Tests
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 VRM_BACKEND_ALLOW_STUB=1 \
  pytest tests/ --ignore=tests/test_chaos_concurrency.py --tb=no --no-cov 2>&1 | tail -3

# GPU mapping
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "import torch; [print(f'torch GPU{i}: {torch.cuda.get_device_properties(i).name}') for i in range(torch.cuda.device_count())]"

# Smoke test
VRM_MINIMAL_TEST=1 python -m tests.smoke 2>&1 | tail -5
```

**Cible attendue tests :** `1 failed, 1070 passed, 39 skipped` (`test_health_imports_fault_manager` est l'unique failure pré-existante).

**SI autre résultat :** STOP. La branche `main` n'est pas dans l'état attendu. Note dans `resultat_v4.md` `[BLOCKED@P0.3]` et demande à l'utilisateur.

Copie les résultats dans `resultat_v4.md` section `[BASELINE]`.

```bash
git add resultat_v4.md
git commit -m "[P0.3] baseline tests/GPU/smoke recorded"
```

---

# P1 — Polish honnêteté V3 (4 mini-fixes)

L'audit V3 a relevé 4 inexactitudes textuelles. Corrections de doc uniquement, pas de code. Effort total ~30 min.

## P1.1 — Annoter les labels GPU dans `resultat_v3.md`

**Problème :** la section V2.1 utilise l'ordre `nvidia-smi` (PCI_BUS_ID), les sections V2.2+ utilisent l'ordre `torch.cuda` (FAST_FIRST). Pas de mention.

1. Ouvre `resultat_v3.md`.
2. Cherche la section `[V2.1]` ReBAR detection.
3. Insère **juste après son titre** :

```markdown
> **Note ordering GPU :**
> - `nvidia-smi` → ordre `PCI_BUS_ID` : GPU0=RTX 5070 Ti (Blackwell, 16GB), GPU1=RTX 3090 (Ampere, 24GB)
> - `torch.cuda` → ordre `FAST_FIRST` (par défaut) : GPU0=RTX 3090, GPU1=RTX 5070 Ti
> Les sections benchmarks suivantes (V2.2+) utilisent l'ordre `torch.cuda`.
```

**Validation :**
```bash
grep -n "FAST_FIRST" resultat_v3.md
```

```bash
git add resultat_v3.md
git commit -m "[P1.1] annotate GPU ordering (nvidia-smi PCI vs torch.cuda FAST_FIRST) in resultat_v3.md"
```

## P1.2 — Réattribuer le delta +382% honnêtement

**Problème :** Attribution générique "ReBAR + Rust P2P bypass" sans pondération.

1. Ouvre `docs/reports/REBAR_PROXMOX_BENCHMARK.md`. Cherche la section qui mentionne `+382%`.
2. **Ajoute** (sans supprimer le texte existant) le bloc :

```markdown
### Attribution honnête du speedup +382% (V4 P1.2 — 2026-05)

**Composantes estimées (à confirmer expérimentalement, voir `TECHNICAL_DEBT.md#REBAR_VS_P2P_ATTRIBUTION`) :**

| Source | Estimation gain | Justification |
|--------|----------------|---------------|
| Rust P2P bypass `send_to_device` ≥512 KB | ~70-80% | Remplace CPU staging (Strategy 4) par `cudaMemcpyPeerAsync` direct. Mars 2026 IOMMU bloquait P2P. |
| ReBAR (BAR1 ≥ VRAM) | ~10-15% | Réduit latence accès, prefetch plus efficace. |
| Optimisations diverses | ~5-10% | accelerate dispatch natif, KV cache pages distribuées. |

**Conclusion honnête :** la décomposition exacte n'est pas mesurée. Le delta est réel, l'attribution est une estimation.
```

3. Ouvre `resultat_v3.md`. Cherche la phrase qui parle de `+382%`. **Ajoute en dessous** (sans supprimer) :

```markdown
> **Note attribution (P1.2 V4) :** Le delta +382% est réel, l'attribution exacte n'est pas mesurée.
> Voir `docs/reports/REBAR_PROXMOX_BENCHMARK.md#attribution-honnete-du-speedup-382-v4-p12-2026-05`.
```

**Validation :**
```bash
grep -A3 "Attribution honnête" docs/reports/REBAR_PROXMOX_BENCHMARK.md
```

```bash
git add docs/reports/REBAR_PROXMOX_BENCHMARK.md resultat_v3.md
git commit -m "[P1.2] honest attribution of +382%: ReBAR ~15% + Rust P2P ~75% + misc ~10%"
```

## P1.3 — Corriger l'explication P6.1 n=1 > sequential

1. Ouvre `resultat_v3.md`. Cherche la section P6.1 / "Stress test" et la mention "warmup".
2. **Ajoute** (sans supprimer) :

```markdown
> **Correction méthodologique (V4 P1.3) :** L'explication "warmup actif" est inexacte.
> Diagnostic réel :
> - Le baseline `Sequential 1 req` mesure `prefill + decode` à froid.
> - Le test `n=1 concurrent` arrive APRÈS → KV cache déjà chaud, tokenizer chaud.
> - Comparaison apples-to-apples = `n=1` vs séquentiel APRÈS warmup identique.
> - À reprendre proprement en P4 (`benchmarks/bench_stress_concurrent_v4.py`).
```

```bash
git add resultat_v3.md
git commit -m "[P1.3] honest methodology note on P6.1 n=1 > sequential (KV hot, not warmup)"
```

## P1.4 — Vérifier les labels TransferManager

**Test concret pour décider** :

```bash
source .venv/bin/activate
python -c "
import torch
from core.transfer_manager import TransferManager
tm = TransferManager()
if torch.cuda.device_count() >= 2:
    print(f'Method 0->1: {tm._get_method_for(0, 1)}')
    print(f'Method 1->0: {tm._get_method_for(1, 0)}')
    print(f'_can_p2p(0,1): {tm._can_p2p(0, 1)}')
else:
    print('SKIP: <2 GPUs')
" 2>&1 | tee /tmp/p14_labels.txt
```

**Décision :**

- **Cas A** : sortie `Method 0->1: CUDA_P2P` → tout va bien.
  ```bash
  git commit --allow-empty -m "[P1.4] verify TransferManager labels: CUDA_P2P correct, no bug"
  ```

- **Cas B** : sortie `Method 0->1: CPU_STAGED` MAIS les benchmarks réels utilisent P2P → label trompeur. Ouvre `docs/reports/TECHNICAL_DEBT.md` et ajoute (dans la table "Stubs réels") :
  ```markdown
  | TRANSFER_MANAGER_LABEL_INCORRECT | core/transfer_manager.py | _get_method_for | Retourne "CPU_STAGED" alors que send_activation utilise effectivement P2P via Rust bypass | Petit | Basse — cosmétique (logs) |
  ```
  ```bash
  git add docs/reports/TECHNICAL_DEBT.md
  git commit -m "[P1.4] document TRANSFER_MANAGER_LABEL_INCORRECT in TECHNICAL_DEBT.md"
  ```

Reporte la sortie exacte dans `resultat_v4.md` section [P1].

---

# P2 — Performance : CUDA Stream Overlap dans `TransferManager`

**Hypothèse :** pipeliner compute // transfer P2P. Si gain ≥3% sur Qwen 14B 2-GPU → MERGE. Sinon REVERT honnêtement.

## P2.1 — Audit des points de synchronisation

```bash
grep -n "synchronize\|wait_stream\|cuda\.Stream\|cuda\.Event" core/transfer_manager.py
grep -n "send_activation\|tensor\.to\|\.to(f.cuda" core/inference_pipeline.py | head -30
```

Crée `docs/reports/STREAM_OVERLAP_AUDIT.md` (30-80 lignes) :

```markdown
# Stream Overlap Audit — VRAMancer V4 P2.1

## Sync points actuels dans TransferManager
- (ligne X) ...
- (ligne Y) ...

## Path P2P actuel
- send_activation() → ... → torch.tensor.to() default stream → sync implicite

## Caller usage dans inference_pipeline.py
- N appels par token décodé
- Pattern actuel : <bloc N compute> → <send_activation> → <bloc N+1 compute>
- Sync implicite force séquentiel.

## Modification proposée
- Stream dédié src_gpu pour cudaMemcpyPeerAsync
- Event sur stream → caller wait avant lecture côté dst
- Flag VRM_TRANSFER_OVERLAP (default off jusqu'à validation)

## Risque
- Race condition si caller lit avant event.wait()
- Mitigation : flag off par défaut + bench A/B explicite
```

```bash
git add docs/reports/STREAM_OVERLAP_AUDIT.md
git commit -m "[P2.1] audit send_activation sync points and stream usage"
```

## P2.2 — Bench BASELINE Qwen 14B 2-GPU (avant modif)

Crée `/tmp/bench_v4_p2_baseline.py` (NE PAS committer dans le repo) :

```python
import os, time, statistics, torch
os.environ['VRM_QUANTIZATION'] = ''  # BF16
os.environ['VRM_TRANSFER_OVERLAP'] = '0'  # baseline explicite
from core.inference_pipeline import InferencePipeline

pipe = InferencePipeline(backend_name='huggingface', verbose=False)
pipe.load('Qwen/Qwen2.5-14B', num_gpus=2)
prompt = 'Explain quantum entanglement in one paragraph.'
pipe.generate(prompt, max_new_tokens=20)  # warmup

times = []
for i in range(5):
    t0 = time.perf_counter()
    pipe.generate(prompt, max_new_tokens=200)
    ts = 200 / (time.perf_counter() - t0)
    times.append(ts)
    print(f'Run {i+1}: {ts:.2f} tok/s')

print(f'\nBASELINE median: {statistics.median(times):.2f} tok/s')
print(f'BASELINE mean:   {sum(times)/len(times):.2f} tok/s')
print(f'BASELINE stddev: {statistics.stdev(times):.2f}')
```

```bash
source .venv/bin/activate
python /tmp/bench_v4_p2_baseline.py 2>&1 | tee /tmp/bench_v4_p2_baseline.txt
```

Reporte les 5 runs + median dans `resultat_v4.md` section [P2.2].

**Si median diffère >10% de 28.92 (V3) :** suspicieux, refais après reboot ou note "INDÉTERMINÉ — variance excessive".

## P2.3 — Implémenter le flag CUDA Stream Overlap

Dans `core/transfer_manager.py`, ajouter dans `__init__` :

```python
import os
self._overlap_enabled = os.environ.get('VRM_TRANSFER_OVERLAP', '0') == '1'
self._transfer_streams = {}  # {src_gpu: torch.cuda.Stream}
```

Ajouter méthode :

```python
def _get_transfer_stream(self, src_gpu: int):
    """Lazy-init un CUDA Stream dédié pour les transferts depuis src_gpu."""
    if not _HAS_TORCH or not torch.cuda.is_available():
        return None
    if src_gpu not in self._transfer_streams:
        with torch.cuda.device(src_gpu):
            self._transfer_streams[src_gpu] = torch.cuda.Stream(priority=-1)
    return self._transfer_streams[src_gpu]
```

Dans `send_activation` (path P2P uniquement, identifié en P2.1), wrapper :

```python
if self._overlap_enabled and self._can_p2p(src_gpu, dst_gpu):
    stream = self._get_transfer_stream(src_gpu)
    if stream is not None:
        with torch.cuda.stream(stream):
            dst_tensor = tensor.to(f'cuda:{dst_gpu}', non_blocking=True)
        # Le caller doit synchroniser via event si overlap actif
        event = torch.cuda.Event()
        event.record(stream)
        dst_tensor._vrm_transfer_event = event  # attaché pour wait possible côté caller
        return dst_tensor
# fallback path actuel inchangé
```

**ATTENTION** : si le code caller actuel ne wait pas sur `_vrm_transfer_event`, c'est une race condition. C'est précisément pour ça que le flag est OFF par défaut — le bench P2.4 va isoler l'effet.

**Validation locale (avant commit) :**

```bash
source .venv/bin/activate
# Tests transfer_manager + pipeline doivent rester verts
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 VRM_BACKEND_ALLOW_STUB=1 \
  pytest tests/test_transfer_manager.py tests/test_pipeline.py --tb=short --no-cov 2>&1 | tail -5

# Smoke
python -c "from core.transfer_manager import TransferManager; tm = TransferManager(); print('overlap=', tm._overlap_enabled)"
```

```bash
git add core/transfer_manager.py
git commit -m "[P2.3] add VRM_TRANSFER_OVERLAP flag with CUDA Stream pool for P2P send_activation"
```

## P2.4 — Bench AVEC overlap

Copie `/tmp/bench_v4_p2_baseline.py` vers `/tmp/bench_v4_p2_overlap.py` et change `VRM_TRANSFER_OVERLAP` à `'1'`.

```bash
source .venv/bin/activate
python /tmp/bench_v4_p2_overlap.py 2>&1 | tee /tmp/bench_v4_p2_overlap.txt
```

Tableau dans `resultat_v4.md` section [P2.4] :

```markdown
| Run | Baseline | Overlap |
|-----|----------|---------|
| 1   | XX.XX    | YY.YY   |
| 2   | XX.XX    | YY.YY   |
| 3   | XX.XX    | YY.YY   |
| 4   | XX.XX    | YY.YY   |
| 5   | XX.XX    | YY.YY   |
| **Median** | **XX.XX** | **YY.YY** |
| **Delta** | — | **+Z.Z%** |
```

## P2.5 — Décision MERGE / REVERT

**Si median delta ≥ +3%** :
```bash
# Active par défaut
sed -i "s|os.environ.get('VRM_TRANSFER_OVERLAP', '0')|os.environ.get('VRM_TRANSFER_OVERLAP', '1')|" core/transfer_manager.py
git add core/transfer_manager.py
git commit -m "[P2.5] enable VRM_TRANSFER_OVERLAP by default (gain +Z.Z% on Qwen 14B 2-GPU)"
```

**Si delta < +3% ou régression** :
```bash
git revert HEAD~1 --no-edit  # revert P2.3 si pas encore P2.5
# Ou si suite plus complexe :
# git reset --hard HEAD~N
```
Documente dans `STREAM_OVERLAP_AUDIT.md` :
```markdown
## Conclusion (P2.5)
Gain mesuré insuffisant (Z.Z%) → REVERT. Le bottleneck n'est pas la sync transfer.
Hypothèses pour futur travail : kernel decode lui-même, GIL Python, overhead accelerate.
```
```bash
git add docs/reports/STREAM_OVERLAP_AUDIT.md
git commit -m "[P2.5] revert overlap (gain insufficient Z.Z%) + document conclusion"
```

**Si bench bloqué (OOM, crash) :**
- Ne brute-force pas. REVERT P2.3 :
  ```bash
  git revert HEAD --no-edit
  ```
- Note `[BLOCKED@P2.4]` dans `resultat_v4.md`.

---

# P3 — Performance : Triton sampling top-k fuser

**Stub identifié dans `TECHNICAL_DEBT.md` : `TRITON_SAMPLING_TOPK`**

`core/triton_sampling.py` a déjà un fast path top-k (Python `torch.topk` + `torch.softmax(k)` + `torch.multinomial`). L'audit V3 disait "fallback PyTorch toujours utilisé". Il faut vérifier ça d'abord.

## P3.1 — Instrumenter et mesurer le path utilisé

Ajoute en haut de `core/triton_sampling.py` :

```python
import os
_DEBUG_SAMPLING = os.environ.get('VRM_DEBUG_SAMPLING', '0') == '1'
_PATH_COUNTS = {'fast_topk': 0, 'triton_full': 0, 'pytorch_fallback': 0, 'greedy': 0}
```

Dans `fused_sample`, à chaque branche :
```python
if greedy or temperature <= 0:
    if _DEBUG_SAMPLING: _PATH_COUNTS['greedy'] += 1
    ...
if top_k > 0 and top_k < vocab_size:
    if _DEBUG_SAMPLING: _PATH_COUNTS['fast_topk'] += 1
    ...
if _HAS_TRITON and logits.is_cuda:
    if _DEBUG_SAMPLING: _PATH_COUNTS['triton_full'] += 1
    ...
# fallback final :
if _DEBUG_SAMPLING: _PATH_COUNTS['pytorch_fallback'] += 1
```

```bash
source .venv/bin/activate
VRM_DEBUG_SAMPLING=1 python -c "
import os
os.environ['VRM_DEBUG_SAMPLING'] = '1'
from core.inference_pipeline import InferencePipeline
from core.triton_sampling import _PATH_COUNTS
pipe = InferencePipeline(backend_name='huggingface', verbose=False)
pipe.load('gpt2', num_gpus=1)
pipe.generate('Hello world, this is', max_new_tokens=100)
print('PATH_COUNTS:', _PATH_COUNTS)
" 2>&1 | tee /tmp/p3_sampling_paths.txt
```

Reporte dans `resultat_v4.md` section [P3.1].

## P3.2 — Décision selon les paths utilisés

- **Si `fast_topk` >90% des appels** : tout va bien, le stub TECHNICAL_DEBT est obsolète.
  - Garde l'instrumentation (utile debug) **OU** revert si elle alourdit le hot path mesurablement.
  - Bench rapide : 50 tokens GPT-2 avec/sans `_DEBUG_SAMPLING=1`. Si <2% overhead : garde.
  ```bash
  git add core/triton_sampling.py
  git commit -m "[P3.2] add VRM_DEBUG_SAMPLING flag — fast_topk path is dominant (>90%)"
  ```
  Mets à jour `TECHNICAL_DEBT.md` : déplace `TRITON_SAMPLING_TOPK` vers "Stubs résolus" :
  ```markdown
  - ✅ `triton_sampling TRITON_SAMPLING_TOPK` → fast_topk path domine déjà (>90% calls), pas de stub réel
  ```
  ```bash
  git add docs/reports/TECHNICAL_DEBT.md
  git commit -m "[P3.2] resolve TRITON_SAMPLING_TOPK in TECHNICAL_DEBT (fast path active)"
  ```

- **Si `pytorch_fallback` >50%** : vrai stub, vrai gain potentiel. Va en P3.3.

- **Si `triton_full` est dominant et `top_k=0` partout** : le caller passe `top_k=0` → la branche Triton full-vocab tourne. Il faut peut-être passer un `top_k=50` par défaut dans le pipeline. Va en P3.4.

## P3.3 — (Si pytorch_fallback dominant) Investiguer pourquoi

Cherche les appels `fused_sample` :
```bash
grep -rn "fused_sample\|from.*triton_sampling" core/ | grep -v __pycache__
```

Identifie le caller principal (probablement `core/inference_pipeline.py` ou `core/backends.py`). Vérifie quels arguments sont passés. La condition `_HAS_TRITON and logits.is_cuda` doit être False quelque part :
```bash
python -c "from core.triton_sampling import _HAS_TRITON; print('_HAS_TRITON =', _HAS_TRITON)"
```

- Si `_HAS_TRITON = False` → triton non installé/non importé. Diagnostic : `pip show triton`.
- Si `logits.is_cuda = False` → caller passe les logits sur CPU. À fixer côté caller.

Ajoute la conclusion au `resultat_v4.md` et un `TECHNICAL_DEBT` entry si nécessaire.

```bash
git commit --allow-empty -m "[P3.3] diagnose triton_sampling pytorch_fallback dominance: <root cause>"
```

## P3.4 — (Si triton_full dominant + top_k=0) Activer top_k=50 par défaut

Dans `core/inference_pipeline.py`, cherche les appels à `fused_sample`. Si `top_k` n'est pas passé, le default est 0. Modifie :

```python
fused_sample(logits, temperature=temp, top_k=50, top_p=0.9, ...)
```

Bench A/B : 200 tokens Qwen 7B avec top_k=0 vs top_k=50.

Si gain ≥3% : MERGE. Sinon REVERT.

```bash
git add core/inference_pipeline.py
git commit -m "[P3.4] default top_k=50 in fused_sample call (gain +Z.Z%)"
```

---

# P4 — Diagnostic ContinuousBatcher concurrent

**Question critique :** V3 P6.1 a montré n=4 (42.9 tok/s) **<** n=1 (74.6 tok/s). Anormal pour un système avec batching. Trancher : bug ou test mal écrit ?

## P4.1 — Audit du routing batcher

```bash
grep -n "continuous_batcher" core/inference_pipeline.py | head -20
```

Lis `core/inference_pipeline.py` autour de la ligne 525-540 (route via batcher si `_running`). Vérifie :
- L'env `VRM_CONTINUOUS_BATCHING=1` initialise bien le batcher (ligne ~317 `_init_continuous_batching`).
- `submit()` est non-bloquant (retourne un Future).
- Le batcher démarre quand on l'appelle (ligne ~1575 `start()`).

Regarde `core/continuous_batcher.py` méthode `submit` :
```bash
grep -n "def submit\|def _run_loop\|with self._lock" core/continuous_batcher.py
```

Documente dans `resultat_v4.md` section [P4.1] : "submit est non-bloquant ✅" ou "submit acquiert un lock global ❌".

```bash
git commit --allow-empty -m "[P4.1] audit continuous_batcher routing: <conclusion>"
```

## P4.2 — Bench stress avec/sans batcher

Crée `benchmarks/bench_stress_concurrent_v4.py` (CE FICHIER VA DANS LE REPO) :

```python
"""Stress test V4: continuous batcher behavior under concurrent load.

Usage:
  python benchmarks/bench_stress_concurrent_v4.py             # batcher OFF
  VRM_CONTINUOUS_BATCHING=1 python benchmarks/bench_stress_concurrent_v4.py  # batcher ON
"""
import os, time, threading, statistics


def run_concurrent(pipe, prompt, n_concurrent, max_tokens=100):
    results = [None] * n_concurrent
    threads = []
    barrier = threading.Barrier(n_concurrent)

    def worker(idx):
        barrier.wait()
        t0 = time.perf_counter()
        pipe.generate(prompt, max_new_tokens=max_tokens)
        results[idx] = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(n_concurrent):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    total_time = time.perf_counter() - t0
    throughput = (n_concurrent * max_tokens) / total_time
    return total_time, throughput, results


def main():
    batcher_on = os.environ.get('VRM_CONTINUOUS_BATCHING', '0') == '1'
    print(f'=== Batcher: {"ON" if batcher_on else "OFF"} ===')

    from core.inference_pipeline import InferencePipeline
    pipe = InferencePipeline(backend_name='huggingface', verbose=False)
    pipe.load('Qwen/Qwen2.5-7B-Instruct', num_gpus=1)

    prompt = 'Explain quantum entanglement.'
    pipe.generate(prompt, max_new_tokens=20)  # warmup

    seq_times = []
    for i in range(3):
        t0 = time.perf_counter()
        pipe.generate(prompt, max_new_tokens=100)
        seq_times.append(100 / (time.perf_counter() - t0))
    print(f'Sequential median: {statistics.median(seq_times):.2f} tok/s')

    for n in [1, 4, 8]:
        total_time, throughput, _ = run_concurrent(pipe, prompt, n, max_tokens=100)
        print(f'N={n}: total {total_time:.2f}s, total throughput {throughput:.2f} tok/s')


if __name__ == '__main__':
    main()
```

Lance les deux modes :
```bash
source .venv/bin/activate
python benchmarks/bench_stress_concurrent_v4.py 2>&1 | tee /tmp/p4_no_batcher.txt
VRM_CONTINUOUS_BATCHING=1 python benchmarks/bench_stress_concurrent_v4.py 2>&1 | tee /tmp/p4_with_batcher.txt
```

Tableau dans `resultat_v4.md` section [P4.2].

```bash
git add benchmarks/bench_stress_concurrent_v4.py
git commit -m "[P4.2] add bench_stress_concurrent_v4.py — diagnostic batcher behavior"
```

## P4.3 — Verdict batcher

Dans `resultat_v4.md`, coche **une seule case** :

```markdown
### [P4.3] Verdict ContinuousBatcher

[ ] (A) Le batcher fonctionne (N=4 throughput > N=1 quand activé).
    → V3 P6.1 a sous-utilisé le batcher (env non activé). Pas de bug.

[ ] (B) Le batcher ne batche pas même activé.
    → BUG réel. Documenter dans TECHNICAL_DEBT.md (CONTINUOUS_BATCHER_NO_BATCHING).
    Hypothèses : lock global submit(), GIL transition par requête, _running mal géré.

[ ] (C) Indéterminé (variance excessive).
    → Refaire avec n_runs=10, prompts variés, max_tokens=200.
```

Si (B) :
```markdown
| CONTINUOUS_BATCHER_NO_BATCHING | core/continuous_batcher.py | submit | Throughput n>1 ≤ n=1 — le batcher sérialise les requêtes au lieu de les grouper | Moyen | Élevée |
```

```bash
git add resultat_v4.md
# (+ TECHNICAL_DEBT.md si cas B)
git commit -m "[P4.3] verdict ContinuousBatcher: case <A|B|C>"
```

---

# P5 — Comparaison externe vLLM

**Pourquoi :** calibrer où VRAMancer se situe vs l'état de l'art. Si vLLM est plus rapide, dis-le honnêtement.

## P5.1 — Tenter d'installer vLLM

```bash
source .venv/bin/activate
pip show vllm 2>&1 | head -3
```

Si pas installé :
```bash
pip install vllm 2>&1 | tee /tmp/p5_vllm_install.log | tail -20
```

**Cas d'échec courants :**
1. CUDA mismatch → `pip install vllm --index-url https://download.pytorch.org/whl/cu128`
2. Conflit torch → `pip install "vllm==0.6.x"`
3. OOM compile → utilise wheel pré-compilé

**Si SKIP impossible** : ajoute dans `TECHNICAL_DEBT.md` "Limitations connues" :
```markdown
| vLLM benchmark non disponible (incompat torch 2.10+cu128) | venv dédié pour bench externe (futur) |
```
```bash
git add docs/reports/TECHNICAL_DEBT.md
git commit -m "[P5.1] document vLLM install incompatibility — skip P5.2/P5.3"
```
Et passe à **P6**.

**Si OK :**
```bash
python -c "import vllm; print('vllm', vllm.__version__)"
git commit --allow-empty -m "[P5.1] vLLM installed: version=X.Y.Z"
```

## P5.2 — Bench Qwen2.5-7B-Instruct 1-GPU

Crée `/tmp/bench_v4_p5_vramancer.py` et `/tmp/bench_v4_p5_vllm.py` (NE PAS committer) — voir templates dans P2.4.

Lance dans cet ordre (un engine à la fois en VRAM) :
```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 python /tmp/bench_v4_p5_vramancer.py 2>&1 | tee /tmp/p5_vram.txt
CUDA_VISIBLE_DEVICES=0 python /tmp/bench_v4_p5_vllm.py 2>&1 | tee /tmp/p5_vllm.txt
```

Tableau dans `resultat_v4.md` :

```markdown
### [P5.2] Qwen 7B 1-GPU

| Backend     | Tok/s (median) | VRAM     |
|-------------|----------------|----------|
| VRAMancer HF| XX.X           | YY.Y GiB |
| vLLM        | XX.X           | YY.Y GiB |
| **Ratio**   | VRAMancer = X.X% de vLLM | |
```

```bash
git commit --allow-empty -m "[P5.2] Qwen 7B vLLM bench: VRAMancer=XX vs vLLM=YY tok/s"
```

## P5.3 — Bench Qwen2.5-14B 2-GPU hétérogène

```python
# /tmp/bench_v4_p5_vllm_14b.py
from vllm import LLM, SamplingParams
try:
    llm = LLM(model='Qwen/Qwen2.5-14B', tensor_parallel_size=2, dtype='bfloat16')
    print('vLLM ACCEPTED hetero TP=2')
    # bench
except Exception as e:
    print(f'vLLM REJECTED hetero: {e}')
```

Si vLLM refuse → c'est un argument fort pour VRAMancer. Documente l'erreur exacte dans `resultat_v4.md`.

```bash
git commit --allow-empty -m "[P5.3] vLLM 14B hetero: <accepted XX tok/s | refused>"
```

---

# P6 — Stubs réels : formaliser ou fixer

Liste depuis `TECHNICAL_DEBT.md` :

| ID | Action V4 |
|----|-----------|
| VTP_L3 | Formaliser (déjà TODO marker, ajoute test skip propre) |
| DMABUF_WRITE | Formaliser (header explicite) |
| NAT_HOLE_PUNCH | Formaliser (skip propre + doc) |
| WEBGPU_BACKEND | Déjà déprécié, vérifier qu'aucun import ne reste |
| BATCH_INFERENCE | Idem |
| CUDA_GRAPH_MULTI_GPU | Doc claire (impossibilité fondamentale, pas un bug) |
| TRITON_SAMPLING_TOPK | Traité en P3 |

## P6.1 — VTP_L3 : test skip propre

Cherche les tests qui couvrent VTP :
```bash
grep -rn "vtp_core\|VTP_L3\|target_tier" tests/ | grep -v __pycache__
```

Si un test existe et est marqué slow/skip, vérifie qu'il est correctement annoté avec un message clair. Sinon, ajoute un test :

```python
# tests/test_vtp_l3_stub.py
import pytest
import importlib.util

@pytest.mark.skipif(
    True,
    reason="VTP_L3 router stub: returns src.clone() — no real RDMA transport. "
           "See docs/reports/TECHNICAL_DEBT.md#vtp_l3"
)
def test_vtp_l3_remote_rdma():
    """Placeholder for VTP_L3 RDMA transport.
    Will be enabled when csrc/vtp_core.cpp implements actual libibverbs path.
    """
    pytest.fail("not implemented")
```

```bash
git add tests/test_vtp_l3_stub.py
git commit -m "[P6.1] add explicit skip test for VTP_L3 stub (TECHNICAL_DEBT)"
```

## P6.2 — DMABUF_WRITE : header explicite

Vérifie `csrc/dmabuf_bridge.c` a un header clair en haut :
```bash
head -30 csrc/dmabuf_bridge.c
```

Si pas de mention "STUB / mmap write not implemented", ajoute :
```c
// =============================================================================
// DMA-BUF BRIDGE — STUB (V4 P6.2)
// -----------------------------------------------------------------------------
// Steps 1-3 (DRM ioctl, fd export) IMPLEMENTED.
// Steps 4-5 (dst mmap write back to GPU) NOT IMPLEMENTED.
// Caller must finalize transfer via torch pinned memory.
// See: docs/reports/TECHNICAL_DEBT.md#DMABUF_WRITE
// =============================================================================
```

```bash
git add csrc/dmabuf_bridge.c
git commit -m "[P6.2] explicit STUB header in csrc/dmabuf_bridge.c"
```

## P6.3 — NAT_HOLE_PUNCH : skip propre

```bash
grep -rn "punch_hole\|nat_traversal" tests/
```

Ajoute (ou complète) un test skip propre :

```python
# tests/test_nat_traversal_stub.py
import pytest

@pytest.mark.skipif(
    True,
    reason="NAT hole punch / TURN relay non testé en WAN réel. "
           "STUN RFC 5389 fonctionnel. See TECHNICAL_DEBT.md#NAT_HOLE_PUNCH"
)
def test_nat_punch_wan():
    pytest.fail("not implemented")


def test_stun_basic():
    """STUN RFC 5389 implemented and reachable in LAN."""
    from core.network.nat_traversal import stun_query
    # Test minimal — server fictif ou skip si pas d'accès réseau
    pytest.skip("requires reachable STUN server (binding test only in CI)")
```

```bash
git add tests/test_nat_traversal_stub.py
git commit -m "[P6.3] add explicit skip tests for NAT hole punch + STUN basic placeholder"
```

## P6.4 — WEBGPU_BACKEND : vérifier déprécation propre

```bash
grep -rn "from.*backends_webgpu\|import backends_webgpu" --include="*.py" | grep -v _deprecated/ | grep -v __pycache__
```

**Aucun import attendu en dehors de `_deprecated/`.** Si tu en trouves :
1. Soit l'import est dans `core/__init__.py` ou un module actif → enlève-le.
2. Soit il faut le rediriger vers un shim qui raise `ImportError("deprecated, see _deprecated/")`.

```bash
git add <fichier modifié>
git commit -m "[P6.4] remove residual import of deprecated backends_webgpu"
# OU si rien à faire :
git commit --allow-empty -m "[P6.4] verified no residual import of deprecated backends_webgpu"
```

## P6.5 — BATCH_INFERENCE : idem

```bash
grep -rn "from.*batch_inference\|import batch_inference" --include="*.py" | grep -v _deprecated/ | grep -v __pycache__
```

Action identique à P6.4.

```bash
git commit --allow-empty -m "[P6.5] verified no residual import of deprecated batch_inference"
```

## P6.6 — CUDA_GRAPH_MULTI_GPU : note explicite

Dans `core/cuda_graph_decode.py`, vérifier qu'il y a un docstring/commentaire en tête :

```python
"""TurboEngine — CUDA Graph capture for single-device decode.

LIMITATION (NOT A BUG):
- CUDA Graphs cannot capture NCCL collectives or P2P operations.
- This is a fundamental PyTorch/CUDA constraint.
- For multi-GPU pipeline parallelism, fall back to eager mode (no graph).
- See: docs/reports/CUDA_GRAPH_MULTI_GPU_AUDIT.md
"""
```

Ajoute si manquant. Sinon commit empty.

```bash
git add core/cuda_graph_decode.py
git commit -m "[P6.6] explicit single-device limitation in cuda_graph_decode docstring"
```

---

# P7 — Dead code cleanup

Dead code identifié dans l'audit : `core/telemetry.py` (orphelin), `core/swarm_ledger.py` (orchestrateur l'ignore).

## P7.1 — `core/telemetry.py`

```bash
grep -rn "from core.telemetry\|import telemetry" --include="*.py" | grep -v __pycache__ | grep -v _deprecated/
```

- Si **aucun import** : déplace dans `_deprecated/telemetry.py` et ajoute une note.
- Si imports actifs : laisse tel quel, ajoute un docstring "experimental, no consumer yet".

```bash
# Cas déplacement :
mkdir -p _deprecated
git mv core/telemetry.py _deprecated/telemetry.py
echo "# Moved 2026-05 — no active consumer (mDNS preferred)" >> _deprecated/telemetry.py
git add _deprecated/telemetry.py
git commit -m "[P7.1] move core/telemetry.py to _deprecated (no active consumer)"
```

## P7.2 — `core/swarm_ledger.py`

```bash
grep -rn "from core.swarm_ledger\|swarm_ledger" --include="*.py" | grep -v __pycache__ | grep -v _deprecated/
```

L'audit dit "fonctionnel mais orphelin". Décision :
- Si peu/pas d'usage → déplace dans `_deprecated/`.
- Si quelques imports → laisse + ajoute docstring "currently unused by orchestrator, kept for SQLite ledger reference".

```bash
git commit --allow-empty -m "[P7.2] swarm_ledger audit: <kept | moved to _deprecated>"
```

## P7.3 — Dead code dans `inference_pipeline.py`

L'audit V3 mentionne "dead code WebGPU/swarm" dans `inference_pipeline.py`. Cherche :
```bash
grep -n "webgpu\|swarm\|holographic" core/inference_pipeline.py
```

Pour chaque match :
- Si la ligne est inaccessible (env flag jamais activé, condition toujours fausse) → supprime la branche.
- Sinon → laisse.

```bash
git add core/inference_pipeline.py
git commit -m "[P7.3] remove dead WebGPU/swarm branches in inference_pipeline (unreachable)"
# OU si rien à faire:
git commit --allow-empty -m "[P7.3] inference_pipeline dead code audit: no unreachable branch found"
```

---

# P8 — Tests : combler les gaps

## P8.1 — Coverage actuel

```bash
source .venv/bin/activate
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 VRM_BACKEND_ALLOW_STUB=1 \
  pytest tests/ --ignore=tests/test_chaos_concurrency.py \
  --cov=core --cov-report=term-missing --tb=no 2>&1 | tail -30
```

Identifie les modules avec coverage <50%. Cible : `core/transfer_manager.py`, `core/triton_sampling.py`, `core/cuda_graph_decode.py`, `core/connectome.py`.

## P8.2 — Test minimal pour `transfer_manager`

Ajoute (si manquant) `tests/test_transfer_manager_basic.py` :

```python
import pytest

def test_transfer_manager_init():
    from core.transfer_manager import TransferManager
    tm = TransferManager()
    assert tm is not None
    assert hasattr(tm, '_transfer_streams')

def test_transfer_manager_method_for_invalid():
    from core.transfer_manager import TransferManager
    tm = TransferManager()
    method = tm._get_method_for(0, 99)
    assert method in ('CUDA_P2P', 'NCCL', 'CPU_STAGED') or method.startswith('CROSS_VENDOR')
```

```bash
git add tests/test_transfer_manager_basic.py
git commit -m "[P8.2] add minimal coverage tests for TransferManager"
```

## P8.3 — Test pour `triton_sampling`

Ajoute (si manquant) `tests/test_triton_sampling_paths.py` :

```python
import pytest

def test_fused_sample_greedy():
    pytest.importorskip("torch")
    import torch
    from core.triton_sampling import fused_sample
    logits = torch.randn(1, 1000)
    out = fused_sample(logits, greedy=True)
    assert out.shape == (1, 1)


def test_fused_sample_topk():
    pytest.importorskip("torch")
    import torch
    from core.triton_sampling import fused_sample
    logits = torch.randn(2, 1000)
    out = fused_sample(logits, temperature=1.0, top_k=50, top_p=0.9)
    assert out.shape == (2, 1)
    assert (out >= 0).all() and (out < 1000).all()
```

```bash
git add tests/test_triton_sampling_paths.py
git commit -m "[P8.3] add minimal coverage tests for triton_sampling fused_sample"
```

## P8.4 — Lance la suite complète

```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 VRM_BACKEND_ALLOW_STUB=1 \
  pytest tests/ --ignore=tests/test_chaos_concurrency.py --tb=short --no-cov 2>&1 | tail -10
```

**Cible :** 1 failed (préexistante), passed ≥ 1070+nouveaux. Si régression → revert le test fautif.

```bash
git commit --allow-empty -m "[P8.4] full suite: 1 failed, NNNN passed (was 1070)"
```

---

# P9 — CI / CD

## P9.1 — Vérifier que les jobs Rust tournent

```bash
ls .github/workflows/ 2>/dev/null
cat .github/workflows/*.yml 2>/dev/null | head -100
```

Confirme :
1. Job `test` → lance pytest avec env stub-safe.
2. Job `rust` → lance `cargo build --release` et `cargo test`.
3. Job `lint` → flake8/ruff.

Si un job manque, ajoute-le. Sinon commit empty.

```bash
git commit --allow-empty -m "[P9.1] CI workflows audit: <ok | added X>"
```

## P9.2 — Pre-commit hook minimal

Si pas de `.pre-commit-config.yaml`, ajoute :

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=2048']
```

```bash
git add .pre-commit-config.yaml
git commit -m "[P9.2] add minimal pre-commit hooks (whitespace, large files)"
```

---

# P10 — Dashboard polish (hardcoded GPU data)

L'audit V3 dit `dashboard/dashboard_web.py` Grade C — "GPU data hardcodees dans templates".

## P10.1 — Identifier les hardcodes

```bash
grep -rn "RTX 3090\|RTX 5070\|GeForce\|24576\|16384" dashboard/templates/ dashboard/static/ 2>/dev/null | head -20
```

Pour chaque match dans un template HTML : remplace par un placeholder Jinja `{{ gpu.name }}`, `{{ gpu.memory }}`, etc., puis fais en sorte que le backend Flask passe les vraies données via `core/monitor.py`.

## P10.2 — Backend dashboard expose les vraies métriques

Dans `dashboard/dashboard_web.py`, route GPU :

```python
@app.route('/api/dashboard/gpus')
def gpus():
    from core.monitor import GPUMonitor
    mon = GPUMonitor()
    return jsonify(mon.snapshot())
```

Et dans le template, remplace les blocs hardcodés par un fetch JS sur `/api/dashboard/gpus`.

```bash
git add dashboard/
git commit -m "[P10.1+P10.2] replace hardcoded GPU data with /api/dashboard/gpus endpoint"
```

## P10.3 — Smoke test dashboard

```bash
source .venv/bin/activate
VRM_API_TOKEN=testtoken python -c "
from dashboard.dashboard_web import app
client = app.test_client()
r = client.get('/api/dashboard/gpus', headers={'X-VRM-Token': 'testtoken'})
print('Status:', r.status_code)
print('JSON:', r.get_json())
"
```

```bash
git commit --allow-empty -m "[P10.3] dashboard smoke: /api/dashboard/gpus returns live data"
```

---

# P11 — Examples reality check

Dossier `examples/` (4 fichiers + quickstart). Chaque exemple doit s'exécuter ou être marqué clairement.

## P11.1 — Inventaire et test

```bash
ls examples/
for f in examples/*.py; do
    echo "=== $f ==="
    head -5 "$f"
    echo "---"
done
```

## P11.2 — Test rapide chaque exemple (mode stub)

```bash
source .venv/bin/activate
for f in examples/*.py; do
    echo "=== Testing $f ==="
    VRM_MINIMAL_TEST=1 timeout 30 python "$f" 2>&1 | tail -10 || echo "FAILED or TIMEOUT"
    echo
done
```

Pour chaque exemple qui fail :
- Si l'exemple démontre une feature spécifique GPU : ajoute un header `# Requires: GPU, model X loaded`.
- Si l'exemple est obsolète (ex: backends_webgpu) : déplace dans `_deprecated/examples/`.

```bash
git add examples/
git commit -m "[P11.2] examples: header requirements + move N obsolete to _deprecated"
```

---

# P12 — Requirements files audit

4 fichiers : `requirements.txt`, `requirements-full.txt`, `requirements-lite.txt`, `requirements-windows.txt`.

## P12.1 — Diff entre fichiers

```bash
diff requirements.txt requirements-lite.txt
diff requirements.txt requirements-full.txt
diff requirements-windows.txt requirements.txt
```

## P12.2 — Confirme la stratégie

Strategy attendue :
- `requirements.txt` : core minimal pour faire tourner pipeline 1-GPU.
- `requirements-lite.txt` : encore plus léger (CPU only? embarqué?).
- `requirements-full.txt` : tout (vLLM, llama-cpp, monitoring, dashboard).
- `requirements-windows.txt` : core + adaptations Windows.

Si la stratégie n'est pas documentée, ajoute en tête de chaque fichier un commentaire clair :

```
# requirements.txt — core: torch, transformers, accelerate, flask, prometheus_client
# Pour le mode léger (sans dashboard) → requirements-lite.txt
# Pour tout (vLLM, llama-cpp, monitoring) → requirements-full.txt
# Voir docs/COMPATIBILITY.md pour les versions supportées.
```

## P12.3 — Vérifie les versions

```bash
grep -E "torch|transformers|accelerate|flask|vllm" requirements*.txt
```

Confirme cohérence (par ex pas torch>=2.0 dans un fichier et torch>=2.10 dans un autre sauf raison).

```bash
git add requirements*.txt
git commit -m "[P12] requirements files: header stratégie + versions cohérentes"
```

---

# P13 — Documentation harmonisation

## P13.1 — Versions cohérentes

```bash
grep -rn "1\.[0-9]\.[0-9]" \
  pyproject.toml setup.cfg core/__init__.py README.md INSTALL_*.md \
  docs/COMPATIBILITY.md docs/QUICKSTART.md 2>/dev/null
```

Source de vérité : `core/__init__.py:__version__`. Tous les autres fichiers doivent matcher (1.5.0).

```bash
# Si désynchronisé :
sed -i 's|version = "1\.4\.0"|version = "1.5.0"|' pyproject.toml
# etc.
git add <fichiers>
git commit -m "[P13.1] harmonize version 1.5.0 across pyproject/setup/README/INSTALL"
```

## P13.2 — README hub

Vérifie que `README.md` mentionne :
- Version actuelle
- Lien vers `docs/QUICKSTART.md`
- Lien vers `docs/COMPATIBILITY.md`
- Lien vers `docs/reports/TECHNICAL_DEBT.md`
- Benchmarks chiffrés (Qwen 14B 28.92 tok/s, etc.)

Si manquant → ajoute. Si présent mais désynchronisé → mets à jour.

```bash
git add README.md
git commit -m "[P13.2] README hub: version, links, latest benchmarks"
```

## P13.3 — INSTALL_MAC / INSTALL_WINDOWS / fix_windows.bat

```bash
head -30 INSTALL_MAC.md INSTALL_WINDOWS.md
```

Vérifie :
- Les versions Python supportées sont à jour (3.10+ probablement).
- Les commandes pip sont correctes.
- Les sections "Troubleshooting" mentionnent les pièges connus (BnB multi-GPU, etc.).

```bash
git add INSTALL_*.md
git commit -m "[P13.3] INSTALL_MAC/WINDOWS: update python versions + troubleshooting"
```

## P13.4 — CHANGELOG.md

```bash
head -30 CHANGELOG.md
```

Ajoute une entrée v1.5.x (futur) avec un placeholder :

```markdown
## [Unreleased] — V4 plan (sera 1.5.1 ou 1.6.0)

### Added
- (selon décisions P2-P5)

### Fixed
- Honnêteté docs V3 (P1)
- Stubs formalisés (P6)

### Changed
- TECHNICAL_DEBT.md mis à jour
```

```bash
git add CHANGELOG.md
git commit -m "[P13.4] CHANGELOG: Unreleased section for V4 changes"
```

---

# P14 — Hygiène repo (fichiers root)

L'arbo racine contient des fichiers louches.

## P14.1 — Inventaire

```bash
ls -la
ls -la _test_kernel.py "=0.43.0" mac mac_mlx mac_echo_backup mac_worker.py 2>/dev/null
ls bench_*.json bench_*.txt 2>/dev/null | wc -l
```

## P14.2 — Triage

| Fichier | Action |
|---------|--------|
| `=0.43.0` | Probablement un fichier accidentel d'un `pip install =0.43.0`. **Supprime.** |
| `_test_kernel.py` | Stub test au niveau racine. **Déplace dans `_deprecated/` ou `tests/`** selon usage. |
| `mac` (11963 octets, avril 6) | Probablement un binaire ou script orphelin. Inspecte (`file mac`), puis décision. |
| `mac_mlx` | Idem |
| `mac_echo_backup` | Backup, **déplace dans `_deprecated/`**. |
| `mac_worker.py` | Worker mac actif ? `grep -rn "mac_worker" --include="*.py"`. |
| `bench_*.json/txt` racine | Ce sont les résultats des benchmarks V0-V3. **Garde** mais documente dans un `benchmarks/RESULTS_INDEX.md`. |

```bash
# Suppression du fichier accidentel
rm "=0.43.0"

# Déplacement backups vers _deprecated
git mv mac_echo_backup _deprecated/mac_echo_backup 2>/dev/null || mv mac_echo_backup _deprecated/

# Stub test
mkdir -p _deprecated
git mv _test_kernel.py _deprecated/_test_kernel.py 2>/dev/null || mv _test_kernel.py _deprecated/

# Inspecte mac/mac_mlx
file mac mac_mlx 2>/dev/null
# Si binaires non utilisés -> _deprecated/

git add -A
git commit -m "[P14.2] hygiene: remove =0.43.0 typo, move mac_echo_backup + _test_kernel.py to _deprecated"
```

## P14.3 — Index des benchmarks

Crée `benchmarks/RESULTS_INDEX.md` :

```markdown
# Index des résultats benchmarks (racine du repo)

| Fichier | Date | Description |
|---------|------|-------------|
| bench_gpt2_out.txt | YYYY-MM | GPT-2 baseline |
| bench_tinyllama_out.txt | ... | ... |
| bench_qwen7b_out.txt | ... | ... |
| bench_14b_bigpu_turboquant.json | ... | Qwen 14B 2-GPU TurboQuant |
| ... | | |
| bench_kv_migration.json | 2026-04 | KV cache migration impact |
| bench_p2p_impact.json | 2026-04 | P2P bandwidth + latency |
| bench_wan_4g_jeje1_synology_me.json | 2026-04 | WAN 4G test |
```

(Remplis pour chaque fichier en regardant son contenu si nécessaire.)

```bash
git add benchmarks/RESULTS_INDEX.md
git commit -m "[P14.3] add benchmarks/RESULTS_INDEX.md — catalog root bench_* files"
```

## P14.4 — `.gitignore` durci

Vérifie `.gitignore` :
```bash
cat .gitignore | head -30
```

Ajoute si manquant :
```
# Builds Rust
rust_core/target/
*.so

# Caches python
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/

# Models / datasets locaux
models/
*.safetensors
*.gguf

# Bench temporaire
/tmp_bench/

# Editor
.vscode/
.idea/
*.swp
```

```bash
git add .gitignore
git commit -m "[P14.4] harden .gitignore (Rust target, python cache, models, editor)"
```

---

# P15 — Mise à jour TECHNICAL_DEBT.md

## P15.1 — Refresh

Réécris `docs/reports/TECHNICAL_DEBT.md` proprement avec :

1. Stubs réels actuels (déduire de P3, P4, P6).
2. Stubs résolus en V4 :
   - TRITON_SAMPLING_TOPK (P3) — résolu ou laissé selon décision
   - CONTINUOUS_BATCHER_NO_BATCHING (P4) — ouvert ou résolu
   - TRANSFER_MANAGER_LABEL_INCORRECT (P1.4) — ouvert ou résolu
3. Limitations connues (BnB multi-GPU, CUDA Graph, GIL, etc.).
4. Date dernière maj : 2026-05-XX (P15.1).

```bash
git add docs/reports/TECHNICAL_DEBT.md
git commit -m "[P15.1] refresh TECHNICAL_DEBT.md with V4 outcomes"
```

---

# P16 — Validation finale

## P16.1 — Tests complets

```bash
source .venv/bin/activate
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 VRM_BACKEND_ALLOW_STUB=1 \
  pytest tests/ --ignore=tests/test_chaos_concurrency.py --tb=short --no-cov 2>&1 | tail -10
```

**Cible :** 1 failed pré-existante, passed ≥ baseline + nouveaux tests P6.1, P6.3, P8.2, P8.3.

Si régression : revert le commit fautif.

## P16.2 — Smoke test

```bash
VRM_MINIMAL_TEST=1 python -m tests.smoke 2>&1 | tail -5
```

Cible : exit 0.

## P16.3 — Sanity GPU end-to-end

```bash
source .venv/bin/activate
python -c "
from core.inference_pipeline import InferencePipeline
p = InferencePipeline(backend_name='huggingface', verbose=False)
p.load('gpt2', num_gpus=1)
out = p.generate('The capital of France is', max_new_tokens=10)
print('OUT:', repr(out))
assert 'paris' in out.lower(), f'Expected Paris, got: {out}'
print('SANITY OK')
"
```

Cible : "SANITY OK".

## P16.4 — flake8 / ruff

```bash
flake8 core/ tests/ benchmarks/ 2>&1 | tail -20 || true
```

Si nouveaux warnings introduits par toi : corrige. Si warnings préexistants : laisse.

```bash
git commit --allow-empty -m "[P16] validation finale: tests OK, smoke OK, sanity OK, lint clean"
```

---

# P17 — Documentation finale

## P17.1 — Compléter `resultat_v4.md`

Toutes les sections [P1] à [P16] doivent contenir des chiffres réels.

Ajoute la section [SUMMARY] :

```markdown
## [SUMMARY]

**Date fin :** YYYY-MM-DD HH:MM
**Durée totale :** XXh

**Commits V4 :**
$(git log main..HEAD --oneline)

**Fichiers modifiés :**
$(git diff --stat main..HEAD)

**Tests :**
- Baseline: 1 failed, 1070 passed, 39 skipped
- Final:    1 failed, NNNN passed, 39 skipped
- Régression: AUCUNE (ou: <description>)

**Performance :**
- Qwen 14B 2-GPU CUDA Stream Overlap: <gain Z.Z% MERGED | REVERT (gain insufficient)>
- Triton sampling top-k: <résolu | non-stub | gain Z.Z%>
- ContinuousBatcher diagnostic: case <A|B|C>

**vLLM comparison :**
- Qwen 7B 1-GPU: VRAMancer XX vs vLLM YY tok/s (XX% de vLLM)
- Qwen 14B 2-GPU hetero: vLLM <accepted XX | refused>

**Stubs résolus :**
- TRITON_SAMPLING_TOPK: <résolu en P3 | reste stub>
- TRANSFER_MANAGER_LABEL: <résolu | documenté>
- VTP_L3, DMABUF, NAT: skip tests propres ajoutés

**Hygiène :**
- =0.43.0 typo supprimé
- mac_echo_backup, _test_kernel.py → _deprecated/
- benchmarks/RESULTS_INDEX.md ajouté
- .gitignore durci

**Documentation :**
- README, INSTALL_MAC, INSTALL_WINDOWS harmonisés v1.5.0
- CHANGELOG.md Unreleased section
- TECHNICAL_DEBT.md refresh
- 4 corrections honnêteté V3 (annotations dans resultat_v3.md + REBAR_PROXMOX_BENCHMARK.md)

**Verdict global V4 :** [SUCCESS / PARTIAL / NEEDS_REVISION]

**Reste à faire (V5 candidat) :**
- (selon ce qui n'a pas pu être fait)
```

```bash
git add resultat_v4.md
git commit -m "[P17.1] resultat_v4.md final — execution log V4 master"
```

## P17.2 — Mise à jour `.github/copilot-instructions.md`

Mets à jour :
1. Section "Variables d'environnement" : `VRM_TRANSFER_OVERLAP`, `VRM_DEBUG_SAMPLING`.
2. Section "Benchmarks reels" : ajoute la ligne CUDA Stream Overlap si MERGE.
3. Section "Pieges connus" : si nouveaux bugs documentés en V4 (CONTINUOUS_BATCHER_NO_BATCHING par ex), les ajouter.
4. Section "Resume honnetete globale" : rééquilibrer si nécessaire (pourcentages).

```bash
git add .github/copilot-instructions.md
git commit -m "[P17.2] update copilot-instructions: V4 outcomes (overlap, sampling, batcher diag)"
```

---

# Annexes

## ANNEXE A — Résolution de blocages

| Symptôme | Action |
|----------|--------|
| `ModuleNotFoundError: torch` | `source .venv/bin/activate` |
| OOM en chargeant Qwen 14B | `nvidia-smi` (kill processus orphelins), reboot si nécessaire |
| Test échoue après une modif | `git diff` puis `git checkout <fichier>` pour revert ciblé |
| Bench bruité (>10% stddev) | Lance 5x, prends median. Si toujours bruité : "INDÉTERMINÉ" |
| Quelqu'un a modifié des fichiers entre 2 phases | `git diff` voir étendue. Si compatible, continue. Sinon STOP. |
| 2 échecs successifs sur la même tâche | STOP. `[BLOCKED@P<x>.<y>]` dans `resultat_v4.md`. Demande. |
| vLLM install impossible | SKIP P5.2/P5.3 + entry TECHNICAL_DEBT, passe à P6 |
| GPU bench tourne avec mauvais ordre | Vérifie `CUDA_VISIBLE_DEVICES`. PCI_BUS_ID order vs torch FAST_FIRST. |
| Rust build échoue | Skip rust-related P9 jobs, signale dans resultat_v4.md |

## ANNEXE B — Fichiers que tu PEUX modifier

```
resultat_v3.md (annotations P1.1, P1.2, P1.3 uniquement)
resultat_v4.md (création + édition continue)
docs/reports/REBAR_PROXMOX_BENCHMARK.md (P1.2)
docs/reports/STREAM_OVERLAP_AUDIT.md (création P2.1)
docs/reports/TECHNICAL_DEBT.md (P1.4, P3, P4, P5.1, P15)
docs/reports/CUDA_GRAPH_MULTI_GPU_AUDIT.md (P6.6 si touche)
core/transfer_manager.py (P2.3, P2.5)
core/triton_sampling.py (P3.1)
core/cuda_graph_decode.py (P6.6 docstring)
core/inference_pipeline.py (P3.4 si applicable, P7.3)
core/telemetry.py (P7.1 — éventuel déplacement)
core/swarm_ledger.py (P7.2 — éventuel déplacement)
csrc/dmabuf_bridge.c (P6.2 header)
benchmarks/bench_stress_concurrent_v4.py (création P4.2)
benchmarks/RESULTS_INDEX.md (création P14.3)
tests/test_transfer_manager_basic.py (création P8.2)
tests/test_triton_sampling_paths.py (création P8.3)
tests/test_vtp_l3_stub.py (création P6.1)
tests/test_nat_traversal_stub.py (création P6.3)
dashboard/dashboard_web.py (P10)
dashboard/templates/* dashboard/static/* (P10)
examples/*.py (P11 — headers)
requirements*.txt (P12 — headers)
README.md (P13.2)
INSTALL_MAC.md INSTALL_WINDOWS.md (P13.3)
CHANGELOG.md (P13.4)
.gitignore (P14.4)
.github/copilot-instructions.md (P17.2)
.github/workflows/*.yml (P9.1 si manque)
.pre-commit-config.yaml (création P9.2)
```

## ANNEXE C — Fichiers que tu NE PEUX PAS modifier

```
_deprecated/* (ne touche pas le contenu existant ; tu peux y AJOUTER via mv)
tests/test_chaos_concurrency.py
core/security/__init__.py
core/security/startup_checks.py
core/paged_attention.py
core/paged_attention_cuda.py
csrc/paged_attention_kernel.cu
rust_core/src/* (sauf instruction explicite)
pyproject.toml setup.cfg (sauf P13.1 version bump si désynchronisé)
Tous les bench_*.json bench_*.txt à la racine (résultats V0-V3 figés)
Tous les fichiers de config/ (sauf instruction explicite)
```

## ANNEXE D — Checklist d'acceptation V4

Le plan V4 est **complet** si :

- [ ] `chore/sonnet-plan-v4` existe et est cohérente
- [ ] **P0** : baseline enregistrée
- [ ] **P1** : 4 corrections honnêteté commitées
- [ ] **P2** : décision MERGE ou REVERT documentée avec chiffres
- [ ] **P3** : décision documentée + TECHNICAL_DEBT mis à jour
- [ ] **P4** : `bench_stress_concurrent_v4.py` créé + verdict A/B/C choisi
- [ ] **P5** : SKIP justifié OU 2 tableaux comparatifs
- [ ] **P6** : 6 stubs adressés (formaliser ou fixer)
- [ ] **P7** : telemetry + swarm_ledger triés
- [ ] **P8** : 2 tests minimum ajoutés
- [ ] **P9** : CI vérifié + pre-commit (optionnel) ajouté
- [ ] **P10** : dashboard hardcodes remplacés ou documentés
- [ ] **P11** : examples annotés + obsolètes déplacés
- [ ] **P12** : requirements files annotés
- [ ] **P13** : versions harmonisées + README/INSTALL/CHANGELOG mis à jour
- [ ] **P14** : =0.43.0 supprimé, mac_echo_backup déplacé, .gitignore durci
- [ ] **P15** : TECHNICAL_DEBT.md à jour
- [ ] **P16** : tests verts, smoke verts, sanity OK
- [ ] **P17** : `resultat_v4.md` complet avec [SUMMARY], copilot-instructions à jour
- [ ] Aucune régression de tests
- [ ] Aucun fichier `_deprecated/` modifié dans son contenu pré-existant
- [ ] Aucun test désactivé pour faire passer la suite
- [ ] Aucun `git push` effectué
- [ ] Tous les commits préfixés `[P<x>.<y>]`

## ANNEXE E — Anti-patterns absolus

❌ Ajouter des features non listées dans ce plan
❌ Réécrire `_deprecated/` files (tu peux y MOVE depuis ailleurs, pas modifier le contenu existant)
❌ Modifier les chiffres V3 (on annote, on ne réécrit pas l'histoire)
❌ Désactiver des tests pour faire passer la suite
❌ Marquer "MERGE" si gain < 3%
❌ Prétendre qu'un bench s'est exécuté sans logs/sortie réels (tee dans /tmp/)
❌ Committer des scripts `/tmp/bench_*.py` dans le repo (sauf explicitement listés en Annexe B)
❌ Brute-forcer 3+ fois après échec
❌ Modifier `rust_core/src/` (sauf instruction explicite — ce plan n'en a pas)
❌ `git push`, `git push --force`, ouvrir PR
❌ Oublier `source .venv/bin/activate`
❌ Mélanger les ordres GPU (PCI vs torch.cuda) dans les rapports
❌ Sauter une phase parce qu'elle "semble inutile"
❌ Réordonner les phases sans raison documentée
❌ Commit avec message vide ou non préfixé
❌ Ajouter de nouvelles dépendances pip sans justification dans le commit message
❌ Toucher au scoring sécurité ou auth (P0 hors scope V4)

---

## ANNEXE F — Estimation effort par phase (pour ton planning)

| Phase | Min | Max | Notes |
|-------|-----|-----|-------|
| P0    | 5m  | 10m | rapide |
| P1    | 30m | 45m | que du texte |
| P2    | 1h  | 3h  | dépend du gain (revert plus rapide) |
| P3    | 1h  | 2h  | instrumentation + bench |
| P4    | 45m | 1h  | script + 2 runs |
| P5    | 30m | 2h  | install vLLM peut être long |
| P6    | 1h  | 1h30 | tests skip + headers |
| P7    | 30m | 45m | grep + mv |
| P8    | 1h  | 1h30 | écrire 2 tests |
| P9    | 30m | 45m | YAML CI |
| P10   | 45m | 1h30 | front + back |
| P11   | 30m | 1h  | tester examples |
| P12   | 30m | 30m | headers fichiers |
| P13   | 1h  | 1h30 | doc cohérence |
| P14   | 20m | 30m | nettoyage root |
| P15   | 15m | 20m | écriture |
| P16   | 15m | 30m | validation |
| P17   | 30m | 45m | résumé final |

**Total estimé :** 10-18h équivalent humain (selon échec/succès des decisions binaires).

---

## ANNEXE G — Ordre de priorité si manque de temps

Si tu ne peux pas tout faire, **respecte cet ordre absolu** :

1. **P0** (toujours)
2. **P1** (corrections honnêteté — important pour traçabilité)
3. **P16** (validation finale — toujours faire avant de s'arrêter)
4. **P17.1** (résultat_v4.md, même partiel)

Puis selon priorité :
5. **P15** (TECHNICAL_DEBT à jour)
6. **P14** (hygiène repo — vite fait)
7. **P6** (stubs formalisés)
8. **P4** (diagnostic batcher — info importante)
9. **P2** (perf win, mais reversible)
10. **P8** (tests coverage)
11. **P12, P13** (doc — peut attendre)
12. **P3** (perf optionnel)
13. **P5** (vLLM — peut SKIP)
14. **P7, P9, P10, P11** (polish secondaire)

**Si tu t'arrêtes en cours de route** : commit `[CHECKPOINT]` ce qui est fait + `resultat_v4.md` partiel + liste claire des phases restantes en `[TODO_NEXT_SESSION]`.

---

**Fin du plan V4 master. Bonne exécution. Honnêteté > marketing. Mesures > impressions.**
