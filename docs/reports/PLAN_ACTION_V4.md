# Plan V4 — Polish honnêteté + CUDA Stream Overlap + vLLM bench + Stress diagnostic

> **Pour l'agent exécutant :** ce plan est conçu pour être suivi à la lettre, étape par étape.
> Lis chaque section dans l'ordre. Ne saute jamais une validation.
> Après chaque tâche numérotée (V0.1, V1.1…), tu dois faire un commit ATOMIQUE.
> Si une tâche échoue, ARRÊTE-TOI et demande à l'utilisateur — ne brute-force pas.
> Si une mesure ne montre pas le gain attendu (≥3%), tu fais REVERT et tu documentes — tu n'arrondis pas, tu ne mens pas.

**Version :** v4.0
**Date :** 5 mai 2026
**Branche source :** `main` après merge V3 (HEAD = `fed3fce`)
**Branche cible :** `chore/sonnet-plan-v4`
**Auteur :** Architecte Claude Opus 4.7
**Exécutant attendu :** Claude Sonnet 4.6 (ou tout agent moins capable suivant pas-à-pas)

---

## Vision

Quatre objectifs **mesurables** :

1. **Polish honnêteté** — corriger 4 inexactitudes mineures dans la doc V3 (labels GPU, attribution +382%, méthodo P6.1, label TransferManager).
2. **Performance win** — Implémenter CUDA Stream Overlap dans `TransferManager.send_activation()` pour pipeliner compute // transfer P2P. Cible **≥+3% sur Qwen 14B 2-GPU BF16**.
3. **Calibrage externe** — Installer vLLM et benchmarker Qwen 7B + 14B. **Documenter le gap honnêtement**, sans embellir.
4. **Diagnostic stress** — Comprendre pourquoi V3 P6.1 a montré n=4 (42.9 tok/s) **<** n=1 (74.6 tok/s). Soit le batcher ne bat pas, soit le test V3 était mal écrit. Trancher.

**Hors-scope V4 :** nouvelles features, nouveaux backends, nouvelles routes API, refactor majeur, dashboard.

---

## Règles globales (ABSOLUES, valables pour TOUTES les tâches)

1. **NE JAMAIS** modifier ces fichiers/dossiers :
   - `_deprecated/`
   - `tests/test_chaos_concurrency.py`
   - `csrc/paged_attention_kernel.cu`
   - `core/security/__init__.py`, `core/security/startup_checks.py`
   - `core/paged_attention.py`
   - `core/paged_attention_cuda.py`
   - `rust_core/src/` (sauf instruction explicite)
2. **NE JAMAIS** push, merge, créer une PR, ni faire `git push --force`.
3. **NE JAMAIS** désactiver des tests existants pour faire passer la suite. Si un test casse → fix le code, pas le test.
4. **NE JAMAIS** modifier rétroactivement les chiffres de `resultat_v3.md` ou `bench_*.json` existants. Si un chiffre V3 est faux, on l'**annote** dans `resultat_v4.md`, on ne le réécrit pas.
5. **NE JAMAIS** committer un script de benchmark temporaire dans le repo (sauf si demandé). Garde-le dans `/tmp/` ou supprime-le après.
6. **TOUJOURS** committer atomiquement (1 tâche = 1 commit), avec préfixe `[V<x>.<y>]`.
7. **TOUJOURS** valider la suite complète (V5.1) avant de signaler "terminé".
8. **TOUJOURS** copier les nombres bruts (tok/s, tailles VRAM, exit codes) dans `resultat_v4.md`. Pas de "ça marche bien".
9. **TOUJOURS** sourcer le venv : `source .venv/bin/activate`. Sinon `ModuleNotFoundError: torch`.
10. **TOUJOURS** lancer les benchmarks GPU avec `CUDA_VISIBLE_DEVICES` explicite si tu cibles un seul GPU. Sinon torch utilise FAST_FIRST par défaut (GPU0=RTX 3090, GPU1=RTX 5070 Ti dans cet environnement Proxmox).

---

## Préparation (UNE FOIS, avant V0)

```bash
cd /home/jeremie/VRAMancer/VRAMancer
source .venv/bin/activate
git status                     # doit être propre
git --no-pager log --oneline -1  # doit être fed3fce ou plus récent sur main
git checkout main
git checkout -b chore/sonnet-plan-v4
```

**Crée immédiatement `resultat_v4.md` avec ce squelette :**

```markdown
# Résultat Plan V4 — Polish + CUDA Stream Overlap + vLLM bench + Stress diagnostic

**Date début :** YYYY-MM-DD HH:MM
**Branche :** chore/sonnet-plan-v4
**Plan :** docs/reports/PLAN_ACTION_V4.md
**Base :** main @ fed3fce

## [BASELINE]
(à remplir maintenant — voir ci-dessous)

## [V0.x] — Polish honnêteté
## [V1.x] — CUDA Stream Overlap
## [V2.x] — Benchmark vs vLLM
## [V3.x] — Diagnostic stress
## [V4.x] — Triton sampling audit
## [V5.x] — Tests
## [SUMMARY]
```

**Baseline tests à enregistrer dans `[BASELINE]` :**

```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 VRM_BACKEND_ALLOW_STUB=1 \
  pytest tests/ --ignore=tests/test_chaos_concurrency.py --tb=no --no-cov 2>&1 | tail -3
```

**Cible attendue baseline :** `1 failed, 1070 passed, 39 skipped` (1 failed pré-existant `test_health_imports_fault_manager`).

**Si tu obtiens autre chose (ex: 2 failed) :** ARRÊTE-TOI et signale à l'utilisateur. La branche `main` n'est pas dans l'état attendu.

**Baseline GPU à enregistrer :**

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python -c "import torch; [print(f'torch GPU{i}: {torch.cuda.get_device_properties(i).name}') for i in range(torch.cuda.device_count())]"
```

Note les deux ordres dans `[BASELINE]` (PCI vs torch).

---

# P0 — Polish honnêteté (4 corrections mineures)

L'audit V3 a relevé 4 inexactitudes dans la doc V3. Ce sont des corrections de **texte** uniquement, pas de code. Effort total : ~30 min.

## V0.1 — Annoter les labels GPU dans `resultat_v3.md`

**Problème :** la section V2.1 utilise `nvidia-smi` ordering (PCI_BUS_ID : GPU0=RTX 5070 Ti, GPU1=RTX 3090). La section V2.2 et suivantes utilisent `torch.cuda` ordering (FAST_FIRST : GPU0=RTX 3090, GPU1=RTX 5070 Ti). Aucune mention. Un lecteur peut conclure à tort que les RTX 3090 et 5070 Ti ont été inversées entre les sections.

**Action :**

1. Ouvrir `resultat_v3.md`.
2. Dans la section V2.1 (juste après `## [V2.1]` ou équivalent), insérer un bloc note :

```markdown
> **Note ordering GPU :**
> - `nvidia-smi` → ordre `PCI_BUS_ID` : GPU0=RTX 5070 Ti, GPU1=RTX 3090
> - `torch.cuda` → ordre `FAST_FIRST` (par défaut) : GPU0=RTX 3090, GPU1=RTX 5070 Ti
> Les sections benchmarks suivantes (V2.2+) utilisent l'ordre `torch.cuda`.
> Référence : https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-environment-variables
```

3. **Ne modifie aucun chiffre.** Juste l'annotation.

**Validation :**

```bash
grep -n "PCI_BUS_ID\|FAST_FIRST" resultat_v3.md
# Doit afficher la note ajoutée
```

**Commit :**

```bash
git add resultat_v3.md
git commit -m "[V0.1] annotate GPU ordering (nvidia-smi PCI vs torch.cuda FAST_FIRST) in resultat_v3.md"
```

---

## V0.2 — Réattribuer le delta +382% sur Qwen 14B

**Problème :** `docs/reports/REBAR_PROXMOX_BENCHMARK.md` et `resultat_v3.md` attribuent le speedup +382% (28.92 vs 6.0 tok/s) au "ReBAR + Rust P2P bypass" sans distinction. La réalité est que ReBAR seul donne ~5–15%, le gros vient du Rust P2P bypass qui remplace CPU staging par `cudaMemcpyPeerAsync` direct.

**Action :**

1. Ouvrir `docs/reports/REBAR_PROXMOX_BENCHMARK.md`.
2. Trouver la section qui mentionne `+382%` (probablement vers la fin, "Conclusion" ou "Analyse").
3. **Remplacer** le bloc d'attribution par :

```markdown
### Attribution du speedup +382% (28.92 vs 6.0 tok/s baseline mars 2026)

**Composantes estimées (à confirmer expérimentalement en V4 P5 — voir plus bas) :**

| Source | Estimation gain | Justification |
|--------|----------------|---------------|
| Rust P2P bypass `send_to_device` ≥512 KB | ~70-80% | Remplace CPU staging (Strategy 4) par `cudaMemcpyPeerAsync` direct. Mars 2026 IOMMU bloquait P2P. |
| ReBAR (BAR1 ≥ VRAM) | ~10-15% | Réduit latence accès, prefetch plus efficace. |
| Optimisations diverses | ~5-10% | accelerate dispatch natif, KV cache pages distribuées, version stack. |

**Validation expérimentale recommandée (non faite en V3, pourrait être faite en V5) :**
1. Désactiver `VRM_RUST_P2P=0` et re-bencher Qwen 14B 2-GPU BF16.
2. Désactiver ReBAR au niveau BIOS et re-bencher (lourd, nécessite reboot).

**Conclusion honnête :** la décomposition exacte n'est pas mesurée à ce jour.
Le delta +382% est réel mais l'attribution est une estimation basée sur la nature des changements.
```

4. Ouvrir `resultat_v3.md`. Trouver la phrase qui dit "+382% grâce à ReBAR" (ou similaire). **Ne pas la supprimer.** Ajouter juste après :

```markdown
> **Note d'attribution (ajoutée V4 P0) :** Le delta +382% est réel.
> L'attribution exacte (ReBAR seul vs Rust P2P bypass vs autres) n'a pas été mesurée
> indépendamment en V3. Voir `docs/reports/REBAR_PROXMOX_BENCHMARK.md` section "Attribution"
> pour l'estimation détaillée.
```

**Validation :**

```bash
grep -A3 "Attribution du speedup" docs/reports/REBAR_PROXMOX_BENCHMARK.md
grep -A2 "Note d'attribution" resultat_v3.md
```

**Commit :**

```bash
git add docs/reports/REBAR_PROXMOX_BENCHMARK.md resultat_v3.md
git commit -m "[V0.2] honest attribution of +382% speedup: ReBAR ~15% + Rust P2P bypass ~75% + misc ~10%"
```

---

## V0.3 — Corriger l'explication P6.1 n=1 > sequential

**Problème :** `resultat_v3.md` section P6.1 dit que n=1 (74.6 tok/s) > sequential (30.8 tok/s) à cause du "warmup actif". Faux : c'est le KV cache du batcher qui reste chaud entre runs, pas un warmup classique.

**Action :**

1. Ouvrir `resultat_v3.md`.
2. Trouver la section P6.1 "Stress test" qui mentionne `74.6 tok/s` et l'explication "warmup".
3. Ajouter juste après l'explication erronée :

```markdown
> **Correction méthodologique (V4 P0) :** L'explication "warmup actif" pour justifier
> n=1 (74.6) > séquentiel (30.8) est inexacte. Diagnostic réel :
> - Le baseline `Sequential 1 req` mesure `prefill + decode` à froid (cold KV cache).
> - Le test `n=1 concurrent` arrive APRÈS le baseline → KV cache batcher déjà chaud,
>   tokenizer HF chaud, pages VRAM stables.
> - Comparaison apples-to-apples requise = `n=1` vs séquentiel APRÈS warmup identique.
> - À reprendre proprement en V4 P3 (voir bench_stress_concurrent_v4.py).
```

4. **Ne supprime pas** l'explication originale. On annote, on ne réécrit pas l'histoire.

**Validation :**

```bash
grep -A5 "Correction méthodologique" resultat_v3.md
```

**Commit :**

```bash
git add resultat_v3.md
git commit -m "[V0.3] honest methodology note on P6.1 n=1 > sequential (KV cache hot, not warmup)"
```

---

## V0.4 — Vérifier label `_get_method_for` dans TransferManager

**Problème suspecté :** Pendant les benchmarks V3, certains logs auraient montré "CPU_STAGED" alors que le P2P fonctionnait. L'audit doit vérifier si c'est un vrai bug ou une illusion.

**Action :**

1. Lire `core/transfer_manager.py` autour de la ligne 930 (méthode `_get_method_for`).
2. Vérifier la logique :
   ```python
   def _get_method_for(self, src: int, dst: int) -> str:
       if cross_vendor: return "CROSS_VENDOR:..."
       if self._can_p2p(src, dst): return "CUDA_P2P"
       if self._nccl_initialized: return "NCCL"
       return "CPU_STAGED"
   ```
3. Lire `_can_p2p()` (chercher `def _can_p2p`).
4. **Test concret** :

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
    print(f'_can_p2p(1,0): {tm._can_p2p(1, 0)}')
else:
    print('Skip: <2 GPUs')
"
```

5. **Deux cas possibles :**

   **Cas A : `Method 0->1: CUDA_P2P`** → tout va bien, pas de bug. Action : juste documenter dans `resultat_v4.md` que la vérif a été faite et rien à changer. Commit avec message `[V0.4] verify TransferManager labels: OK, no bug`.

   **Cas B : `Method 0->1: CPU_STAGED` mais P2P fonctionne en réalité** → vrai bug. Action : ne pas tenter de fix maintenant. Ouvrir `docs/reports/TECHNICAL_DEBT.md` et ajouter une section :

```markdown
### TRANSFER_MANAGER_LABEL_INCORRECT
**Détecté :** V4 P0.4 (mai 2026)
**Symptôme :** `_get_method_for(src, dst)` retourne "CPU_STAGED" alors que `send_activation()` utilise effectivement le path P2P.
**Cause suspectée :** `_can_p2p()` retourne False mais le path Rust bypass court-circuite cette vérif.
**Impact :** cosmétique (logs trompeurs), pas fonctionnel.
**Fix futur :** aligner `_can_p2p` avec le runtime réel ou retirer le label "CPU_STAGED".
```

6. Dans les deux cas, **commit** :

```bash
# Cas A:
git commit --allow-empty -m "[V0.4] verify TransferManager._get_method_for labels: correct, no bug"

# Cas B:
git add docs/reports/TECHNICAL_DEBT.md
git commit -m "[V0.4] document TRANSFER_MANAGER_LABEL_INCORRECT in TECHNICAL_DEBT.md"
```

**Validation :** dans `resultat_v4.md`, écris **explicitement** lequel des deux cas tu as observé, avec la sortie brute du script Python ci-dessus.

---

# P1 — Performance : CUDA Stream Overlap dans TransferManager

**Hypothèse :** `send_activation()` actuellement synchronise implicitement entre src et dst (via `tensor.to(device)`). Si on lance le `cudaMemcpyPeerAsync` sur un stream non-default côté src, le compute du bloc N+1 peut commencer pendant que le transfer du bloc N est en transit. Cible : **≥+3% sur Qwen 14B 2-GPU BF16** (28.92 → ≥29.8 tok/s).

**Risques :**
- Race condition (output corrompu si on lit avant que le transfer finisse)
- Pas de gain si le compute est déjà beaucoup plus long que le transfer
- Complexité : torch.cuda.Stream cross-device a des subtilités (event recording sur src, wait sur dst)

**Stratégie :** mesure d'abord, code ensuite. Si pas de gain → REVERT et documenter.

## V1.1 — Audit `send_activation()` et points de synchronisation

**Action :**

1. Lire `core/transfer_manager.py` :
   - Méthode `send_activation()` (ligne ~371)
   - Méthode `_send_p2p()` ou équivalent (path CUDA_P2P)
   - Tous les appels à `torch.cuda.synchronize`, `.synchronize()`, `wait_stream`

```bash
grep -n "synchronize\|wait_stream\|Stream\|Event" core/transfer_manager.py
```

2. Lire `core/inference_pipeline.py` :
   - Toutes les utilisations de `transfer_mgr.send_activation` (s'il y en a)
   - Ou plus généralement les transferts inter-GPU dans `_forward_block` ou équivalent

```bash
grep -n "send_activation\|to(.cuda\|\.to(f.cuda" core/inference_pipeline.py
```

3. Créer `docs/reports/STREAM_OVERLAP_AUDIT.md` avec le résumé :
   - Combien d'appels à `send_activation` par token décodé sur Qwen 14B 2-GPU ?
   - Y a-t-il déjà un Stream non-default utilisé ?
   - Quels sont les sync points implicites ?

**Le doc d'audit doit faire entre 30 et 80 lignes.** Pas plus, pas moins.

**Validation :**

```bash
wc -l docs/reports/STREAM_OVERLAP_AUDIT.md  # entre 30 et 80
```

**Commit :**

```bash
git add docs/reports/STREAM_OVERLAP_AUDIT.md
git commit -m "[V1.1] audit send_activation sync points and stream usage"
```

---

## V1.2 — Benchmark BASELINE Qwen 14B 2-GPU (avant modif)

**Action :**

1. Créer un script `/tmp/bench_v1_baseline.py` (PAS dans le repo) :

```python
import os, time, torch
os.environ['VRM_QUANTIZATION'] = ''  # BF16
from core.inference_pipeline import InferencePipeline

print('=== V1.2 BASELINE (no stream overlap) ===')
pipe = InferencePipeline(backend_name='huggingface', verbose=False)
pipe.load('Qwen/Qwen2.5-14B', num_gpus=2)

prompt = 'Explain quantum entanglement in one paragraph.'
pipe.generate(prompt, max_new_tokens=20)  # warmup

times = []
for i in range(5):
    t0 = time.perf_counter()
    pipe.generate(prompt, max_new_tokens=200)
    dt = time.perf_counter() - t0
    ts = 200 / dt
    times.append(ts)
    print(f'  Run {i+1}: {ts:.2f} tok/s')

import statistics
median = statistics.median(times)
mean = sum(times) / len(times)
print(f'\nMedian: {median:.2f} tok/s')
print(f'Mean:   {mean:.2f} tok/s')
print(f'StdDev: {statistics.stdev(times):.2f}')
```

2. Lancer :

```bash
source .venv/bin/activate
python /tmp/bench_v1_baseline.py 2>&1 | tee /tmp/bench_v1_baseline.txt
```

3. Copier les 5 runs + median/mean/stddev dans `resultat_v4.md` section V1.2.

4. **Note** : si le median diffère de >10% du chiffre V3 (28.92 tok/s), c'est suspect. Re-bencher après reboot.

**PAS de commit** pour cette étape (rien dans le repo).

---

## V1.3 — Implémenter CUDA Stream Overlap dans `send_activation()`

**Précautions :**
- Modification minimale. **Pas de refactor large**.
- Si le path actuel utilise déjà un stream non-default → SKIP V1.3 et documenter dans `resultat_v4.md` (le gain n'est pas accessible par cette voie).

**Patch attendu (à adapter selon V1.1 audit) :**

Dans `core/transfer_manager.py`, ajouter dans `__init__` :

```python
# CUDA Stream pool pour overlap compute / transfer
self._transfer_streams = {}  # {src_gpu: torch.cuda.Stream}
```

Ajouter une méthode :

```python
def _get_transfer_stream(self, src_gpu: int) -> "torch.cuda.Stream":
    """Lazy-init un stream dédié transfer pour src_gpu."""
    if src_gpu not in self._transfer_streams:
        if not _HAS_TORCH or not torch.cuda.is_available():
            return None
        with torch.cuda.device(src_gpu):
            self._transfer_streams[src_gpu] = torch.cuda.Stream(priority=-1)
    return self._transfer_streams[src_gpu]
```

Dans le path P2P de `send_activation` (à identifier en V1.1), wrapper le transfer dans le stream :

```python
stream = self._get_transfer_stream(src_gpu)
if stream is not None:
    with torch.cuda.stream(stream):
        dst_tensor = tensor.to(f'cuda:{dst_gpu}', non_blocking=True)
    # IMPORTANT: si on retourne dst_tensor, il faut s'assurer que le caller
    # le synchronise avant lecture. Simple sécurité : event sur stream.
    event = torch.cuda.Event()
    event.record(stream)
    # Le caller peut wait via:  torch.cuda.current_stream().wait_event(event)
```

**ATTENTION race condition :** si le caller lit `dst_tensor` immédiatement sans wait, c'est un crash garanti (read-before-write).

**Stratégie sûre :** si le caller actuel ne sait pas attendre l'event, **NE PAS** activer l'overlap. Faire un flag `VRM_TRANSFER_OVERLAP=1` (off par défaut) et l'activer manuellement pour le bench.

```python
import os
self._overlap_enabled = os.environ.get('VRM_TRANSFER_OVERLAP', '0') == '1'
```

**Validation locale (avant commit) :**

```bash
# Test 1: tests baseline non régressés
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 VRM_BACKEND_ALLOW_STUB=1 \
  pytest tests/test_transfer_manager.py tests/test_pipeline.py --tb=short --no-cov 2>&1 | tail -5

# Test 2: smoke test
source .venv/bin/activate
python -c "
from core.transfer_manager import TransferManager
tm = TransferManager()
print('TransferManager OK')
print(f'_overlap_enabled = {tm._overlap_enabled}')
"
```

**Commit :**

```bash
git add core/transfer_manager.py
git commit -m "[V1.3] add VRM_TRANSFER_OVERLAP flag with CUDA Stream for P2P send_activation"
```

---

## V1.4 — Benchmark AVEC overlap activé

**Action :**

1. Créer `/tmp/bench_v1_overlap.py` — IDENTIQUE à `/tmp/bench_v1_baseline.py` mais avec :

```python
import os
os.environ['VRM_TRANSFER_OVERLAP'] = '1'
# (le reste identique)
```

2. Lancer :

```bash
source .venv/bin/activate
python /tmp/bench_v1_overlap.py 2>&1 | tee /tmp/bench_v1_overlap.txt
```

3. Comparer dans `resultat_v4.md` :

```markdown
### V1.4 — Benchmark CUDA Stream Overlap

| Run | Baseline (V1.2) | Overlap (V1.4) |
|-----|-----------------|----------------|
| 1   | XX.XX tok/s     | YY.YY tok/s    |
| 2   | XX.XX           | YY.YY          |
| 3   | XX.XX           | YY.YY          |
| 4   | XX.XX           | YY.YY          |
| 5   | XX.XX           | YY.YY          |
| **Median** | **XX.XX** | **YY.YY** |
| **Delta**  | — | **+Z.Z%** |

Verdict : MERGE / REVERT / INDÉCIS
```

4. **Décision binaire :**

   - **Si delta median ≥ +3%** → MERGE. Activer par défaut : changer `'0'` → `'1'` dans le default du flag, OU créer un commit `[V1.5] enable VRM_TRANSFER_OVERLAP by default`.

   - **Si delta median < +3%** → REVERT V1.3 :
     ```bash
     git revert HEAD --no-edit  # ou git reset --hard HEAD~1 si pas pushé
     ```
     Documenter dans `docs/reports/STREAM_OVERLAP_AUDIT.md` la conclusion : "gain insuffisant, le bottleneck n'est pas le sync transfer".

   - **Si delta < -2% (régression)** → REVERT obligatoire, et marquer le commit comme "experiment failed" dans le message de revert.

**PAS de commit pour le bench lui-même.** Le commit est soit V1.5 (enable by default), soit le revert.

---

## V1.5 — Si MERGE : enable by default

(Skip si REVERT en V1.4)

**Action :**

```python
# core/transfer_manager.py
self._overlap_enabled = os.environ.get('VRM_TRANSFER_OVERLAP', '1') == '1'  # default ON
```

**Mettre à jour `.github/copilot-instructions.md`** :
- Trouver la section variables d'env
- Ajouter `VRM_TRANSFER_OVERLAP` dans le tableau

**Commit :**

```bash
git add core/transfer_manager.py .github/copilot-instructions.md
git commit -m "[V1.5] enable VRM_TRANSFER_OVERLAP by default (gain +Z.Z% on Qwen 14B 2-GPU)"
```

---

# P2 — Benchmark vs vLLM (calibrage externe)

**Pourquoi :** VRAMancer est en compétition implicite avec vLLM. Sans benchmark direct, on n'a aucune idée de notre position. **Soyez honnête** : si vLLM est plus rapide, dites-le.

## V2.1 — Tenter d'installer vLLM

**Action :**

```bash
source .venv/bin/activate
pip show vllm 2>&1 | head -5  # vérifier si déjà installé
```

Si pas installé :

```bash
pip install vllm 2>&1 | tee /tmp/vllm_install.log
```

**Cas d'échec courants :**

1. **CUDA version mismatch** : vLLM a des wheels pré-compilés pour CUDA spécifiques.
   - Solution : `pip install vllm --index-url https://download.pytorch.org/whl/cu128` (à adapter)

2. **PyTorch version conflict** : vLLM peut exiger torch 2.4 ou 2.5 alors qu'on a torch 2.10+.
   - Solution : tester `pip install "vllm==0.6.x"` (ancienne version, compatible torch plus ancien).
   - **Si conflit insoluble** : SKIP V2.2/V2.3. Documenter clairement.

3. **OOM compilation** : si compile-from-source, peut OOM le RAM.
   - Solution : utiliser wheel pré-compilé.

**Si SKIP :**

Ajouter dans `docs/reports/TECHNICAL_DEBT.md` :

```markdown
### VLLM_INCOMPATIBLE_PYTORCH
**Détecté :** V4 P2.1 (mai 2026)
**Cause :** vLLM x.y.z incompatible avec torch 2.10.0+cu128 installé.
**Impact :** pas de benchmark comparatif vLLM vs VRAMancer.
**Workaround :** créer un venv dédié vllm pour bench séparé (futur).
```

Et passer directement à P3.

**Si réussite :**

```bash
python -c "import vllm; print('vllm', vllm.__version__)"
```

Note la version dans `resultat_v4.md`.

**Commit :**

```bash
# Cas SKIP:
git add docs/reports/TECHNICAL_DEBT.md
git commit -m "[V2.1] document vLLM install failure: incompatible with torch 2.10+cu128"

# Cas OK (rien à committer dans le repo, juste noter dans resultat_v4.md):
git commit --allow-empty -m "[V2.1] vllm installed, version=X.Y.Z"
```

---

## V2.2 — Benchmark Qwen2.5-7B-Instruct (1 GPU)

(Skip si V2.1 SKIP)

**Setup :**
- 1 GPU (`CUDA_VISIBLE_DEVICES=1` pour cibler RTX 5070 Ti dans torch.cuda FAST_FIRST → mais vLLM utilise PCI order. **Attention** : `CUDA_VISIBLE_DEVICES=0` côté vLLM = RTX 5070 Ti, côté VRAMancer aussi → ils voient le même GPU).
- Prompt : `"Explain quantum entanglement in one paragraph."`
- Max tokens : 200
- 5 runs après warmup

**Script `/tmp/bench_v2_vramancer.py` :**

```python
import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # RTX 5070 Ti côté PCI = côté GPU0 unique vu
from core.inference_pipeline import InferencePipeline

pipe = InferencePipeline(backend_name='huggingface', verbose=False)
pipe.load('Qwen/Qwen2.5-7B-Instruct', num_gpus=1)
prompt = 'Explain quantum entanglement in one paragraph.'
pipe.generate(prompt, max_new_tokens=20)  # warmup

times = []
for i in range(5):
    t0 = time.perf_counter()
    pipe.generate(prompt, max_new_tokens=200)
    times.append(200 / (time.perf_counter() - t0))

import statistics
print(f'VRAMancer median: {statistics.median(times):.2f} tok/s')
import torch
print(f'VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GiB')
```

**Script `/tmp/bench_v2_vllm.py` :**

```python
import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from vllm import LLM, SamplingParams

llm = LLM(model='Qwen/Qwen2.5-7B-Instruct', tensor_parallel_size=1, dtype='bfloat16')
params = SamplingParams(max_tokens=200, temperature=0.0)  # greedy pour comparable
prompt = 'Explain quantum entanglement in one paragraph.'

llm.generate([prompt], params)  # warmup

import statistics
times = []
for i in range(5):
    t0 = time.perf_counter()
    out = llm.generate([prompt], params)
    n_toks = len(out[0].outputs[0].token_ids)
    times.append(n_toks / (time.perf_counter() - t0))

print(f'vLLM median: {statistics.median(times):.2f} tok/s')
import torch
print(f'VRAM: {torch.cuda.memory_allocated()/1024**3:.2f} GiB')
```

**Lancer dans cet ordre (les deux engines ne peuvent pas vivre en même temps en VRAM) :**

```bash
source .venv/bin/activate
python /tmp/bench_v2_vramancer.py 2>&1 | tee /tmp/v2_vramancer.txt
python /tmp/bench_v2_vllm.py 2>&1 | tee /tmp/v2_vllm.txt
```

**Tableau dans `resultat_v4.md` :**

```markdown
### V2.2 — Qwen2.5-7B-Instruct (1 GPU = RTX 5070 Ti)

| Backend     | Tok/s (median) | VRAM     | Notes |
|-------------|----------------|----------|-------|
| VRAMancer HF| XX.X           | YY.Y GiB | torch_dtype=bfloat16 |
| vLLM        | XX.X           | YY.Y GiB | dtype=bfloat16, greedy |
| **Ratio**   | VRAMancer = X.X% de vLLM | | |
```

**Commit (vide, juste pour tracer la phase) :**

```bash
git commit --allow-empty -m "[V2.2] benchmark Qwen 7B 1-GPU: VRAMancer XX vs vLLM YY tok/s"
```

---

## V2.3 — Benchmark Qwen2.5-14B 2-GPU (hétérogène)

(Skip si V2.1 SKIP)

**Particularité :** vLLM peut **refuser** le tensor parallelism sur GPU hétérogènes. C'est un argument fort pour VRAMancer.

**Script `/tmp/bench_v2_vllm_14b.py` :**

```python
import os, time
# PAS de CUDA_VISIBLE_DEVICES restrictif → 2 GPUs visibles
from vllm import LLM, SamplingParams

try:
    llm = LLM(model='Qwen/Qwen2.5-14B', tensor_parallel_size=2, dtype='bfloat16')
    print('vLLM accepted heterogeneous TP=2')
    params = SamplingParams(max_tokens=200, temperature=0.0)
    out = llm.generate(['Explain quantum entanglement.'], params)
    # ... bench logic ...
except Exception as e:
    print(f'vLLM REJECTED heterogeneous: {e}')
```

**Si vLLM refuse :**

Documenter dans `resultat_v4.md` :

```markdown
### V2.3 — Qwen2.5-14B 2-GPU hétérogène (RTX 3090 + RTX 5070 Ti)

**vLLM :** REFUSE le setup hétérogène avec erreur :
```
<copier-coller l'erreur exacte>
```

**VRAMancer :** 28.92 tok/s (mesuré V3 — toujours valide).

**Verdict :** VRAMancer est le seul des deux capable de fonctionner sur GPU hétérogènes.
C'est une niche réelle, pas du marketing.
```

**Si vLLM accepte :** comparer comme en V2.2.

**Commit :**

```bash
git commit --allow-empty -m "[V2.3] vLLM heterogeneous 14B 2-GPU: refused/accepted X.X tok/s"
```

---

# P3 — Diagnostic stress concurrent

**Question critique :** V3 P6.1 a montré n=4 (42.9 tok/s total) **<** n=1 (74.6 tok/s). Ce n'est PAS le comportement attendu d'un système avec batching. Soit :
- Le batcher ne fonctionne pas (fallback séquentiel).
- Le test est mal écrit (génère via `generate()` qui bypass le batcher).
- Le GIL Python sérialise.

V4 P3 doit **trancher** la question.

## V3.1 — Vérifier le routing batcher dans `inference_pipeline.py`

**Action :**

1. Lire `core/inference_pipeline.py` lignes 525–540 :

```python
elif (self.continuous_batcher is not None
      and self.continuous_batcher._running):
    # Route via batcher
    future = self.continuous_batcher.submit(...)
```

2. Confirmer que **TOUS** les appels `pipe.generate()` passent bien par cette branche quand `VRM_CONTINUOUS_BATCHING=1`.

3. Vérifier comment le test V3 P6.1 lançait les requêtes concurrent :

```bash
grep -A20 "def.*concurrent\|threading.Thread" benchmarks/bench_*.py | head -50
```

4. **Si le test V3 lance N threads qui appellent `pipe.generate()` :**
   - Chaque appel passe par la branche batcher (ligne 528+)
   - Les futures sont batchées côté ContinuousBatcher
   - Mais le `submit()` est-il vraiment non-bloquant ? Ou y a-t-il un lock ?

5. Lire `core/continuous_batcher.py` méthode `submit` pour voir si elle bloque.

**Documenter dans `resultat_v4.md` section V3.1** ce qui est observé.

**Commit :**

```bash
git commit --allow-empty -m "[V3.1] audit continuous_batcher routing: <observation>"
```

---

## V3.2 — Test concret avec batcher activé

**Action :**

1. Créer `benchmarks/bench_stress_concurrent_v4.py` (CE fichier va dans le repo, contrairement aux autres) :

```python
"""Stress test V4: continuous batcher behavior under concurrent load.

Compares 3 modes:
- Sequential (1 request at a time, baseline)
- Concurrent N=4 with batcher OFF
- Concurrent N=4 with batcher ON (VRM_CONTINUOUS_BATCHING=1)
"""
import os, time, threading, statistics, sys

def run_concurrent(pipe, prompt, n_concurrent, max_tokens=100):
    """Lance n_concurrent requêtes en parallèle, retourne (total_time, throughput)."""
    results = [None] * n_concurrent
    threads = []
    barrier = threading.Barrier(n_concurrent)

    def worker(idx):
        barrier.wait()  # synchroniser le départ
        t0 = time.perf_counter()
        out = pipe.generate(prompt, max_new_tokens=max_tokens)
        dt = time.perf_counter() - t0
        results[idx] = (dt, len(out) if isinstance(out, str) else max_tokens)

    t0 = time.perf_counter()
    for i in range(n_concurrent):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    total_time = time.perf_counter() - t0

    total_tokens = n_concurrent * max_tokens
    throughput = total_tokens / total_time
    return total_time, throughput, results


def main():
    batcher_on = os.environ.get('VRM_CONTINUOUS_BATCHING', '0') == '1'
    print(f'=== Batcher: {"ON" if batcher_on else "OFF"} ===')

    from core.inference_pipeline import InferencePipeline
    pipe = InferencePipeline(backend_name='huggingface', verbose=False)
    pipe.load('Qwen/Qwen2.5-7B-Instruct', num_gpus=1)

    prompt = 'Explain quantum entanglement.'
    pipe.generate(prompt, max_new_tokens=20)  # warmup

    # Sequential baseline
    seq_times = []
    for i in range(3):
        t0 = time.perf_counter()
        pipe.generate(prompt, max_new_tokens=100)
        seq_times.append(100 / (time.perf_counter() - t0))
    print(f'Sequential: median {statistics.median(seq_times):.2f} tok/s')

    # Concurrent N=4
    for n in [1, 4, 8]:
        total_time, throughput, _ = run_concurrent(pipe, prompt, n, max_tokens=100)
        print(f'N={n}: total {total_time:.2f}s, throughput {throughput:.2f} tok/s')


if __name__ == '__main__':
    main()
```

2. Lancer 2 fois — une fois batcher OFF, une fois ON :

```bash
source .venv/bin/activate
python benchmarks/bench_stress_concurrent_v4.py 2>&1 | tee /tmp/v3_no_batcher.txt
VRM_CONTINUOUS_BATCHING=1 python benchmarks/bench_stress_concurrent_v4.py 2>&1 | tee /tmp/v3_with_batcher.txt
```

3. Tableau dans `resultat_v4.md` :

```markdown
### V3.2 — Stress concurrent avec/sans batcher (Qwen 7B 1-GPU)

| Mode      | Sequential | N=1   | N=4   | N=8   |
|-----------|------------|-------|-------|-------|
| Batcher OFF | XX.X tok/s | XX.X | XX.X | XX.X |
| Batcher ON  | XX.X tok/s | XX.X | XX.X | XX.X |

**Observation :**
- Batcher ON N=4 doit être ≥ Batcher ON N=1 (sinon le batcher ne batche pas)
- Si Batcher ON N=4 < N=1 → BUG du batcher, à investiguer en V5
- Si Batcher ON N=4 > N=1 → batcher fonctionne, V3 P6.1 n'avait juste pas activé
  VRM_CONTINUOUS_BATCHING=1
```

**Commit :**

```bash
git add benchmarks/bench_stress_concurrent_v4.py
git commit -m "[V3.2] add bench_stress_concurrent_v4.py + diagnostic batcher behavior"
```

---

## V3.3 — Conclusion diagnostic

**Action :** rédiger une section finale dans `resultat_v4.md` :

```markdown
### V3.3 — Verdict batcher

[ ] Le batcher fonctionne correctement (N=4 throughput > N=1 quand activé).
    → V3 P6.1 a sous-utilisé le batcher (VRM_CONTINUOUS_BATCHING non activé).
    → Pas de bug à fixer, juste documentation à mettre à jour.

[ ] Le batcher ne batche pas (N=4 throughput ≤ N=1 même activé).
    → Bug réel, à investiguer en V5. Hypothèses :
      1. Lock global dans submit() qui sérialise.
      2. GIL Python qui empêche le parallélisme.
      3. _running flag mal géré.
    → Ouvrir issue dans TECHNICAL_DEBT.md : CONTINUOUS_BATCHER_NO_BATCHING.

[ ] Indéterminé (mesures bruitées).
    → Refaire avec n_runs=10, prompts variés, max_tokens=200.
```

Coche la case appropriée selon les résultats observés.

**Commit :**

```bash
git add resultat_v4.md
git commit -m "[V3.3] verdict diagnostic stress: <case>"
```

Si bug confirmé, ajouter aussi :

```bash
git add docs/reports/TECHNICAL_DEBT.md
git commit -m "[V3.3] document CONTINUOUS_BATCHER_NO_BATCHING in TECHNICAL_DEBT.md"
```

---

# P4 — Audit triton_sampling.py (mise à jour doc)

**Note :** L'idée initiale "fuser top-k dans Triton" est **caduque** — `core/triton_sampling.py` a déjà un fast path `topk + softmax(k) + multinomial(k)` (ligne 138+). Pas besoin de fuser plus.

V4 P4 = juste **audit** pour confirmer ce fast path est utilisé.

## V4.1 — Vérifier que le fast path est emprunté

**Action :**

1. Ajouter un compteur diagnostic temporaire dans `core/triton_sampling.py` :

```python
# tout en haut du module
import os
_DEBUG = os.environ.get('VRM_DEBUG_SAMPLING', '0') == '1'
_PATH_COUNTS = {'fast_topk': 0, 'triton_full': 0, 'pytorch_fallback': 0, 'greedy': 0}
```

Dans chaque branche de `fused_sample`, incrémenter le compteur correspondant si `_DEBUG`.

2. Lancer un bench court avec `VRM_DEBUG_SAMPLING=1` :

```bash
source .venv/bin/activate
VRM_DEBUG_SAMPLING=1 python -c "
import os, torch
os.environ['VRM_DEBUG_SAMPLING'] = '1'
from core.inference_pipeline import InferencePipeline
from core.triton_sampling import _PATH_COUNTS
pipe = InferencePipeline(backend_name='huggingface', verbose=False)
pipe.load('gpt2', num_gpus=1)
pipe.generate('Hello world', max_new_tokens=50)
print('Path counts:', _PATH_COUNTS)
"
```

3. Documenter le résultat dans `resultat_v4.md` section V4.1.

4. **Décision :**
   - Si `fast_topk` est dominant (>90%) → tout va bien, **REVERT le compteur** (commit nul).
   - Si `pytorch_fallback` est dominant → étudier pourquoi le fast path est sauté. Probablement `top_k=0` par défaut quelque part.

**Commit (selon issue) :**

```bash
# Cas OK (fast path utilisé):
git checkout core/triton_sampling.py  # revert compteur
git commit --allow-empty -m "[V4.1] verified triton_sampling fast_topk path used >90% of calls"

# Cas problème:
# garder le compteur en place (utile pour debug futur)
git add core/triton_sampling.py
git commit -m "[V4.1] add VRM_DEBUG_SAMPLING flag — fast_topk only X% of calls, see TECHNICAL_DEBT"

# + ouvrir item TECHNICAL_DEBT
git add docs/reports/TECHNICAL_DEBT.md
git commit -m "[V4.1] document SAMPLING_FAST_PATH_NOT_USED"
```

---

# P5 — Tests et validation finale

## V5.1 — Suite de tests complète

**Action :**

```bash
cd /home/jeremie/VRAMancer/VRAMancer
source .venv/bin/activate
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 VRM_BACKEND_ALLOW_STUB=1 \
  pytest tests/ --ignore=tests/test_chaos_concurrency.py --tb=no --no-cov 2>&1 | tail -5
```

**Cible :** `1 failed, 1070 passed, 39 skipped` (identique baseline).

**Si régression (>1 failed ou <1070 passed) :**

1. Identifier le test fautif :
   ```bash
   VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 VRM_BACKEND_ALLOW_STUB=1 \
     pytest tests/ --ignore=tests/test_chaos_concurrency.py --tb=short --no-cov 2>&1 | grep "FAILED"
   ```
2. Localiser le commit fautif :
   ```bash
   git --no-pager log --oneline main..HEAD
   ```
3. **Revert le commit fautif**, ne désactive PAS le test.
4. Re-run V5.1 jusqu'à 1 failed/1070 passed.

**Documenter dans `resultat_v4.md` :**

```markdown
### V5.1 — Tests
Result: `1 failed, 1070 passed, 39 skipped`
Pre-existing failure: test_health_imports_fault_manager (V3 baseline)
Status: ✅ NO REGRESSION
```

**Commit :** aucun (juste validation).

---

## V5.2 — Smoke test E2E

**Action :**

```bash
source .venv/bin/activate
VRM_MINIMAL_TEST=1 python -m tests.smoke 2>&1 | tail -10
```

**Cible :** exit code 0.

Documenter dans `resultat_v4.md` section V5.2.

---

## V5.3 — Sanity GPU

```bash
source .venv/bin/activate
python -c "
from core.inference_pipeline import InferencePipeline
p = InferencePipeline(backend_name='huggingface', verbose=False)
p.load('gpt2', num_gpus=1)
out = p.generate('The capital of France is', max_new_tokens=10)
print(f'OUT: {out!r}')
assert 'Paris' in out or 'paris' in out.lower(), f'Expected Paris, got: {out}'
print('SANITY OK')
"
```

**Cible :** "SANITY OK" affiché.

---

# P6 — Documentation finale

## V6.1 — Compléter `resultat_v4.md`

Toutes les sections doivent être remplies. Pas de "TODO", pas de placeholder.

Ajouter une section finale :

```markdown
## [SUMMARY]

**Commits V4 (`git log main..HEAD --oneline`) :**
```
<copier-coller la sortie>
```

**Fichiers modifiés (`git diff --stat main..HEAD`) :**
```
<copier-coller la sortie>
```

**Tests :** 1 failed, 1070 passed (identique baseline)

**Performance gains :**
- CUDA Stream Overlap : +Z.Z% sur Qwen 14B 2-GPU OU REVERT (gain insuffisant)
- vLLM comparison : VRAMancer = X% de vLLM sur 7B / vLLM refuse 14B hétérogène

**Stubs résolus / documentés :**
- TRANSFER_MANAGER_LABEL_INCORRECT : <résolu / documenté>
- CONTINUOUS_BATCHER_NO_BATCHING : <résolu / documenté / non détecté>

**Verdict global V4 :** [SUCCESS / PARTIAL / NEEDS_REVISION]
```

**Commit :**

```bash
git add resultat_v4.md
git commit -m "[V6.1] resultat_v4.md final — execution log V4"
```

---

## V6.2 — Mise à jour `.github/copilot-instructions.md`

**Action :**

1. Section "Variables d'environnement essentielles" : ajouter (si V1.5 mergée) :

```markdown
| `VRM_TRANSFER_OVERLAP` | CUDA Stream overlap pour `send_activation` P2P (defaut `1` si V1.5 merged, sinon `0`) |
```

2. Section "Benchmarks reels" : si CUDA Stream Overlap a apporté un gain, ajouter une ligne dans le tableau Qwen 14B.

3. Section "Pieges connus" : si V3.3 a documenté un nouveau bug, l'ajouter en RED FLAGS.

**Commit :**

```bash
git add .github/copilot-instructions.md
git commit -m "[V6.2] update copilot-instructions: V4 changes (transfer overlap, vLLM, batcher diag)"
```

---

# Annexes

## ANNEXE A — Workflow git complet

```bash
# Setup (une fois)
cd /home/jeremie/VRAMancer/VRAMancer
git checkout main
git checkout -b chore/sonnet-plan-v4

# Pour chaque tâche V<x>.<y>:
# 1. Lis la section
# 2. Exécute les commandes
# 3. Si tests passent ET résultats attendus:
git add <files>
git commit -m "[V<x>.<y>] description courte"

# 4. Si tests cassent OU résultats inattendus:
#    -> ARRÊTE, signale à l'utilisateur, ne brute-force pas

# Final
git --no-pager log --oneline main..HEAD  # voir tous les commits V4
# (PAS de git push — l'utilisateur le fera lui-même)
```

---

## ANNEXE B — Critères d'acceptation V4

Le plan V4 est **complet** si TOUS ces critères sont vrais :

- [ ] Branche `chore/sonnet-plan-v4` créée à partir de `main`
- [ ] V0.1 : note ordering GPU ajoutée dans `resultat_v3.md`
- [ ] V0.2 : attribution honnête +382% dans `REBAR_PROXMOX_BENCHMARK.md` + `resultat_v3.md`
- [ ] V0.3 : note méthodologique P6.1 dans `resultat_v3.md`
- [ ] V0.4 : vérification labels TransferManager (commit OK ou TECHNICAL_DEBT entry)
- [ ] V1.1 : `STREAM_OVERLAP_AUDIT.md` créé (30-80 lignes)
- [ ] V1.2 : baseline Qwen 14B mesurée (5 runs, médian dans `resultat_v4.md`)
- [ ] V1.3 : flag `VRM_TRANSFER_OVERLAP` implémenté avec stream
- [ ] V1.4 : benchmark avec/sans stream comparé, decision MERGE/REVERT documentée
- [ ] V1.5 : si MERGE, default à 1 et copilot-instructions mis à jour
- [ ] V2.1 : vLLM installé OU SKIP avec entry TECHNICAL_DEBT
- [ ] V2.2 : Qwen 7B 1-GPU comparaison (si vLLM dispo)
- [ ] V2.3 : Qwen 14B 2-GPU hétérogène (si vLLM dispo)
- [ ] V3.1 : audit routing batcher dans inference_pipeline
- [ ] V3.2 : `bench_stress_concurrent_v4.py` créé + résultats avec/sans batcher
- [ ] V3.3 : verdict diagnostic batcher (case cochée dans `resultat_v4.md`)
- [ ] V4.1 : audit triton_sampling fast path
- [ ] V5.1 : `1 failed, 1070 passed, 39 skipped` confirmé
- [ ] V5.2 : smoke test exit code 0
- [ ] V5.3 : sanity GPU "SANITY OK"
- [ ] V6.1 : `resultat_v4.md` complet avec section [SUMMARY]
- [ ] V6.2 : `.github/copilot-instructions.md` à jour
- [ ] Aucune régression de tests
- [ ] Tous les commits préfixés `[V<x>.<y>]`
- [ ] Aucun script `/tmp/bench_*.py` committé dans le repo
- [ ] Aucun fichier dans `_deprecated/` modifié
- [ ] Aucun test désactivé

---

## ANNEXE C — Anti-patterns à éviter

❌ **Ne PAS** ajouter de fonctionnalités non listées dans ce plan
❌ **Ne PAS** réécrire `_deprecated/` files
❌ **Ne PAS** modifier les chiffres V3 a posteriori (les chiffres sont gravés, on annote)
❌ **Ne PAS** créer de nouveaux modules core/ sans nécessité
❌ **Ne PAS** désactiver des tests pour faire passer la suite
❌ **Ne PAS** marquer "MERGE" si gain < 3%
❌ **Ne PAS** prétendre qu'un bench s'est exécuté sans logs/sortie réels
❌ **Ne PAS** committer de scripts temporaires dans le repo (sauf `bench_stress_concurrent_v4.py` explicitement demandé)
❌ **Ne PAS** brute-forcer : si une tâche échoue 2 fois, ARRÊTE et demande
❌ **Ne PAS** modifier `rust_core/src/` (sauf instruction explicite)
❌ **Ne PAS** push, merge, ouvrir PR — l'utilisateur s'en charge
❌ **Ne PAS** oublier `source .venv/bin/activate` (sinon torch manquant)
❌ **Ne PAS** mélanger les ordres GPU (PCI vs torch.cuda) dans les rapports

---

## ANNEXE D — Fichiers cibles autorisés

**Tu peux modifier :**
- `resultat_v3.md` (uniquement annotations V0.1, V0.2, V0.3)
- `resultat_v4.md` (création + édition continue)
- `docs/reports/REBAR_PROXMOX_BENCHMARK.md` (V0.2 attribution)
- `docs/reports/STREAM_OVERLAP_AUDIT.md` (création V1.1)
- `docs/reports/TECHNICAL_DEBT.md` (ajout d'entries V0.4, V2.1, V3.3, V4.1 si applicable)
- `core/transfer_manager.py` (V1.3, V1.5)
- `core/triton_sampling.py` (V4.1, possible revert)
- `.github/copilot-instructions.md` (V6.2)
- `benchmarks/bench_stress_concurrent_v4.py` (création V3.2)

**Tu NE peux PAS modifier :**
- Tout ce qui est dans `_deprecated/`
- Tout ce qui est dans `tests/` (sauf si V5.1 révèle qu'un test n'est pas adapté à V4 — alors STOP et demande)
- `core/security/`
- `core/paged_attention.py`, `core/paged_attention_cuda.py`
- `csrc/paged_attention_kernel.cu`
- `rust_core/src/`
- `pyproject.toml`, `setup.cfg` (sauf si bump version explicite demandé)
- N'importe quel `bench_*.json` ou `bench_*.txt` existant à la racine (résultats V0-V3 figés)

---

## ANNEXE E — Estimation effort par phase

| Phase | Effort agent | Risque | Bloquant si échec ? |
|-------|--------------|--------|---------------------|
| Préparation | 5 min | Faible | OUI (sans baseline pas de plan) |
| P0 (V0.1-V0.4) | 30 min | Faible | NON |
| P1 (V1.1-V1.5) | 1-2h | Moyen (race conditions) | NON (peut REVERT) |
| P2 (V2.1-V2.3) | 30 min - 2h | Moyen (install vLLM peut foirer) | NON (peut SKIP) |
| P3 (V3.1-V3.3) | 45 min | Faible | NON |
| P4 (V4.1) | 20 min | Faible | NON |
| P5 (V5.1-V5.3) | 10 min | Faible | OUI (validation finale) |
| P6 (V6.1-V6.2) | 30 min | Faible | OUI (livrable) |

**Total estimé :** 4-6h équivalent humain. En une session focus.

---

## ANNEXE F — Quoi faire si tu es bloqué

**Cas 1 : Erreur Python `ModuleNotFoundError`**
→ Vérifie `source .venv/bin/activate`. Si ça persiste, vérifie `pip show <package>`.

**Cas 2 : Test échoue après une modif**
→ `git diff` pour voir ce qui a changé. `git checkout <fichier>` pour revert un fichier précis. Re-run le test ciblé.

**Cas 3 : Bench montre des chiffres bizarres**
→ Lance 3 fois, prends le median. Si toujours bizarre : note dans `resultat_v4.md` "INDÉTERMINÉ — variance excessive" et passe à la tâche suivante.

**Cas 4 : Erreur OOM en chargeant Qwen 14B**
→ Vérifie `nvidia-smi` que rien d'autre n'occupe les GPUs. `python -c "import torch; torch.cuda.empty_cache()"` ne suffit pas — il faut tuer les processus orphelins.

**Cas 5 : Tu ne sais pas comment faire**
→ ARRÊTE-TOI. Édite `resultat_v4.md` avec une section `[BLOCKED@V<x>.<y>]` décrivant le problème. Demande à l'utilisateur. **Ne brute-force pas.**

**Cas 6 : Quelqu'un (formatter, autre agent) a modifié des fichiers entre deux étapes**
→ `git diff` pour voir l'étendue. Si les modifs sont compatibles avec le plan, continue. Sinon, ARRÊTE et demande.

---

**Fin du plan V4. Bonne exécution. Honnêteté > marketing.**
