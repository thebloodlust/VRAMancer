# Plan d'exécution V6 — suite (pour AI exécutante moins capable)

Branche de travail : `feat/v6-lending-cooperative` @ `5796ce8`. Ne pas merger sur `main` avant validation.

Convention : chaque tâche = 1 commit. Toujours lancer la suite de tests après chaque commit avec :
```bash
source .venv/bin/activate && VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 python -m pytest --no-cov tests/ -k "not gpu and not slow" --ignore=tests/test_chaos_concurrency.py
```
Baseline attendue : **967 passed, 1 pre-existing failure (`test_health_imports_fault_manager`), 33 skipped**. Toute régression au-delà de cette failure → revert.

---

## TÂCHE 1 — V6.E batch 2 : sweep silent excepts (BLOC 1/4)

### Contexte
Le commit `247f167` a corrigé 30 silent excepts. Il en reste **~163** dans `core/`. Un *silent except* = `except Exception: pass` ou `except: pass` qui avale toute erreur sans log. Politique projet : remplacer par `_logger.debug("contexte court", exc_info=True)` (ou `_logger.warning` si l'erreur est attendue mais notable).

### Étape 1.1 — Inventaire
```bash
grep -rn "except Exception:\s*$\|except:\s*$" core/ --include="*.py" -A 1 | grep -B 1 "pass\s*$" | head -80
```
Objectif : lister les **30 prochains sites** à corriger. Ne pas dépasser 30 sites par commit (revue plus facile).

### Étape 1.2 — Pour chaque site
1. Vérifier qu'il y a déjà un logger : `grep -n "_logger\s*=\|logger\s*=" <fichier>`. Sinon ajouter en haut du fichier :
   ```python
   import logging
   _logger = logging.getLogger(__name__)
   ```
2. Remplacer le `except: pass` par :
   ```python
   except Exception:
       _logger.debug("<verbe court décrivant ce qui a échoué>", exc_info=True)
   ```
3. **Cas particuliers à NE PAS toucher** :
   - Imports optionnels (`try: import torch ... except: pass`) → garder silencieux, c'est du fallback intentionnel.
   - Cleanup dans `__del__` ou `atexit` → garder silencieux (logging peut être indisponible).
   - Boucles de monitoring qui doivent absolument continuer (ex: `while True: try: poll() except: pass`) → log mais en `_logger.debug` uniquement, pas warning.

### Étape 1.3 — Vérifier
```bash
python -c "import core.<module_modifié>"  # pour chaque fichier
python -m pytest --no-cov tests/ -k "not gpu and not slow" --ignore=tests/test_chaos_concurrency.py 2>&1 | tail -3
```

### Étape 1.4 — Commit
```bash
git add -A
git commit -m "v6e-batch2: silent except sweep (30 sites in N modules)

Replaces bare 'except: pass' / 'except Exception: pass' with
_logger.debug(..., exc_info=True) in N modules:
  - core/<file1>.py: K sites
  - core/<file2>.py: K sites
  ...

Preserves intentional silent fallbacks for:
  - optional dependency imports
  - __del__ / atexit cleanup paths
  - monitoring loops that must not crash

Suite: 967 passed (no regression).

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

### Critère d'acceptation
- 30 sites modifiés (±5 OK).
- Suite reste à 967 passed.
- `grep -rn "except Exception:\s*$" core/ -A 1 | grep -B 1 "pass\s*$" | wc -l` doit avoir diminué de ~60 (chaque site = 2 lignes matchées).

---

## TÂCHE 2 — V6.E batch 3 + 4 + 5

Répéter exactement la TÂCHE 1, **3 fois** (batch 3, 4, 5), 30 sites par batch. Préfixes commit : `v6e-batch3:`, `v6e-batch4:`, `v6e-batch5:`. Arrêter quand l'inventaire (étape 1.1) ne retourne plus de sites légitimes (uniquement fallbacks intentionnels listés en 1.2 cas particuliers).

---

## TÂCHE 3 — V6.D Phase 5 : route les page-slice writes via TransferManager

### Contexte
Dans `core/paged_attention.py:from_hf_cache()`, les écritures de tranche (`page.borrowed_tensor[layer_idx, K|V, :, :slot_end, :] = source_slice`) passent par `cudaMemcpy` natif PyTorch. Pour des tranches ≥ 512 KB, le bypass Rust DtoD de `TransferManager.send_tensor()` est plus rapide. **À faire UNIQUEMENT si la tranche est ≥ 512 KB** (sinon overhead Python > gain).

### Étape 3.1 — Localiser le code à modifier
Lire `core/paged_attention.py` autour de la ligne 740 (`from_hf_cache`). Identifier la boucle `for page_id in pages_to_write:` et son bloc Phase 4 (env-gated `VRM_KV_LEND_ATTENTION=1`).

### Étape 3.2 — Ajouter un seuil et router via TransferManager
Pseudo-code à insérer dans la branche `routed = False ... if env=1 and is_borrowed`:
```python
slice_bytes = source_slice.numel() * source_slice.element_size()
if slice_bytes >= 512 * 1024 and self._transfer_manager is not None:
    try:
        # source = borrower GPU, target = lender GPU (page.lease.lender_gpu)
        self._transfer_manager.send_tensor(
            src_gpu=source_slice.device.index,
            tgt_gpu=page.borrowed_tensor.device.index,
            tensor=source_slice,
            dst_view=page.borrowed_tensor[layer_idx, kv_idx, :, :slot_end, :],
        )
        routed = True
    except Exception:
        _logger.debug("V6.D Phase 5: TransferManager send failed, falling back", exc_info=True)
```

**ATTENTION** : avant de faire ça, vérifier la signature exacte de `TransferManager.send_tensor()` dans `core/transfer_manager.py`. Si elle ne supporte pas `dst_view`, ne PAS inventer de paramètre — fallback à PyTorch assignment ou ouvrir une issue/skip cette tâche.

### Étape 3.3 — Tests
Ajouter dans `tests/test_lending_data_plane.py` :
1. `test_phase5_large_slice_routed_via_transfer_manager` : monkeypatch `TransferManager.send_tensor` pour compter les appels, faire un `from_hf_cache` avec une tranche ≥ 512 KB → assert appel = 1.
2. `test_phase5_small_slice_uses_pytorch_path` : tranche < 512 KB → assert send_tensor non appelé, fallback PyTorch utilisé.

### Étape 3.4 — Commit
Préfixe `v6d-phase5:`. Mentionner le seuil 512 KB et que le decode loop (`write_kv`) reste sur PyTorch (slots trop petits).

### Critère d'acceptation
- 2 nouveaux tests verts.
- Suite à 969 passed, 1 pre-existing failure.
- Si `TransferManager.send_tensor` n'a pas `dst_view` → **SKIP cette tâche**, commit vide non créé, log dans `/memories/repo/todo-next-sessions.md`.

---

## TÂCHE 4 — Préparer le bench GPU réel (sans le lancer)

### Contexte
Le bench réel doit tourner sur la machine 2-GPU (RTX 3090 + RTX 5070 Ti) avec `VRM_KV_LEND=1 VRM_KV_LEND_ATTENTION=1`. Cette tâche **prépare uniquement le script**, ne le lance pas (l'AI moins capable n'a pas le hardware).

### Étape 4.1 — Créer le script
Créer `benchmarks/bench_v6_lending_kv.py`. Squelette :
```python
"""V6.D Phase 3+4 lending KV benchmark.

Compares decode latency with and without VRM_KV_LEND_ATTENTION on a model
that overflows GPU0's pool and would normally evict to CPU.

Run manually on 2-GPU box:
  VRM_KV_LEND=1 VRM_KV_LEND_ATTENTION=1 python benchmarks/bench_v6_lending_kv.py
"""
import os, time, torch
from core.inference_pipeline import get_pipeline

MODEL = os.getenv("VRM_BENCH_MODEL", "Qwen/Qwen2.5-7B-Instruct")
PROMPT = "Once upon a time " * 100  # ~400 tokens to fill cache
MAX_NEW = 256

def run(label):
    p = get_pipeline()
    p.load(MODEL, num_gpus=2)
    t0 = time.perf_counter()
    out = p.generate(PROMPT, max_new_tokens=MAX_NEW)
    dt = time.perf_counter() - t0
    tps = MAX_NEW / dt
    print(f"[{label}] {tps:.2f} tok/s ({dt:.2f}s for {MAX_NEW} tokens)")
    return tps

if __name__ == "__main__":
    print(f"VRM_KV_LEND_ATTENTION={os.getenv('VRM_KV_LEND_ATTENTION', '0')}")
    run("phase3+4" if os.getenv("VRM_KV_LEND_ATTENTION") == "1" else "baseline")
```

### Étape 4.2 — Documenter le protocole
Créer/modifier `/memories/repo/v6-lending-bench-protocol.md` (via outil memory) avec :
- Commandes exactes à lancer (baseline puis phase3+4).
- Métriques attendues : tok/s, peak VRAM par GPU (`nvidia-smi --query-gpu=memory.used --format=csv -l 1`).
- Critère succès : phase3+4 ≥ baseline en tok/s ET utilise les 2 GPUs (pas seulement GPU0 + CPU swap).

### Étape 4.3 — Commit
Préfixe `v6d-bench:`. Mentionner que le script n'est pas lancé en CI.

### Critère d'acceptation
- Le script s'importe sans erreur : `python -c "import benchmarks.bench_v6_lending_kv"`.
- Commit créé, suite toujours à 967.

---

## RÈGLES GÉNÉRALES (à respecter pour toutes les tâches)

1. **Ne jamais** faire `git push` ni `git push --force` ni rebase. Seulement `git commit`.
2. **Ne jamais** modifier des fichiers hors `core/`, `tests/`, `benchmarks/` sans demander.
3. **Ne jamais** désactiver des tests qui échouent. Si un test casse → revert le commit fautif.
4. **Toujours** lancer la suite complète après chaque commit. Si régression au-delà du failure pre-existant unique → `git reset --hard HEAD~1` et redemander à l'utilisateur.
5. **Ne jamais** ajouter de fichier markdown de documentation sauf si demandé explicitement.
6. **Ne jamais** monkeypatch en dehors des fichiers de tests.
7. **Format commit** : titre court (≤72 chars) avec préfixe `v6X-...:`, ligne vide, corps explicatif, ligne vide, `Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>`.
8. Si bloqué sur une tâche → **SKIP**, ne pas inventer de solution, noter dans `/memories/repo/todo-next-sessions.md` et passer à la suivante.

## ORDRE D'EXÉCUTION
1. TÂCHE 1 (v6e-batch2)
2. TÂCHE 2 × 3 (v6e-batch3/4/5)
3. TÂCHE 4 (v6d-bench) — facile, fait avant tâche 3 si tâche 3 bloque
4. TÂCHE 3 (v6d-phase5) — si `TransferManager.send_tensor` supporte `dst_view`

Résultat attendu fin de session : 5-6 nouveaux commits, suite à 967-969 passed, ~30-90 silent excepts éliminés, bench script prêt.
