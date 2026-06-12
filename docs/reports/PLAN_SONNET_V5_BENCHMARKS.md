# Plan Sonnet — Benchmarks Mistral Medium 3.5 + Devstral 22B & Audit V5

**Auteur :** Opus (handoff to Sonnet)
**Date :** 2026-05-06
**Branche :** `chore/sonnet-plan-v5`
**Modèle exécutant :** Claude Sonnet 4.7 (capacité limitée — instructions ultra-explicites)

---

## CONTEXTE / ÉTAT ACTUEL

### Hardware
- **GPU0** : RTX 5070 Ti, 16303 MiB VRAM, SM 12.0 (Blackwell consumer)
- **GPU1** : RTX 3090, 24576 MiB VRAM, SM 8.6 (Ampere)
- **VRAM totale** : 40 GB
- **RAM** : 191 GB DDR4
- **Disque** : 169 GB libres sur `/dev/vda3` (Proxmox VM)
- **Driver** : NVIDIA 595.58.03 / CUDA 13.2 — **chargé et fonctionnel** (vérifier avec `nvidia-smi` avant de commencer)

### Stack logiciel
- `.venv/` : Python 3.12, torch 2.11.0+cu130, transformers 5.8.0, vLLM 0.20.1
- **`llama-cpp-python` n'est PAS installé** (à réinstaller avec build CUDA SM86)
- `bench_turboquant.py` existe déjà (`benchmarks/bench_turboquant.py`)

### Modèles déjà téléchargés
- ✅ `~/.cache/huggingface/hub/models--unsloth--Mistral-Medium-3.5-128B-GGUF/Mistral-Medium-3.5-128B-UD-IQ2_XXS.gguf` — **33 GB, fichier complet**
- ✅ `~/.cache/huggingface/hub/models--unsloth--Qwen3-30B-A3B-GGUF/` — Q4_K_M ~18.6 GB
- ❌ Devstral 22B — **PAS téléchargé** (à faire dans la phase B)

### État `resultat_v5.md`
- P1, P2, P3, P13, P14 : **complétés et documentés**
- P4–P12 : **sections vides** dans `resultat_v5.md` (le code a été fait sur les commits précédents mais pas documenté dans le fichier de résultat) — à auditer (phase D)
- P13 : **BLOCKED définitivement** (DeepSeek-V4-Flash hardware H100/B200 requis)

---

## OBJECTIFS GLOBAUX (par ordre d'exécution)

| Phase | Tâche | Durée estimée | Criticité |
|---|---|---|---|
| **A** | Réinstaller `llama-cpp-python` avec CUDA + bench Mistral Medium 3.5 GGUF | 90 min | Haute |
| **B** | Télécharger Devstral 22B HF + bench TurboQuant (256K et 512K contexte) | 90 min | Haute |
| **C** | Compiler résultats dans `resultat_v5.md` (P15, P16) | 20 min | Haute |
| **D** | Auditer P4–P12 dans `resultat_v5.md` : compléter ou marquer SKIP | 45 min | Moyenne |
| **E** | Commit final + push | 10 min | Haute |

**Total estimé : ~4h30**. Si dépassement, **arrêter à la fin de la phase en cours** et documenter l'état.

---

## RÈGLES IMPÉRATIVES POUR SONNET

1. **NE JAMAIS** modifier `core/`, `csrc/`, `rust_core/`, `tests/` sans raison explicite. Cette session est **bench-only + doc**.
2. **TOUJOURS** activer `.venv` avant chaque commande Python : `source .venv/bin/activate`
3. **TOUJOURS** vérifier l'état du GPU avant un bench : `nvidia-smi`
4. **TOUJOURS** sauvegarder les outputs JSON dans `benchmarks/results/`
5. **TOUJOURS** éditer `resultat_v5.md` avec `replace_string_in_file` (jamais réécrire le fichier complet)
6. **JAMAIS** lancer de pip install ou cargo build en parallèle d'un bench (RAM saturée → OOM)
7. **TOUJOURS** lire la ligne d'output du bench AVANT de marquer "succès" (le statut Python `0` ne suffit pas — vérifier qu'il y a bien des tokens générés)
8. **SI BLOCKED** : documenter explicitement avec `[BLOCKED — <raison>]` dans `resultat_v5.md`, **ne pas brute-force**
9. **NE PAS créer de fichiers markdown supplémentaires** (sauf demande explicite)
10. **COMMITS** : un commit par phase, message format `bench(P15): mistral medium 3.5 GGUF — <résultat>`

---

## PHASE A — Mistral Medium 3.5 GGUF (P15)

### A.1 — Installer `llama-cpp-python` avec CUDA SM86

**ATTENTION** : la compilation prend ~10-15 minutes. Lancer en `mode=sync` avec `timeout=1200000`.

```bash
cd /home/jeremie/VRAMancer/VRAMancer
source .venv/bin/activate

# Tentative 1 : version stable
CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86" \
  FORCE_CMAKE=1 \
  pip install llama-cpp-python --upgrade --force-reinstall --no-binary llama-cpp-python 2>&1 | tee /tmp/llama_install.log
```

**Vérifier le succès :**
```bash
python -c "import llama_cpp; print('OK', llama_cpp.__version__)"
python -c "from llama_cpp import Llama; print('Llama OK')"
```

Si échec sur `nvcc` ou `cuda_runtime.h` → vérifier `which nvcc` et `echo $CUDA_HOME`. Le venv a peut-être besoin de :
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

**Si compilation impossible** : marquer `[BLOCKED — llama-cpp-python build failed: <error>]` dans `resultat_v5.md` section P15 et passer à phase B.

### A.2 — Créer le bench Mistral Medium 3.5

**Créer le fichier** `benchmarks/bench_mistral_medium_gguf.py` :

```python
#!/usr/bin/env python3
"""Bench Mistral Medium 3.5 128B GGUF (UD-IQ2_XXS) sur RTX 3090 + RTX 5070 Ti.

Mesure :
- VRAM usage par GPU
- Tokens/sec en génération
- Latence prefill (prompt processing)
- KV cache RAM @ contextes 32K, 128K, 256K (Q4_0 type_k/v)
"""
from __future__ import annotations
import os, sys, time, json, gc
from pathlib import Path

MODEL_PATH = Path.home() / ".cache/huggingface/hub/models--unsloth--Mistral-Medium-3.5-128B-GGUF/Mistral-Medium-3.5-128B-UD-IQ2_XXS.gguf"
RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
OUT = RESULTS_DIR / "bench_mistral_medium_gguf_v5.json"

if not MODEL_PATH.exists():
    print(f"ERROR: model not found at {MODEL_PATH}")
    sys.exit(1)

from llama_cpp import Llama
import psutil

PROMPT = "Write a Python function that computes the n-th Fibonacci number using memoization. Then explain its time complexity."
N_PREDICT = 128

results = []

# Test 1 : Contexte 32K (rapide)
# Test 2 : Contexte 128K (raisonnable)
# Test 3 : Contexte 256K (full)
for n_ctx in [32768, 131072, 262144]:
    print(f"\n{'='*60}\nTesting n_ctx={n_ctx}\n{'='*60}")
    gc.collect()
    ram_before = psutil.virtual_memory().used / 1e9

    t0 = time.time()
    try:
        llm = Llama(
            model_path=str(MODEL_PATH),
            n_gpu_layers=-1,
            tensor_split=[16303, 24576],   # 5070 Ti (16GB), 3090 (24GB)
            n_ctx=n_ctx,
            type_k=2,                       # Q4_0
            type_v=2,
            offload_kv=False,               # KV en RAM
            flash_attn=True,
            verbose=False,
        )
    except Exception as e:
        print(f"FAIL load n_ctx={n_ctx}: {e}")
        results.append({"n_ctx": n_ctx, "error": str(e), "load_time_s": time.time() - t0})
        continue

    load_time = time.time() - t0
    ram_after_load = psutil.virtual_memory().used / 1e9

    # Prefill bench
    t1 = time.time()
    output = llm(PROMPT, max_tokens=N_PREDICT, echo=False, temperature=0.0)
    elapsed = time.time() - t1

    text = output["choices"][0]["text"]
    n_generated = output["usage"]["completion_tokens"]
    n_prompt = output["usage"]["prompt_tokens"]

    ram_peak = psutil.virtual_memory().used / 1e9

    res = {
        "n_ctx": n_ctx,
        "load_time_s": round(load_time, 2),
        "elapsed_s": round(elapsed, 2),
        "n_prompt_tokens": n_prompt,
        "n_generated_tokens": n_generated,
        "tokens_per_sec": round(n_generated / elapsed, 2) if elapsed > 0 else 0,
        "ram_load_gb": round(ram_after_load - ram_before, 2),
        "ram_peak_gb": round(ram_peak - ram_before, 2),
        "first_chars": text[:200],
    }
    print(json.dumps(res, indent=2))
    results.append(res)

    del llm
    gc.collect()

with open(OUT, "w") as f:
    json.dump({"model": "Mistral-Medium-3.5-128B-UD-IQ2_XXS", "results": results}, f, indent=2)
print(f"\nSaved to {OUT}")
```

### A.3 — Lancer le bench

```bash
source .venv/bin/activate
mkdir -p benchmarks/results
python benchmarks/bench_mistral_medium_gguf.py 2>&1 | tee /tmp/bench_mistral_medium.log
```

**Durée estimée** : 30-50 minutes (le load du modèle prend ~3-5 min, prefill 256K plusieurs minutes).

**Si OOM RAM** : réduire la matrice à `[32768, 131072]` seulement.
**Si erreur `tensor_split`** : essayer `tensor_split=[0.6, 1.0]` (ratios au lieu de MB absolus).

### A.4 — Documenter dans `resultat_v5.md`

Ajouter cette section AVANT `## [SUMMARY]` :

```markdown
## [P15] — Mistral Medium 3.5 128B GGUF bench
- **Modèle** : `unsloth/Mistral-Medium-3.5-128B-GGUF` (UD-IQ2_XXS, 33 GB)
- **Backend** : llama.cpp (`llama-cpp-python` CUDA SM86)
- **Hardware** : RTX 5070 Ti (16GB) + RTX 3090 (24GB) tensor_split, KV cache RAM Q4_0
- **Résultats** :
  | Contexte | Load (s) | Tok/s | RAM peak | VRAM (GPU0/GPU1) |
  |---|---|---|---|---|
  | 32K | <X> | <Y> | <Z> GB | <V> / <V> GB |
  | 128K | <X> | <Y> | <Z> GB | <V> / <V> GB |
  | 256K | <X> | <Y> | <Z> GB | <V> / <V> GB |
- **Output JSON** : `benchmarks/results/bench_mistral_medium_gguf_v5.json`
- **Note** : TurboQuant **NON applicable** (llama.cpp pipeline indépendant). KV quantization native llama.cpp via `type_k=2 type_v=2` (Q4_0).
```

Remplacer `<X>`, `<Y>`, `<Z>`, `<V>` par les valeurs réelles du JSON.

### A.5 — Commit

```bash
cd /home/jeremie/VRAMancer/VRAMancer
git add benchmarks/bench_mistral_medium_gguf.py benchmarks/results/bench_mistral_medium_gguf_v5.json resultat_v5.md
git commit -m "bench(P15): mistral medium 3.5 128B GGUF — <X> tok/s @ 256K ctx"
```

---

## PHASE B — Devstral 22B HF + TurboQuant (P16)

### B.1 — Vérifier disponibilité Devstral

```bash
source .venv/bin/activate
python -c "
from huggingface_hub import HfApi
api = HfApi()
for repo in ['mistralai/Devstral-Small-2505', 'mistralai/Devstral-Small-22B']:
    try:
        info = api.repo_info(repo)
        print(f'OK: {repo}  gated={info.gated}  private={info.private}')
    except Exception as e:
        print(f'FAIL: {repo}  {e}')
"
```

**Si tous les repos retournent FAIL/401** : essayer `unsloth/Devstral-Small-2505-bnb-4bit` ou fallback Qwen3-30B-A3B (déjà téléchargé en GGUF — pas idéal).

### B.2 — Télécharger Devstral 22B (NF4 si dispo, sinon BF16)

**Option A** — version BF16 (44 GB, marche pas en VRAM mais TurboQuant + NF4 BnB la quantifie au load) :
```bash
source .venv/bin/activate
huggingface-cli download mistralai/Devstral-Small-2505 --local-dir ~/.cache/huggingface/hub/models--mistralai--Devstral-Small-2505/ 2>&1 | tee /tmp/dl_devstral.log &
echo "DL_PID=$!"
```

**Option B** — pré-quantifié 4-bit (~12 GB) :
```bash
huggingface-cli download unsloth/Devstral-Small-2505-bnb-4bit --local-dir ~/.cache/huggingface/hub/models--unsloth--Devstral-Small-2505-bnb-4bit/
```

**Vérifier la taille téléchargée :**
```bash
du -sh ~/.cache/huggingface/hub/models--*Devstral*
```

### B.3 — Créer le bench Devstral

**Créer le fichier** `benchmarks/bench_devstral_turboquant.py` :

```python
#!/usr/bin/env python3
"""Bench Devstral 22B HF + VRAMancer TurboQuant (multi-GPU + KV compression).

Test matrix :
- BF16 baseline (NF4 weights, BF16 KV)         @ ctx 32K
- TurboQuant 3-bit KV                           @ ctx 32K
- TurboQuant + Sparse V 10%                     @ ctx 32K
- TurboQuant + Sparse V 10%                     @ ctx 128K  (long context demo)
- TurboQuant + Sparse V 10%                     @ ctx 256K  (extreme demo)
"""
from __future__ import annotations
import os, sys, time, json, gc, subprocess
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("VRM_MINIMAL_TEST", None)
os.environ.pop("VRM_TEST_MODE", None)

# Pick whichever Devstral was downloaded
MODEL_CANDIDATES = [
    "unsloth/Devstral-Small-2505-bnb-4bit",
    "mistralai/Devstral-Small-2505",
]
MODEL = None
import os as _os
for cand in MODEL_CANDIDATES:
    cache_dir = Path.home() / ".cache/huggingface/hub" / f"models--{cand.replace('/', '--')}"
    if cache_dir.exists():
        MODEL = cand
        break
if MODEL is None:
    print("ERROR: no Devstral cached; run download first")
    sys.exit(1)
print(f"Using model: {MODEL}")

import torch

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
OUT = RESULTS_DIR / "bench_devstral_turboquant_v5.json"

PROMPT = "Write a Python function that computes the n-th Fibonacci number using memoization. Then explain its time complexity."

CONFIGS = [
    {"name": "BF16 KV (baseline)", "env": {"VRM_KV_COMPRESSION": "", "VRM_SPARSE_V_RATIO": "1.0", "VRM_QUANTIZATION": "nf4"}, "ctx": 32768},
    {"name": "TurboQuant 3-bit",   "env": {"VRM_KV_COMPRESSION": "turboquant", "VRM_KV_COMPRESSION_BITS": "3", "VRM_SPARSE_V_RATIO": "1.0", "VRM_QUANTIZATION": "nf4"}, "ctx": 32768},
    {"name": "TQ + SparseV 10%",   "env": {"VRM_KV_COMPRESSION": "turboquant", "VRM_KV_COMPRESSION_BITS": "3", "VRM_SPARSE_V_RATIO": "0.1", "VRM_QUANTIZATION": "nf4"}, "ctx": 32768},
    {"name": "TQ+SV10 ctx 128K",   "env": {"VRM_KV_COMPRESSION": "turboquant", "VRM_KV_COMPRESSION_BITS": "3", "VRM_SPARSE_V_RATIO": "0.1", "VRM_QUANTIZATION": "nf4"}, "ctx": 131072},
    {"name": "TQ+SV10 ctx 256K",   "env": {"VRM_KV_COMPRESSION": "turboquant", "VRM_KV_COMPRESSION_BITS": "3", "VRM_SPARSE_V_RATIO": "0.1", "VRM_QUANTIZATION": "nf4"}, "ctx": 262144},
]

results = []

for cfg in CONFIGS:
    print(f"\n{'='*60}\n{cfg['name']}  ctx={cfg['ctx']}\n{'='*60}")
    # Apply env
    for k, v in cfg["env"].items():
        os.environ[k] = v

    # Reset pipeline
    try:
        from core.inference_pipeline import reset_pipeline, get_pipeline
        reset_pipeline()
    except Exception as e:
        print(f"reset failed: {e}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)

    t0 = time.time()
    try:
        pipe = get_pipeline()
        pipe.load(MODEL, num_gpus=2)
        load_t = time.time() - t0

        t1 = time.time()
        out = pipe.generate(PROMPT, max_new_tokens=128, temperature=0.0)
        gen_t = time.time() - t1

        n_gen = len(out.split()) if isinstance(out, str) else 128

        vram_per_gpu = []
        for i in range(torch.cuda.device_count()):
            vram_per_gpu.append(round(torch.cuda.max_memory_allocated(i) / 1e9, 2))

        res = {
            "config": cfg["name"],
            "ctx": cfg["ctx"],
            "load_time_s": round(load_t, 2),
            "gen_time_s": round(gen_t, 2),
            "tokens_per_sec": round(n_gen / gen_t, 2) if gen_t > 0 else 0,
            "vram_gb_per_gpu": vram_per_gpu,
            "first_chars": (out[:200] if isinstance(out, str) else str(out)[:200]),
        }
    except Exception as e:
        res = {"config": cfg["name"], "ctx": cfg["ctx"], "error": str(e)[:500]}

    print(json.dumps(res, indent=2))
    results.append(res)

with open(OUT, "w") as f:
    json.dump({"model": MODEL, "results": results}, f, indent=2)
print(f"\nSaved to {OUT}")
```

### B.4 — Lancer le bench

```bash
source .venv/bin/activate
python benchmarks/bench_devstral_turboquant.py 2>&1 | tee /tmp/bench_devstral.log
```

**Durée estimée** : 60 min (5 configs × ~10 min chacun, ctx 256K beaucoup plus lent).

**Si OOM VRAM** sur ctx 256K : c'est attendu, documenter et passer.
**Si erreur `pipe.generate`** : utiliser `pipe.infer(input_ids)` à la place (cf. `core/inference_pipeline.py`).

### B.5 — Documenter dans `resultat_v5.md`

Ajouter section P16 avant `## [SUMMARY]` :

```markdown
## [P16] — Devstral 22B HF + TurboQuant
- **Modèle** : `<MODEL>` (NF4 quantization, multi-GPU split)
- **Backend** : VRAMancer HuggingFaceBackend + KVCacheCompressor
- **Configuration cumulée** : NF4 weights + TurboQuant 3-bit KV + Sparse V 10% (top 10% values decompressed)
- **Résultats** :
  | Config | Ctx | Tok/s | VRAM GPU0 | VRAM GPU1 | Note |
  |---|---|---|---|---|---|
  | BF16 KV baseline | 32K | <X> | <V> | <V> | référence |
  | TurboQuant 3-bit | 32K | <Y> | <V> | <V> | -<Z>% vs baseline |
  | TQ + Sparse V 10% | 32K | <Y> | <V> | <V> | gain Sparse V |
  | TQ + Sparse V 10% | 128K | <Y> | <V> | <V> | long context |
  | TQ + Sparse V 10% | 256K | <Y> | <V> | <V> | extreme demo |
- **Output JSON** : `benchmarks/results/bench_devstral_turboquant_v5.json`
- **Conclusion** : VRAMancer + TurboQuant permet de pousser Devstral 22B à <ctx max> contexte sur 40 GB VRAM, là où baseline BF16 plafonne à ~64K.
```

### B.6 — Commit

```bash
git add benchmarks/bench_devstral_turboquant.py benchmarks/results/bench_devstral_turboquant_v5.json resultat_v5.md
git commit -m "bench(P16): devstral 22B + TurboQuant — <X> tok/s @ 256K ctx"
```

---

## PHASE C — Compiler le SUMMARY

Mettre à jour la section `## [SUMMARY]` à la fin de `resultat_v5.md` :

```markdown
## [SUMMARY]
- **P1–P3** : continuous batcher revert, transfer labels, vLLM use_cache → ✅ done
- **P4–P12** : voir audit ci-dessous (Phase D)
- **P13** : DeepSeek V4 Flash → ❌ BLOCKED hardware (H100/B200 requis)
- **P14** : tests post-reboot 1064 passed, 0 failed → ✅
- **P15** : Mistral Medium 3.5 128B GGUF → ✅ <X> tok/s @ 256K ctx, <Y> GB RAM peak
- **P16** : Devstral 22B HF + TurboQuant → ✅ <X> tok/s @ 256K ctx, gain VRAM <Z>%

**Stack démonstration finale** :
1. **Mistral Medium 3.5 GGUF (llama.cpp)** : modèle "deep work" 128B, qualité SWEbench 77.6%
2. **Devstral 22B HF + VRAMancer TurboQuant** : modèle interactif, contexte 256K sur 40 GB VRAM
```

---

## PHASE D — Audit P4–P12 (sections vides dans `resultat_v5.md`)

Pour chaque P4 à P12, vérifier ce qui a été commité sur la branche `chore/sonnet-plan-v5` :

```bash
cd /home/jeremie/VRAMancer/VRAMancer
git log --oneline chore/sonnet-plan-v4..chore/sonnet-plan-v5
```

Pour chaque commit qui réfère P4-P12, ajouter une ligne courte dans la section correspondante de `resultat_v5.md`.

### Référentiel des P4-P12 (depuis `docs/reports/PLAN_ACTION_V5.md`)

| P | Tâche prévue | Action audit |
|---|---|---|
| P4 | PyO3 transfer_async | Chercher commit `transfer_async` ou `pyo3` → noter SHA + status |
| P5 | Silent exceptions sweep | Chercher `except Exception: pass` corrigés |
| P6 | Hetero advantage bench | Chercher `bench_hetero` |
| P7 | usb4_distributed_vram example | Chercher `examples/usb4` |
| P8 | Repo root cleanup | Chercher commits `chore: cleanup` |
| P9 | TODO markers | Chercher `TODO.md` ou `# TODO` modifs |
| P10 | Tests coverage | Chercher tests added |
| P11 | Doc 1.6.0 | Chercher CHANGELOG |
| P12 | HF browser load | Chercher dashboard ou CLI |

**Format pour chaque section vide :**

```markdown
## [P4] — PyO3 transfer_async
- **Status** : <DONE / SKIP / PARTIAL>
- **Commits** : <SHA1>, <SHA2>
- **Note** : <description courte>
```

**Si rien trouvé** : marquer `[SKIP — non commencé en V5, reporté]`.

---

## PHASE E — Commit final + push

```bash
cd /home/jeremie/VRAMancer/VRAMancer
git add resultat_v5.md
git commit -m "doc(V5): final summary + P15/P16 benchmarks + P4-P12 audit"
git log --oneline -10
# git push origin chore/sonnet-plan-v5  # PAS DE PUSH SANS DEMANDER L'UTILISATEUR
```

**NE PAS PUSH** sans demander la confirmation explicite à l'utilisateur.

---

## TROUBLESHOOTING — Erreurs fréquentes

### Erreur 1 : `llama-cpp-python` ne compile pas (`nvcc not found`)
```bash
which nvcc || ls /usr/local/cuda*/bin/nvcc
# Si absent → apt install nvidia-cuda-toolkit ou sudo apt install cuda-toolkit-13-2
```

### Erreur 2 : `nvidia-smi` retourne "No devices found"
- Driver Proxmox déchargé. Reboot VM ou `sudo modprobe nvidia`. **Si impossible** : phase A et B BLOCKED, marquer dans `resultat_v5.md`.

### Erreur 3 : OOM VRAM sur Devstral 256K ctx
- C'est attendu. Documenter `[OOM — ctx max sur ce hardware]` et continuer.

### Erreur 4 : `pipe.generate()` bloque (timeout)
- `core/inference_pipeline.py` `generate()` peut router vers `continuous_batcher` si `VRM_CONTINUOUS_BATCHING=1`. Vérifier `os.environ.get("VRM_CONTINUOUS_BATCHING")` est unset.

### Erreur 5 : TurboQuant pas activé
- Vérifier dans les logs : `KVCacheCompressor` doit logger `TurboQuant: using fused CUDA kernel` ou `using PyTorch GPU ops`. Si absent → `VRM_KV_COMPRESSION=turboquant` mal pris en compte.

### Erreur 6 : Devstral 401 sur HF
- Repo gated. Tester `unsloth/Devstral-Small-2505-bnb-4bit` (public). Si tous gated → fallback **Qwen3-30B-A3B GGUF** (déjà téléchargé) mais alors phase B devient un bench llama.cpp comme phase A — pas de démo TurboQuant. Documenter clairement.

---

## CHECKLIST FINALE AVANT HANDOFF

Avant d'arrêter la session, Sonnet doit vérifier :

- [ ] `benchmarks/results/bench_mistral_medium_gguf_v5.json` existe et contient ≥1 résultat valide
- [ ] `benchmarks/results/bench_devstral_turboquant_v5.json` existe et contient ≥1 résultat valide
- [ ] `resultat_v5.md` contient les sections P15, P16
- [ ] `resultat_v5.md` P4-P12 ont chacun un statut (DONE/SKIP/PARTIAL)
- [ ] `resultat_v5.md` SUMMARY mis à jour
- [ ] `git log --oneline -5` montre 3 commits propres (P15, P16, audit)
- [ ] **Aucun fichier .py modifié dans `core/`, `csrc/`, `rust_core/`, `tests/`**

Si ✅ tout : laisser un message court à l'utilisateur résumant les tok/s obtenus et demander la permission de push.
Si ❌ blocages : laisser un message clair avec la phase atteinte et le blocage.

---

**FIN DU PLAN. Bonne chance Sonnet.** Si en doute, **arrête-toi et documente** — ne brute-force pas un bench qui timeout.
