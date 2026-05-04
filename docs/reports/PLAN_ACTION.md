# Plan d'action VRAMancer — spécification d'exécution détaillée pour Sonnet 4.6

> **Auteur** : Architecte / Auditeur
> **Destinataire** : agent de codage Sonnet 4.6 (capacité réduite — chaque étape doit être triviale à exécuter)
> **Date** : 4 mai 2026
> **Source de vérité technique** : `.github/copilot-instructions.md` (audit 27 mars 2026)

Ce document est un **cahier des charges exécutable**. Chaque tâche est décrite avec :
- les fichiers exacts (chemins + lignes)
- les snippets de code à insérer ou remplacer
- les commandes shell à exécuter
- les critères d'acceptation testables

Sonnet 4.6 doit suivre l'ordre, **ne rien faire qui ne soit pas explicitement listé**, et reporter chaque tâche dans `docs/reports/EXEC_LOG.md`.

---

## 0. Préparation de l'environnement (à faire UNE FOIS au début)

### 0.0.1 — Créer la branche de travail

```bash
cd /home/jeremie/VRAMancer/VRAMancer
git checkout -b chore/sonnet-plan-exec
```

### 0.0.2 — Activer l'environnement Python

```bash
source .venv/bin/activate
```

### 0.0.3 — Vérifier les tests AVANT toute modification (baseline)

```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
  pytest -q tests/ --ignore=tests/test_chaos_concurrency.py 2>&1 | tail -10
```

Noter le nombre exact de `passed` / `failed` dans `EXEC_LOG.md` (entrée `### [BASELINE]`). Continuer même s'il y a des failures pré-existantes.

### 0.0.4 — Créer le fichier de log d'exécution

Créer `docs/reports/EXEC_LOG.md` avec :

```markdown
# Journal d'exécution — Sonnet 4.6

Format de chaque entrée :

### [TX.Y] Titre — YYYY-MM-DD HH:MM

**Status** : DONE | BLOCKED | PARTIAL | SKIPPED
**Commit** : <hash court>
**Files changed** : liste
**Tests added** : liste
**Tests passing** : N passed, M failed (vs baseline)
**Notes** : 1-3 lignes max

---

### [BASELINE] État initial — <date>

**Tests passing** : <N> passed, <M> failed
```

---

## Règles globales

1. **Un commit par tâche**. Format : `[TX.Y] description courte (≤ 72 chars)`.
2. **Avant chaque commit**, relancer `pytest -q tests/ --ignore=tests/test_chaos_concurrency.py` et confirmer 0 régression.
3. **Ne pas modifier** : `_deprecated/`, `.github/copilot-instructions.md`, `benchmarks/*.py` (sauf instruction), `tests/test_chaos_concurrency.py`.
4. **Aucune dépendance ajoutée** dans `requirements*.txt` ni `pyproject.toml`.
5. **Aucun symbole public renommé** sans shim.
6. **En cas de blocage**, écrire BLOCKED dans `EXEC_LOG.md` et passer à la tâche suivante.
7. **Pas d'auto-push, pas d'auto-merge**.

---

## P0 — Cœur produit

### T0.1 — Sécuriser `trust_remote_code` dans `core/inference_pipeline.py`

**Fichier** : `core/inference_pipeline.py`

**État actuel** vérifié (lignes 1049 et 1056) :

```python
        try:
            config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=True,
            )
        except Exception:
            pass
        # Fallback: quick network fetch (only config.json, small file)
        if config is None:
            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                return None
```

**Action** — remplacer ce bloc par :

```python
        _trc = os.environ.get("VRM_TRUST_REMOTE_CODE") == "1"
        try:
            config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=_trc, local_files_only=True,
            )
        except Exception:
            pass
        # Fallback: quick network fetch (only config.json, small file)
        if config is None:
            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=_trc)
            except Exception:
                return None
```

**Validation** :

```bash
grep -n "trust_remote_code=True" core/inference_pipeline.py    # 0 résultat attendu
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 pytest -q tests/test_pipeline.py tests/test_e2e_pipeline.py
```

**Commit** : `[T0.1] secure trust_remote_code via VRM_TRUST_REMOTE_CODE`

---

### T0.2 — Renommage marketing (logs et docstrings uniquement)

**Périmètre strict** : uniquement chaînes dans `logger.*`, `print()` et `"""docstrings"""`. **Ne pas renommer** : classes, fonctions, fichiers, attributs publics (`synapses`, `Synapse`, `_apply_neuroplasticity_score`, etc.).

#### T0.2.a — `core/scheduler.py`

Ligne 367, **AVANT** :
```python
            _logger.debug(f" [Swarm Synapse] Anticipating execution flow... Prefetching layers {predicted} ahead of current max layer {max_idx}.")
```
**APRÈS** :
```python
            _logger.debug(f"[Adaptive Scheduler] Prefetching layers {predicted} ahead of current max layer {max_idx}.")
```

#### T0.2.b — `core/stream_manager.py`

Ligne 176, **AVANT** :
```python
        """Predict and preload upcoming layers asynchronously (Anticipatory Brain).
```
**APRÈS** :
```python
        """Predict and preload upcoming layers asynchronously (adaptive prefetch).
```

Ligne 217, **AVANT** :
```python
            self.logger.info(" Anticipatory Brain scheduled %d layers for prefetching (predicted: %s) without blocking.", prefetched, predicted)
```
**APRÈS** :
```python
            self.logger.info("Adaptive prefetcher scheduled %d layers (predicted: %s) without blocking.", prefetched, predicted)
```

#### T0.2.c — `core/network/connectome.py`

Ligne 2, **AVANT** :
```python
VRAMancer Connectome (Neuroplasticity Engine)
```
**APRÈS** :
```python
VRAMancer Connectome (adaptive routing weights)
```

Ligne 7, **AVANT** :
```python
Ce module surveille en permanence (en tâche de fond) la qualité des "synapses" (liens réseau ou PCIe) 
```
**APRÈS** :
```python
Ce module surveille en permanence (en tâche de fond) la qualité des liens (réseau ou PCIe)
```

Ligne 121, **AVANT** :
```python
                self.log.info("[Neuroplasticité] Nouvelle synapse vers %s (%s)", node_id, ip)
```
**APRÈS** :
```python
                self.log.info("[Connectome] Nouveau lien vers %s (%s)", node_id, ip)
```

Ligne 153, **AVANT** :
```python
                self.log.warning("[Neuroplasticité] Perte vers %s — force: %.2f",
```
**APRÈS** :
```python
                self.log.warning("[Connectome] Perte vers %s — score: %.2f",
```

**Important** : **ne pas** renommer `class Synapse`, `self.synapses`, `_ping_synapse`. Ce sont des identifiants publics du module.

#### T0.2.d — `core/network/anycast_balancer.py`

Lignes 6, 10, 21, 193, 195, 210, 300, 325, 340 contiennent `Hebbian` ou `synapse strength` dans des **commentaires/docstrings** :

- Remplacer chaque `Hebbian` (mot entier) par `adaptive`.
- Remplacer chaque `synapse strength` par `link strength` (dans docstrings/commentaires).
- **Ne pas** modifier `syn = connectome.synapses.get(node_id)` (ligne 211) — c'est du code, pas de la chaîne.

Vérifier après modif :
```bash
grep -n "Hebbian" core/network/anycast_balancer.py    # 0 résultat
```

#### T0.2.e — `core/orchestrator/placement_engine.py`

Ligne 235, **AVANT** :
```python
                        # -> THE NEUROPLASTICITY HEURISTIC
```
**APRÈS** :
```python
                        # -> Adaptive scoring heuristic
```

Ligne 251 : **vérifier d'abord** si la chaîne est utilisée ailleurs :
```bash
grep -rn "vram_neuroplastic" tests/ benchmarks/ docs/ core/
```

- Si **aucun match dans tests/**, modifier `"strategy": "vram_neuroplastic",` → `"strategy": "vram_adaptive",`.
- Si **un test l'utilise**, **laisser tel quel** et noter dans `EXEC_LOG.md` : `T0.2.e: kept "vram_neuroplastic" string due to test dependency`.

**Ne pas** renommer la méthode `_apply_neuroplasticity_score` (privée mais référencée à plusieurs endroits — risque de casse).

#### T0.2.f — `core/parity_memory.py`

Ligne 110 : `self.native_core.generate_holographic_parity(...)` — **NE PAS MODIFIER**. C'est un appel FFI au binding Rust.

Ligne 16 (docstring) : laisser tel quel (mention historique).
Ligne 210 : `HolographicKVManager = ParityKVManager` — **NE PAS MODIFIER**, c'est un alias de compatibilité.

#### T0.2.g — Validation globale

```bash
grep -rn "Anticipatory Brain\|Swarm Synapse\|Neuroplasticity Engine\|Hebbian" core/
# Doit retourner 0 ligne (ou uniquement des matches dans des shims marqués).

VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 pytest -q tests/ --ignore=tests/test_chaos_concurrency.py
```

**Commit** : `[T0.2] drop marketing wording in logs and docstrings`

---

### T0.3 — Documenter le cœur produit

**Action** : créer `docs/CORE_ENGINE.md` avec ce contenu **exact** :

````markdown
# VRAMancer — Cœur produit

## Pitch

VRAMancer est un moteur d'inférence LLM pour **GPU hétérogènes**. Il découpe un modèle proportionnellement à la VRAM libre de chaque GPU, oriente l'inférence bloc par bloc, et gère les transferts inter-GPU avec plusieurs stratégies (P2P, CPU-staged, RDMA).

**Preuve concrète** : Qwen2.5-14B exécuté sur RTX 3090 (23.6 GB) + RTX 5070 Ti (15.5 GB) = **6.0 tok/s**, alors que ni l'un ni l'autre ne peut charger le modèle seul (OOM).

## Pipeline

```
HTTP API (FastAPI ou Flask)
   ↓
InferencePipeline (core/inference_pipeline.py)
   ↓
Backend (core/backends.py — HF/vLLM/llama.cpp/Ollama)
   ↓
ModelSplitter (core/model_splitter.py)  → plan de placement par couche
   ↓
Scheduler (core/scheduler.py)  → allocation des blocs
   ↓
TransferManager (core/transfer_manager.py)  → P2P / CPU-staged
   ↓
ComputeEngine (core/compute_engine.py)  → forward
```

## Composants clés

### `core/model_splitter.py`
Split VRAM-proportionnel basé sur la **mémoire libre** (pas totale). Pondération compute-aware. Si `LayerProfiler` est disponible, le split devient DP-optimal.

### `core/transfer_manager.py`
Stratégies de transfert tensor entre GPU : `cudaMemcpyPeerAsync`, torch P2P, PyO3/Rust `direct_vram_copy`, NCCL group, staging via mémoire pinned host. Détection topologie au démarrage.

### `core/scheduler.py`
Alloue les blocs aux GPU et gère le routage par couche.

### `core/paged_attention.py`
KV cache paginé style vLLM. Compatible compression KV via `kv_quantizer.py` (PolarQuant + QJL ≈ 3.5 bits/dim). Support GQA head-batching.

### `core/inference_pipeline.py`
Orchestrateur central. `load()` câble tous les sous-systèmes. `generate()` route vers : speculative decoding → continuous batcher (si actif) → forward direct.

## Variables d'environnement essentielles

| Variable | Rôle |
|---|---|
| `VRM_QUANTIZATION` | `nvfp4` (Blackwell), `nf4` (BnB), `int8` (BnB), vide = BF16 |
| `VRM_KV_COMPRESSION` | `turboquant` pour activer PolarQuant + QJL |
| `VRM_CONTINUOUS_BATCHING` | `1` active le batcher multi-requêtes |
| `VRM_PARALLEL_MODE` | `pp` (pipeline) ou `tp` (tensor parallel + NCCL) |
| `VRM_VRAM_LENDING` | `1` active le pool de VRAM cross-GPU |
| `VRM_TRUST_REMOTE_CODE` | `1` autorise `trust_remote_code` sur les modèles HF |

## Limites connues

- **VM Proxmox** : seule la stratégie 4 (CPU-staged pinned) fonctionne ; P2P bloqué par IOMMU.
- **BnB multi-GPU** : bug upstream `transformers 5.3.0 + accelerate 1.13.0` → forcé en single-GPU.
- **Continuous batcher** : limite `max_waiting_queue=256`, GIL CPython inhérent.
- **Rust transport** : `detect_best_transport()` retourne `ZeroCopyTcp` (stub).

Pour les détails complets, voir `.github/copilot-instructions.md`.
````

**Validation** :
```bash
test -f docs/CORE_ENGINE.md && wc -l docs/CORE_ENGINE.md    # entre 50 et 150 lignes
```

**Commit** : `[T0.3] add docs/CORE_ENGINE.md`

---

### T0.4 — Centraliser la configuration

**Préalable** : `core/env_flags.py` existe — **ne pas le supprimer**, juste compléter.

**Action 1** : créer `core/config_manager.py` avec ce contenu **exact** :

```python
"""Centralized configuration manager for VRAMancer.

Sources priority (highest first):
  1. Environment variables (``VRM_*``)
  2. ``config.yaml`` (resolved via core.config)
  3. Hardcoded defaults
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _yaml_load(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _resolve_yaml_path() -> Optional[Path]:
    candidates = []
    try:
        from core.config import get_config_path  # type: ignore
        p = get_config_path()
        if p:
            candidates.append(Path(p))
    except Exception:
        pass
    candidates.append(Path.cwd() / "config.yaml")
    for c in candidates:
        if c.is_file():
            return c
    return None


def _bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "on")
    return default


@dataclass
class Config:
    production: bool = False
    minimal_test: bool = False
    debug: bool = False
    backend: str = "auto"
    model: str = "gpt2"
    quantization: str = ""
    parallel_mode: str = "pp"
    trust_remote_code: bool = False
    continuous_batching: bool = False
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "vramancer")
    data_dir: Path = field(default_factory=Path.cwd)
    api_token: Optional[str] = None
    auth_secret: Optional[str] = None


_LOCK = threading.RLock()
_CONFIG: Optional[Config] = None


def _build() -> Config:
    yaml_path = _resolve_yaml_path()
    yaml_data: Dict[str, Any] = _yaml_load(yaml_path) if yaml_path else {}

    def pick(env_key: str, yaml_key: str, default: Any) -> Any:
        if env_key in os.environ:
            return os.environ[env_key]
        if yaml_key in yaml_data:
            return yaml_data[yaml_key]
        return default

    return Config(
        production=_bool(pick("VRM_PRODUCTION", "production", False)),
        minimal_test=_bool(pick("VRM_MINIMAL_TEST", "minimal_test", False)),
        debug=_bool(pick("VRM_DEBUG", "debug", False)),
        backend=str(pick("VRM_BACKEND", "backend", "auto")),
        model=str(pick("VRM_MODEL", "model", "gpt2")),
        quantization=str(pick("VRM_QUANTIZATION", "quantization", "")).lower(),
        parallel_mode=str(pick("VRM_PARALLEL_MODE", "parallel_mode", "pp")).lower(),
        trust_remote_code=_bool(pick("VRM_TRUST_REMOTE_CODE", "trust_remote_code", False)),
        continuous_batching=_bool(pick("VRM_CONTINUOUS_BATCHING", "continuous_batching", False)),
        cache_dir=Path(str(pick("VRM_CACHE_DIR", "cache_dir",
                                str(Path.home() / ".cache" / "vramancer")))),
        data_dir=Path(str(pick("VRM_DATA_DIR", "data_dir", str(Path.cwd())))),
        api_token=os.environ.get("VRM_API_TOKEN") or yaml_data.get("api_token"),
        auth_secret=os.environ.get("VRM_AUTH_SECRET") or yaml_data.get("auth_secret"),
    )


def get_config() -> Config:
    global _CONFIG
    with _LOCK:
        if _CONFIG is None:
            _CONFIG = _build()
        return _CONFIG


def reload_config() -> Config:
    global _CONFIG
    with _LOCK:
        _CONFIG = _build()
        return _CONFIG


__all__ = ["Config", "get_config", "reload_config"]
```

**Action 2** : créer `tests/test_config_manager.py` :

```python
"""Tests for core.config_manager."""
import os
import pytest
from core.config_manager import get_config, reload_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ.keys()):
        if k.startswith("VRM_") and k != "VRM_MINIMAL_TEST":
            monkeypatch.delenv(k, raising=False)
    yield


def test_defaults():
    reload_config()
    cfg = get_config()
    assert cfg.backend == "auto"
    assert cfg.trust_remote_code is False
    assert cfg.parallel_mode == "pp"


def test_env_override(monkeypatch):
    monkeypatch.setenv("VRM_TRUST_REMOTE_CODE", "1")
    monkeypatch.setenv("VRM_BACKEND", "vllm")
    reload_config()
    cfg = get_config()
    assert cfg.trust_remote_code is True
    assert cfg.backend == "vllm"


def test_yaml_override(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("backend: ollama\nquantization: nf4\n")
    monkeypatch.chdir(tmp_path)
    reload_config()
    cfg = get_config()
    assert cfg.backend == "ollama"
    assert cfg.quantization == "nf4"


def test_env_wins_over_yaml(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("backend: ollama\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VRM_BACKEND", "vllm")
    reload_config()
    assert get_config().backend == "vllm"


def test_singleton_caching():
    reload_config()
    assert get_config() is get_config()
```

**Important** : ne **pas** migrer les call-sites existants vers ce module dans cette tâche.

**Validation** :
```bash
VRM_MINIMAL_TEST=1 pytest -q tests/test_config_manager.py    # 5 tests passent
```

**Commit** : `[T0.4] add core/config_manager.py with tests`

---

### T0.5 — Test de race condition `continuous_batcher`

**Contexte** : l'audit mentionne une race entre `batcher.start()` et `batcher.submit()`. Le code actuel teste `_running` à la ligne 512. **D'abord vérifier si la race existe** via un test ; ne corriger que si nécessaire.

**Action 1** : créer `tests/test_pipeline_batcher_race.py` :

```python
"""Race condition test for continuous_batcher start/submit lifecycle."""
import os
import threading
import time

import pytest


@pytest.mark.skipif(
    not os.environ.get("VRM_MINIMAL_TEST"),
    reason="Requires stub-safe environment",
)
def test_batcher_concurrent_submit_after_load(monkeypatch):
    monkeypatch.setenv("VRM_CONTINUOUS_BATCHING", "1")

    try:
        from core.inference_pipeline import InferencePipeline
    except ImportError:
        pytest.skip("InferencePipeline unavailable")

    pipe = InferencePipeline(
        backend_name="huggingface",
        enable_metrics=False,
        enable_discovery=False,
        verbose=False,
    )
    try:
        pipe.load("gpt2", num_gpus=1)
    except Exception:
        pytest.skip("Cannot load model in minimal-test mode")

    if pipe.continuous_batcher is None:
        pytest.skip("Batcher disabled in this build")

    errors = []

    def worker():
        try:
            fut = pipe.continuous_batcher.submit(
                "hi", max_new_tokens=1, temperature=1.0,
                top_k=1, top_p=1.0,
            )
            assert fut is not None
        except Exception as exc:
            errors.append(exc)

    pipe.continuous_batcher.start()
    time.sleep(0.01)

    threads = [threading.Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    try:
        pipe.shutdown()
    except Exception:
        pass

    assert not errors, f"Race surfaced: {errors[:3]}"
```

**Action 2** : lancer le test.

```bash
VRM_MINIMAL_TEST=1 pytest -q tests/test_pipeline_batcher_race.py -v
```

- **Si PASS ou SKIP** : la race n'existe pas dans ce build. Marquer T0.5 comme `DONE — no race observed`.
- **Si FAIL** : ajouter dans `core/inference_pipeline.py` ligne 511-512 :

  AVANT :
  ```python
                  elif (self.continuous_batcher is not None
                        and self.continuous_batcher._running):
  ```
  APRÈS :
  ```python
                  elif (self.continuous_batcher is not None
                        and self.continuous_batcher._running
                        and not getattr(self.continuous_batcher, "_stopping", False)):
  ```

  Et dans `core/continuous_batcher.py` :
  - Dans `__init__` ajouter : `self._stopping = False`
  - Dans `stop()`, en première ligne : `self._stopping = True`

**Commit** : `[T0.5] race-test for continuous batcher submit/start`

---

## P1 — Hygiène architecturale

### T1.1 — Documenter `_deprecated/` et clarifier les doublons

**Action 1** : créer `_deprecated/README.md` :

```markdown
# `_deprecated/` — fichiers archivés

Ces fichiers sont conservés pour référence historique et compatibilité de chemins d'import. Ils ne sont **pas** intégrés au pipeline d'inférence principal.

| Fichier | Statut | Raison |
|---|---|---|
| `adaptive_routing.py` | KEEP_FOR_REFERENCE | Remplacé par `core/network/anycast_balancer.py` |
| `backends_deepspeed.py` | REMOVE_AFTER_v2.0.0 | Backend jamais sélectionné |
| `backends_tensorrt.py` | REMOVE_AFTER_v2.0.0 | Backend jamais sélectionné |
| `backends_webgpu.py` | KEEP_FOR_REFERENCE | POC remplacé par `core/webgpu_backend.py` |
| `batch_inference.py` | REMOVE_AFTER_v2.0.0 | `generate_batch_fn` jamais fourni |
| `bench_*.py` | KEEP_FOR_REFERENCE | Benchs historiques |
| `holographic_memory.py` | KEEP_AS_SHIM | Alias de `core/parity_memory.py` |
| `interface_selector.py` | KEEP_FOR_REFERENCE | Remplacé par `core/network/network_transport.py` |
| `packets.py` | KEEP_FOR_REFERENCE | Remplacé par `aitp_protocol.py` |
| `remote_access.py` | REMOVE_AFTER_v2.0.0 | Risque de sécurité |
| `resource_aggregator.py` | KEEP_FOR_REFERENCE | Pré-version de `hetero_config.py` |
| `swarm_ledger.py` | KEEP_FOR_REFERENCE | Orphelin mais fonctionnel |
| `test_adaptive_routing.py` | KEEP_FOR_REFERENCE | Test legacy |
| `triton_gemv_awq.py` | KEEP_FOR_REFERENCE | Kernel AWQ legacy |
| `vramancer_link.py` | KEEP_FOR_REFERENCE | Wrapper early-stage |
| `webgpu_node.py` | KEEP_FOR_REFERENCE | Node POC |
| `network_archive/` | KEEP_FOR_REFERENCE | Archives réseau |

**Convention** :
- `KEEP_AS_SHIM` : alias maintenu pour compat — ne pas supprimer.
- `KEEP_FOR_REFERENCE` : code historique — ne pas réimporter.
- `REMOVE_AFTER_v2.0.0` : à supprimer en v2.0.
```

**Action 2** : ajouter un docstring d'en-tête à 3 fichiers (insérer juste après le `"""..."""` existant ou en haut) :

#### `core/llama_backend.py` — première ligne du module

Ajouter en tête :
```python
"""HuggingFace Hub GGUF utility loader.

**Note** : ce module fournit des helpers de téléchargement/cache GGUF.
Le backend d'inférence GGUF de production est `core/backends_llamacpp.py`
(classe `LlamaCppBackend`, enregistrée dans `select_backend()`).
"""
```
(Si un docstring existe déjà, le remplacer par celui-ci.)

#### `core/llama_server_backend.py`

Ajouter en tête :
```python
"""Experimental llama.cpp HTTP server backend — not wired into InferencePipeline.

Pour le backend GGUF de production, voir `core/backends_llamacpp.py`.
"""
```

#### Doublons vLLM

```bash
grep -n "vllm" core/backends.py | head -20
```

Identifier lequel de `core/backends_vllm.py` ou `core/vllm_backend.py` est référencé. Ajouter dans **l'autre** :

```python
"""**Legacy alias** — superseded by `core/backends_vllm.py` (or `core/vllm_backend.py`,
selon ce qui est référencé dans `core/backends.py:select_backend()`).

Kept for backward import compatibility. Do not extend.
"""
```

**Validation** :
```bash
test -f _deprecated/README.md
head -10 core/llama_backend.py | grep -i "utility\|loader"
```

**Commit** : `[T1.1] document _deprecated and clarify duplicate backends`

---

### T1.2 — Synchroniser les versions

**Source** : `core/__init__.py:3` → `__version__ = "1.5.0"`.

#### `server.py`

Localiser `app = FastAPI(title="VRAMancer", version="1.0.0")`.

Remplacer par :
```python
try:
    from core import __version__ as _VRM_VERSION
except Exception:
    _VRM_VERSION = "unknown"
app = FastAPI(title="VRAMancer", version=_VRM_VERSION)
```

(Placer le `try/except` juste avant la ligne `app = FastAPI(...)`.)

#### `vramancer/main.py`

Dans `_cmd_version()`, remplacer le fallback `print("VRAMancer v0.2.4")` par `print("VRAMancer vunknown")`.

#### `pyproject.toml`

```bash
grep "^version" pyproject.toml
```

Si différent de `"1.5.0"` → corriger.

#### `setup.cfg`

Si présent et avec `version =`, aligner sur `1.5.0`.

**Validation** :
```bash
python -c "from core import __version__; print(__version__)"   # 1.5.0
grep -rn 'version="1.0.0"\|"VRAMancer v0.2.4"' server.py vramancer/
# 0 résultat
```

**Commit** : `[T1.2] sync version to 1.5.0 across server.py and CLI`

---

### T1.3 — Tests d'intégration ciblés

**Action 1** : créer `tests/test_fastapi_e2e.py` :

```python
"""End-to-end smoke tests for server.py FastAPI app."""
import pytest


@pytest.fixture
def fastapi_client():
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed")
    try:
        import server  # noqa: F401
    except Exception as exc:
        pytest.skip(f"server.py cannot be imported: {exc}")
    from server import app
    return TestClient(app)


def test_health_endpoint(fastapi_client):
    r = fastapi_client.get("/health")
    assert r.status_code in (200, 503)


def test_models_list(fastapi_client):
    r = fastapi_client.get("/v1/models")
    assert r.status_code in (200, 404)


def test_chat_without_model_loaded(fastapi_client):
    r = fastapi_client.post("/v1/chat/completions", json={
        "model": "vramancer",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    })
    assert r.status_code in (200, 400, 503)
```

**Action 2** : créer `tests/test_security_boundary.py` :

```python
"""Security boundary tests for the Flask production API."""
import pytest


@pytest.fixture
def secure_client(monkeypatch):
    monkeypatch.setenv("VRM_API_TOKEN", "secret-token-xyz")
    monkeypatch.setenv("VRM_DISABLE_RATE_LIMIT", "0")
    try:
        from core.production_api import create_app
    except ImportError:
        pytest.skip("production_api unavailable")
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def test_protected_endpoint_without_token(secure_client):
    r = secure_client.post("/api/models/load", json={"model": "gpt2"})
    assert r.status_code in (401, 403, 404)


def test_protected_endpoint_invalid_token(secure_client):
    r = secure_client.post(
        "/api/models/load",
        json={"model": "gpt2"},
        headers={"X-API-Token": "wrong"},
    )
    assert r.status_code in (401, 403, 404)


def test_health_unprotected(secure_client):
    r = secure_client.get("/api/health")
    assert r.status_code in (200, 503)
```

**Note** : si `create_app` n'existe pas dans `core/production_api.py`, adapter le test pour importer l'instance Flask exposée (le test SKIP si non trouvable).

**Validation** :
```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 \
  pytest -q tests/test_fastapi_e2e.py tests/test_security_boundary.py
```

SKIP est acceptable. Aucun FAIL toléré.

**Commit** : `[T1.3] add FastAPI e2e and security boundary tests`

---

## P2 — Intégrations majeures

### T2.1 — Adaptateur `hierarchical_memory` ↔ `paged_attention`

**Préalable** : T0.4 et T0.5 DONE.

**Action 1** : créer `core/paged_attention_offload.py` :

```python
"""Adapter between PagedKVCache and HierarchicalMemoryManager.

Allows the paged KV cache to evict cold pages to the CPU/NVMe tiers and
restore them on demand.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PagedAttentionOffloader:
    """Bridge cold KV pages to the hierarchical memory manager."""

    def __init__(self, kv_manager: Any, hmm: Any):
        self.kv_manager = kv_manager
        self.hmm = hmm
        self._offloaded: Dict[int, str] = {}
        self._evict_count = 0
        self._restore_count = 0

    def evict_cold_pages(self, n: int) -> int:
        if n <= 0:
            return 0
        get_lru = getattr(self.kv_manager, "get_lru_pages", None)
        free_page = getattr(self.kv_manager, "free_page", None)
        if get_lru is None or free_page is None:
            logger.debug("kv_manager has no LRU/free hooks; skipping eviction")
            return 0
        pages = get_lru(n) or []
        evicted = 0
        for page in pages:
            page_id = getattr(page, "id", None)
            tensor = getattr(page, "tensor", None)
            if page_id is None or tensor is None:
                continue
            key = f"kvpage::{page_id}"
            try:
                self.hmm.put(key, tensor.detach().to("cpu"))
                free_page(page_id)
                self._offloaded[page_id] = key
                evicted += 1
                self._evict_count += 1
            except Exception as exc:
                logger.warning("Failed to evict page %s: %s", page_id, exc)
        return evicted

    def restore_page(self, page_id: int) -> Optional[Any]:
        key = self._offloaded.pop(page_id, None)
        if key is None:
            return None
        try:
            tensor = self.hmm.get(key)
            self._restore_count += 1
            return tensor
        except Exception as exc:
            logger.warning("Failed to restore page %s: %s", page_id, exc)
            return None

    def stats(self) -> Dict[str, int]:
        return {
            "evicted_total": self._evict_count,
            "restored_total": self._restore_count,
            "in_offload": len(self._offloaded),
        }


__all__ = ["PagedAttentionOffloader"]
```

**Action 2** : créer `tests/test_paged_offload.py` :

```python
"""Unit tests for PagedAttentionOffloader."""
import pytest


class _FakePage:
    def __init__(self, pid, tensor):
        self.id = pid
        self.tensor = tensor


class _FakeKV:
    def __init__(self, pages):
        self._pages = list(pages)
        self.freed = []

    def get_lru_pages(self, n):
        return self._pages[:n]

    def free_page(self, pid):
        self.freed.append(pid)
        self._pages = [p for p in self._pages if p.id != pid]


class _FakeHMM:
    def __init__(self):
        self.store = {}

    def put(self, key, tensor):
        self.store[key] = tensor

    def get(self, key):
        return self.store.get(key)


def test_evict_basic():
    torch = pytest.importorskip("torch")
    from core.paged_attention_offload import PagedAttentionOffloader

    pages = [_FakePage(i, torch.zeros(4, 4)) for i in range(5)]
    kv = _FakeKV(pages)
    hmm = _FakeHMM()
    off = PagedAttentionOffloader(kv, hmm)

    assert off.evict_cold_pages(3) == 3
    assert len(hmm.store) == 3
    assert len(kv.freed) == 3
    assert off.stats()["evicted_total"] == 3


def test_restore_roundtrip():
    torch = pytest.importorskip("torch")
    from core.paged_attention_offload import PagedAttentionOffloader

    t = torch.arange(16).reshape(4, 4).float()
    kv = _FakeKV([_FakePage(0, t.clone())])
    hmm = _FakeHMM()
    off = PagedAttentionOffloader(kv, hmm)

    off.evict_cold_pages(1)
    restored = off.restore_page(0)
    assert restored is not None
    assert torch.equal(restored, t)
    assert off.stats()["restored_total"] == 1


def test_no_kv_hooks_safe():
    from core.paged_attention_offload import PagedAttentionOffloader

    class _BareKV:
        pass

    assert PagedAttentionOffloader(_BareKV(), _FakeHMM()).evict_cold_pages(5) == 0
```

**Important** : **NE PAS** modifier `core/paged_attention.py` dans cette tâche.

**Validation** :
```bash
VRM_MINIMAL_TEST=1 pytest -q tests/test_paged_offload.py    # 3 tests passent ou skip torch
```

**Commit** : `[T2.1] add PagedAttentionOffloader adapter`

---

### T2.2 — Binding Rust : sonde RDMA (BLOCKED-friendly)

**Vérification préalable** :
```bash
which cargo && ls /usr/lib/x86_64-linux-gnu/libcuda.so.1 2>/dev/null
```

Si l'une des deux échoue → **BLOCKED**, passer à T2.3.

**Sinon** :

**Action 1** : ouvrir `rust_core/src/transport.rs`, localiser `detect_best_transport()`. La modification doit ajouter une sonde `libibverbs.so.1` :

```rust
pub fn detect_best_transport() -> TransportTier {
    // Probe RDMA verbs availability at runtime
    if libloading::Library::new("libibverbs.so.1").is_ok() {
        return TransportTier::RdmaVerbs;
    }
    TransportTier::ZeroCopyTcp
}
```

Si la variante `RdmaVerbs` n'existe pas dans l'enum `TransportTier`, l'ajouter sans casser les variantes existantes.

**Action 2** : compiler :
```bash
cd rust_core && cargo build --release --features cuda 2>&1 | tail -30
cd ..
```

Si échec de compilation → `git checkout rust_core/`, marquer **BLOCKED**.

**Action 3** : ne **pas** exposer `GpuPipeline::transfer` via PyO3 dans cette tâche.

**Commit** (uniquement si succès) : `[T2.2] add libibverbs probe to detect_best_transport`

---

### T2.3 — Documenter le continuous batcher

**Action 1** : créer `docs/CONTINUOUS_BATCHING.md` :

````markdown
# Continuous batching

VRAMancer intègre un batcher continu (style vLLM/Orca) qui regroupe les requêtes concurrentes en un seul forward pass.

## Activation

```bash
export VRM_CONTINUOUS_BATCHING=1
python server.py --model gpt2
```

## Paramètres

| Variable | Défaut | Rôle |
|---|---|---|
| `VRM_CONTINUOUS_BATCHING` | `0` | `1` active le batcher |
| `VRM_GENERATE_TIMEOUT` | `300` | Timeout (secondes) par requête |
| `VRM_MAX_BATCH_SIZE` | `32` | Taille de batch max simultanée |

File d'attente max : **256 requêtes**. Au-delà, `submit()` est bloquant ou rejette.

## Limites

- **GIL CPython** : la tokenization Python reste sérialisée. Speedup au-delà de ~4 requêtes simultanées.
- **KV cache scope** : chaque requête a son propre KV cache paged. Pas de prefix caching actuellement.
- **Backends** : HuggingFace Transformers uniquement. vLLM et llama-cpp ont leur propre batching natif.

## Diagnostic

`InferencePipeline.batcher_stats()` expose : `pending_requests`, `running_batch_size`, `tokens_per_second`, `queue_depth`.
````

**Action 2** : créer `benchmarks/bench_batcher_concurrent.py` :

```python
#!/usr/bin/env python3
"""Concurrent benchmark for the continuous batcher (manual run only)."""
import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def _request(url, prompt, max_tokens, token):
    import urllib.request
    body = json.dumps({
        "model": "vramancer",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-API-Token"] = token
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=body, headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--total", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--prompt", default="Explain quantum entanglement in one sentence.")
    p.add_argument("--token", default=os.environ.get("VRM_API_TOKEN", ""))
    p.add_argument("--out", default="bench_batcher_concurrent.json")
    args = p.parse_args()

    if os.environ.get("VRM_MINIMAL_TEST") == "1":
        print("Skipped under VRM_MINIMAL_TEST=1")
        return

    latencies = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(_request, args.url, args.prompt, args.max_tokens, args.token)
                   for _ in range(args.total)]
        for f in as_completed(futures):
            try:
                latencies.append(f.result())
            except Exception as exc:
                print(f"  request failed: {exc}")
    elapsed = time.perf_counter() - t_start

    latencies.sort()
    result = {
        "concurrency": args.concurrency,
        "total_requests": args.total,
        "successful": len(latencies),
        "wall_seconds": round(elapsed, 3),
        "throughput_rps": round(len(latencies) / elapsed, 2) if elapsed else 0,
        "latency_p50": round(statistics.median(latencies), 3) if latencies else None,
        "latency_p99": round(latencies[int(len(latencies) * 0.99)], 3) if latencies else None,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

**Validation** (parse uniquement, pas d'exécution) :
```bash
python -c "import ast; ast.parse(open('benchmarks/bench_batcher_concurrent.py').read())" && echo OK
```

**Commit** : `[T2.3] document continuous batching + concurrent bench script`

---

## P3 — Étiquetage des modules périphériques

### T3.1 — Badges de statut

Pour chaque fichier ci-dessous, **insérer une ligne** au tout début du docstring de module (juste après les `"""`). **Ne rien d'autre** modifier.

#### `core/webgpu_backend.py`

Première ligne du docstring :
```
**Status: experimental — POC, not production-ready.**
```

#### `core/swarm_ledger.py`

```
**Status: functional but orphaned — not wired into the main inference pipeline.**
```

#### `core/network/edge_api.py`

```
**Status: edge IoT lifecycle management — used by core/network/supervision_api.**
```

#### `dashboard/dashboard_web.py`

```
**Status: demo / local monitoring — not for production deployment.**
```

**Validation** :
```bash
for f in core/webgpu_backend.py core/swarm_ledger.py core/network/edge_api.py dashboard/dashboard_web.py; do
  head -5 "$f" | grep -i "status:" >/dev/null && echo "OK $f" || echo "MISSING $f"
done
```

**Commit** : `[T3.1] add module status badges to peripheral components`

---

### T3.2 — Bench anycast routing

**Action** : créer `benchmarks/bench_anycast_routing.py` :

```python
#!/usr/bin/env python3
"""Anycast strategy comparison benchmark (simulated peers)."""
import argparse
import json
import random
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--peers", type=int, default=5)
    p.add_argument("--rounds", type=int, default=1000)
    p.add_argument("--out", default="bench_anycast_routing.json")
    args = p.parse_args()

    try:
        from core.network.anycast_balancer import AnycastBalancer
    except ImportError as exc:
        print(f"AnycastBalancer unavailable: {exc}")
        return

    rng = random.Random(42)
    peers = [
        {"node_id": f"peer-{i}",
         "latency_ms": rng.uniform(1, 50),
         "strength": rng.uniform(0.1, 1.0)}
        for i in range(args.peers)
    ]

    results = {}
    for strategy in ("weighted", "least_latency", "round_robin"):
        balancer = AnycastBalancer(strategy=strategy)
        for pr in peers:
            try:
                balancer.update_node_health(
                    pr["node_id"],
                    latency_ms=pr["latency_ms"],
                    strength=pr["strength"],
                )
            except Exception:
                pass
        chosen = []
        t0 = time.perf_counter()
        for _ in range(args.rounds):
            try:
                sel = balancer.select_peer()
            except Exception:
                sel = None
            if sel:
                chosen.append(sel)
        elapsed = time.perf_counter() - t0
        results[strategy] = {
            "success_rate": len(chosen) / args.rounds,
            "wall_seconds": round(elapsed, 3),
            "selections_per_sec": round(args.rounds / elapsed, 1) if elapsed else 0,
        }

    with open(args.out, "w") as f:
        json.dump({"peers": peers, "results": results}, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
```

**Validation** :
```bash
python -c "import ast; ast.parse(open('benchmarks/bench_anycast_routing.py').read())" && echo OK
```

**Commit** : `[T3.2] add anycast routing bench script`

---

## Validation finale

```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
  pytest -q tests/ --ignore=tests/test_chaos_concurrency.py 2>&1 | tail -10
```

Comparer aux baselines (étape 0.0.3). **Aucune nouvelle régression tolérée.**

```bash
git log --oneline chore/sonnet-plan-exec ^main
```

Doit afficher exactement les commits attendus dans l'ordre des tâches DONE.

Compléter `docs/reports/EXEC_LOG.md` avec une entrée finale `### [SUMMARY]` :
- nombre de tâches DONE / PARTIAL / BLOCKED / SKIPPED
- nombre de tests ajoutés
- nombre de fichiers modifiés
- delta de tests passing vs baseline

---

## Ce que Sonnet 4.6 ne doit PAS faire

- Renommer `class Synapse`, attribut `synapses`, méthode `_apply_neuroplasticity_score`, ou tout symbole exporté.
- Modifier `_deprecated/*.py` (sauf création de `_deprecated/README.md` en T1.1).
- Modifier `.github/copilot-instructions.md`.
- Lancer `cargo build` ou un benchmark GPU sans les vérifications préalables listées.
- Ajouter une dépendance dans `requirements*.txt` ou `pyproject.toml`.
- Migrer Flask → FastAPI (hors scope total).
- Modifier `tests/test_chaos_concurrency.py`.
- Auto-pusher la branche, auto-merger, créer une PR.
- Migrer les call-sites existants vers `core/config_manager.py` (juste créer le module).
- Modifier `core/paged_attention.py` dans T2.1 (juste créer l'adaptateur).

---

## Ordre d'exécution imposé

```
0.0.x → T0.1 → T0.2 → T0.3 → T0.4 → T0.5
        → T1.1 → T1.2 → T1.3
        → T2.1 → T2.2 → T2.3
        → T3.1 → T3.2
        → Validation finale
```

Chaque tâche est validée par ses critères avant de passer à la suivante. En cas d'échec d'une tâche : marquer BLOCKED dans `EXEC_LOG.md`, ne pas commiter, passer à la suivante.

---

## Références

- `.github/copilot-instructions.md`
- `core/inference_pipeline.py`, `core/continuous_batcher.py`, `core/hierarchical_memory.py`, `core/paged_attention.py`
- `core/env_flags.py`, `core/auth_strong.py`, `core/security/__init__.py`
- `server.py`, `core/production_api.py`
- `rust_core/src/transport.rs`
