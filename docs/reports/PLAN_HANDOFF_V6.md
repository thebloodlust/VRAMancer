# PLAN HANDOFF V6 — Tâches restantes pour la lending pool

> **Public** : développeur IA d'exécution (Sonnet, Haiku, ou junior dev humain).
> **Auteur** : Opus 4.7 (1M context), session V6 du 2026-05-08.
> **Branche de départ** : `feat/v6-lending-cooperative` @ `73218f9`.
> **Repo racine** : `/home/jeremie/VRAMancer/VRAMancer/`.

Ce document contient les tâches V6 restantes après V6.A + V6.B (fix double-reserve + lending pool dans `accelerate.max_memory`). Chaque tâche est auto-suffisante. Tu n'as pas besoin de comprendre pourquoi V6.A/B ont été faites — lis juste `resultat_v5.md` section "V6 — Lending pool cooperative placement" si tu veux le contexte.

---

## Règles globales (lis-les AVANT de coder)

### R1. Honnêteté des claims dans les résultats

> **Règle d'or** : les markdowns/JSON de bench doivent décrire ce qui a été *mesuré*, pas ce qu'on *espérait* mesurer.

- Si une feature s'active mais ne fait rien d'observable, écris-le explicitement : *"pool s'instancie mais aucune lease formelle n'est créée pendant l'inférence"*.
- Si un bench échoue, ne le maquille pas. Utilise les tags `[BLOCKED@PX]`, `[NEGATIVE@PX]`, `[SKIPPED@PX]`, `[PARTIAL@PX]`.
- **Ne jamais** prétendre "showcase fonctionnel" si le test ne tourne pas end-to-end.
- L'utilisateur est attentif — il a déjà repéré une affirmation fausse en V5 P13. Ne refais pas l'erreur.

### R2. Workflow git

- **Branche** : reste sur `feat/v6-lending-cooperative`. Ne crée pas de nouvelle branche sauf instruction explicite.
- **Commits atomiques** : un commit par tâche (V6.C, V6.D, etc.). Pas de commits "WIP" ou "fix typo" — squash si nécessaire.
- **Format des messages** : voir les commits `74dc904` (V6.A) et `73218f9` (V6.B) pour le style. Header court (`v6c:` ou `v6d:`), body multi-paragraphes décrivant le **pourquoi** et la **validation**.
- **Trailer obligatoire** :
  ```
  Co-Authored-By: <ton modèle> <noreply@anthropic.com>
  ```
- **Ne jamais** : `git push --force`, `git reset --hard`, `git rebase -i`, modifier un commit déjà publié, skip les hooks pre-commit.
- **Si pre-commit échoue** : fix l'issue, re-stage, **nouveau** commit. Pas d'`--amend`.

### R3. Validation systématique

Pour chaque modification :

1. **AST check** : `python -c "import ast; ast.parse(open('chemin.py').read())"` — passe ou stop.
2. **Tests unitaires lending** :
   ```bash
   VRM_MINIMAL_TEST=1 python -m pytest tests/test_vram_lending.py tests/test_lending_stress.py tests/test_rebar_lending.py --no-cov -x
   ```
   Tu dois voir `95 passed` (ou plus si tu en ajoutes). Si une régression — analyse, ne contourne pas.
3. **Bench régression** (pour les tâches qui touchent l'inférence path) :
   ```bash
   python benchmarks/bench_lending_hetero_real.py \
     --model Qwen/Qwen2.5-14B-Instruct \
     --num-gpus 2 --max-new 32 --warmup 3 --timeout 1500 \
     --out-suffix _<tag>_regression
   ```
   Le tok/s doit rester dans la fourchette **13–15 tok/s** sur LENDING_ON. Si chute > 10 % → investiger.

### R4. Environnement spécifique du repo

- **Hardware** : RTX 5070 Ti (GPU0, SM 12.0, 16 GB) + RTX 3090 (GPU1, SM 8.6, 24 GB) sous Proxmox VFIO. ReBAR ACTIF sur les deux.
- **Stack** : Python 3.12, torch 2.11.0+cu130, transformers 5.8.0, dans `.venv/` à la racine.
- **`LD_LIBRARY_PATH` cu13** : pour les modèles qui chargent BitsAndBytes / nvJitLink, prefixer :
  ```bash
  LD_LIBRARY_PATH=/home/jeremie/VRAMancer/VRAMancer/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
  ```
- **NVML pour les helpers GPU** : `torch.cuda.get_device_properties(i)` et `torch.cuda.mem_get_info(i)` deviennent invalides après `CUDA_VISIBLE_DEVICES=N` car torch cache l'état au premier `cuda.init()`. Utilise `pynvml` qui est insensible à CVD. Voir `benchmarks/bench_lending_hetero_real.py:_gpu_topology()` pour le pattern.
- **`VRM_FORCE_MULTI_GPU=1`** : indispensable pour exercer la lending pool — sans ça, `_auto_select_num_gpus()` ramène `num_gpus` à 1 quand le modèle tient sur un GPU et shortcut le pool. Ce flag est déjà settled par défaut dans `bench_lending_hetero_real.py`.
- **`VRM_VRAM_LENDING=0/1`** : contrôle l'activation du pool. Le bench fait l'A/B automatiquement.

### R5. Quand t'arrêter

Si tu rencontres :

- Un test qui se met à échouer alors qu'il passait avant → **stop**, analyse, ne désactive pas le test.
- Un bench qui régresse > 20 % → **stop**, investiger.
- Une dépendance manquante qui demande > 2 essais d'install → **stop**, demande à l'utilisateur (ne lance pas une chasse aux deps qui peut casser tout l'env).
- Une exception silencieuse que tu ne comprends pas → **stop**, log avec `exc_info=True`, demande.

Mieux vaut livrer 1 tâche propre que 3 tâches cassées.

---

## Tâche V6.C — `allocate_on_lease()` via `TransferManager`

### Objectif

Quand le borrower veut utiliser la VRAM louée du lender, le data plane passe par `TransferManager` au lieu d'un `torch.zeros()` direct. Ça active automatiquement les stratégies déjà existantes : Rust DtoD ≥ 512 KB, ReBAR pipelined ≥ 64 MB, CPU staged en fallback. **Aujourd'hui le pool ne déplace aucune donnée à l'inférence** — la louée est juste un buffer alloué côté lender. V6.C corrige ça.

### Pourquoi

`TransferManager` (`core/transfer_manager.py`) a déjà toute l'intelligence (P2P probing, ReBAR detection, Rust GpuPipeline persistant). Le pool n'utilise rien de tout ça — c'est gâché. V6.C le branche.

### Fichiers à modifier

#### `core/vram_lending.py`

1. **Importer `TransferManager`** au top du fichier (après les autres imports core) :
   ```python
   try:
       from core.transfer_manager import TransferManager  # noqa: F401
   except ImportError:
       TransferManager = None  # type: ignore
   ```

2. **Lire la signature actuelle** de `allocate_on_lease()` à la ligne ~567 (cherche `def allocate_on_lease`). Garde rétro-compatibilité (même retour `torch.Tensor`).

3. **Ajouter une nouvelle méthode** `transfer_into_lease(lease, src_tensor)` qui :
   - vérifie que `lease.state == LeaseState.ACTIVE`
   - récupère le tensor sur le lender (déjà alloué via `allocate_on_lease`)
   - appelle `self._transfer_manager.transfer(src_gpu, lease.owner_gpu, src_tensor)` pour copier `src_tensor` dans le buffer du lease
   - retourne le tensor sur le lender (le résultat de transfer)
   - capture les exceptions et log avec `exc_info=True`

4. **Ajouter une nouvelle méthode** `transfer_from_lease(lease, dst_gpu)` qui :
   - vérifie que `lease.state == LeaseState.ACTIVE` et que `lease.tensor_ref is not None`
   - appelle `self._transfer_manager.transfer(lease.owner_gpu, dst_gpu, lease.tensor_ref)`
   - retourne le tensor matérialisé sur `dst_gpu`

5. **Pas modifier** `allocate_on_lease()` lui-même — l'allocation reste sur le lender, c'est correct.

Code minimal de référence (à intégrer aux endroits adaptés) :

```python
def transfer_into_lease(self, lease: "VRAMLease", src_tensor: Any) -> Optional[Any]:
    """Copy src_tensor into the leased VRAM buffer using TransferManager.

    Routes through the Rust DtoD bypass (≥ 512 KB) or ReBAR pipelined
    transport (≥ 64 MB) — see core.transfer_manager. Falls back to
    torch's tensor.copy_() if no TransferManager was injected.
    """
    if not _TORCH or _MINIMAL:
        return None
    if lease.state != LeaseState.ACTIVE or lease.tensor_ref is None:
        log.warning("transfer_into_lease on non-active lease %s", lease.lease_id)
        return None
    src_gpu = src_tensor.device.index if hasattr(src_tensor, 'device') else None
    if src_gpu is None:
        return None
    try:
        if self._transfer_manager is not None and src_gpu != lease.owner_gpu:
            _, dst = self._transfer_manager.transfer(
                src_gpu, lease.owner_gpu, src_tensor,
            )
            return dst
        # Same GPU or no transfer manager: direct copy.
        lease.tensor_ref.copy_(src_tensor)
        return lease.tensor_ref
    except Exception as e:
        log.warning("transfer_into_lease failed for %s: %s",
                    lease.lease_id, e, exc_info=True)
        return None


def transfer_from_lease(self, lease: "VRAMLease", dst_gpu: int) -> Optional[Any]:
    """Materialise a copy of the leased buffer on dst_gpu via TransferManager."""
    if not _TORCH or _MINIMAL:
        return None
    if lease.state != LeaseState.ACTIVE or lease.tensor_ref is None:
        return None
    try:
        if self._transfer_manager is not None and dst_gpu != lease.owner_gpu:
            _, dst = self._transfer_manager.transfer(
                lease.owner_gpu, dst_gpu, lease.tensor_ref,
            )
            return dst
        return lease.tensor_ref  # Same GPU, no transfer needed.
    except Exception as e:
        log.warning("transfer_from_lease failed for %s: %s",
                    lease.lease_id, e, exc_info=True)
        return None
```

Tu peux modifier ce code, tant que la sémantique est respectée. **Important** : `lease.tensor_ref` est positionné dans `allocate_on_lease()` actuel — vérifie que c'est bien le tensor backé par le buffer leasé.

#### `tests/test_lending_data_plane.py` (nouveau)

Crée un nouveau fichier de tests qui couvre :

```python
"""V6.C — TransferManager-routed data plane for lending leases."""
import pytest
import torch
from core.vram_lending import VRAMLendingPool, LendingPolicy

_HAS_2GPU = torch.cuda.is_available() and torch.cuda.device_count() >= 2


@pytest.mark.skipif(not _HAS_2GPU, reason="needs 2 CUDA devices")
def test_transfer_into_lease_cross_gpu():
    """Writing to a lease on GPU1 from a tensor on GPU0 routes through
    TransferManager and the data lands correctly on GPU1."""
    from core.transfer_manager import TransferManager
    tm = TransferManager(protocol="nccl", secure=False, verbose=False)
    pool = VRAMLendingPool(policy=LendingPolicy(buffer_prealloc_ratio=0.0),
                           transfer_manager=tm)
    pool.register_gpu(0, total_bytes=int(1e10), model_bytes=0,
                      device_name="src", pcie_gen=4)
    pool.register_gpu(1, total_bytes=int(1e10), model_bytes=0,
                      device_name="dst", pcie_gen=4)

    lease = pool.borrow(borrower_gpu=0, size_bytes=4 * 1024 * 1024,
                         purpose="test", priority=1, preferred_lender=1)
    assert lease is not None

    leased = pool.allocate_on_lease(lease, shape=(1024, 1024), dtype=torch.float32)
    assert leased is not None
    assert leased.device.index == 1

    src = torch.arange(1024 * 1024, dtype=torch.float32, device="cuda:0").view(1024, 1024)
    result = pool.transfer_into_lease(lease, src)
    assert result is not None
    assert result.device.index == 1
    assert torch.allclose(result.cpu(), src.cpu())

    # Read back to GPU0
    back = pool.transfer_from_lease(lease, dst_gpu=0)
    assert back is not None
    assert back.device.index == 0
    assert torch.allclose(back.cpu(), src.cpu())

    pool.return_lease(lease)


@pytest.mark.skipif(not _HAS_2GPU, reason="needs 2 CUDA devices")
def test_transfer_into_lease_no_transfer_manager_fallback():
    """Without a TransferManager, transfer_into_lease falls back to copy_()."""
    pool = VRAMLendingPool(policy=LendingPolicy(buffer_prealloc_ratio=0.0))
    pool.register_gpu(0, total_bytes=int(1e10), model_bytes=0, device_name="g0")
    pool.register_gpu(1, total_bytes=int(1e10), model_bytes=0, device_name="g1")

    lease = pool.borrow(borrower_gpu=0, size_bytes=1024 * 1024,
                         purpose="test", priority=1, preferred_lender=1)
    leased = pool.allocate_on_lease(lease, shape=(256, 256), dtype=torch.float32)
    assert leased is not None

    # Same-GPU transfer should still work via direct copy
    src_same = torch.ones(256, 256, dtype=torch.float32, device="cuda:1")
    result = pool.transfer_into_lease(lease, src_same)
    assert result is not None
    assert torch.allclose(result, src_same)

    pool.return_lease(lease)
```

Ne suppose **rien** sur l'API exacte du `TransferManager.transfer()` — vérifie sa signature dans `core/transfer_manager.py:483-…` (cherche `def transfer`). Les paramètres réels sont `(source_gpu, target_gpu, tensor, stream=None)` et il retourne `(method_used, output_tensor)`.

### Validation

1. **Tests** :
   ```bash
   VRM_MINIMAL_TEST=1 python -m pytest tests/test_lending_data_plane.py -v --no-cov
   ```
   Doit passer (au moins 2 tests si tu as 2 GPUs).

2. **Tests existants** ne doivent pas régresser :
   ```bash
   VRM_MINIMAL_TEST=1 python -m pytest tests/test_vram_lending.py tests/test_lending_stress.py tests/test_rebar_lending.py --no-cov
   ```
   95 passed minimum.

3. **Bench régression Qwen14B** :
   ```bash
   python benchmarks/bench_lending_hetero_real.py \
     --model Qwen/Qwen2.5-14B-Instruct \
     --num-gpus 2 --max-new 32 --warmup 3 --timeout 1500 \
     --out-suffix _v6c_regression
   ```
   tok/s LENDING_ON doit rester dans 13–15.

### Commit

```
v6c: data plane des leases via TransferManager (Rust P2P + ReBAR)

Avant V6.C, ``VRAMLendingPool.allocate_on_lease()`` allouait simplement
``torch.zeros(..., device=f"cuda:{lease.owner_gpu}")`` — le tensor
existait sur le lender mais aucun transfert effectif n'avait jamais
lieu. La pool était inerte côté data plane.

V6.C ajoute deux helpers :
  - transfer_into_lease(lease, src_tensor) : copie src dans la VRAM
    leasée via TransferManager.transfer(), routant automatiquement
    sur Rust DtoD (≥ 512 KB), ReBAR pipelined (≥ 64 MB), ou CPU
    staged en fallback.
  - transfer_from_lease(lease, dst_gpu) : symétrique en lecture.

Quand aucun TransferManager n'a été injecté, les helpers retombent
sur tensor.copy_() — sémantique préservée pour les tests qui
n'ont pas besoin d'un transport complet.

Validation: 2 nouveaux tests dans tests/test_lending_data_plane.py
(skip si moins de 2 GPUs CUDA), 95 tests existants OK,
Qwen14B regression: <X.XX> tok/s LENDING_ON (vs 13.72 baseline V6.B).

Co-Authored-By: <ton modèle> <noreply@anthropic.com>
```

Remplace `<X.XX>` par le tok/s mesuré et `<ton modèle>` par ton identifiant exact (ex. `Claude Sonnet 4.6`).

### Stop / blocker

Si tu n'arrives pas à invoquer `TransferManager.transfer()` proprement (par ex. il demande NCCL init ou des env vars manquants), **abandonne ce wire et reste sur fallback `copy_()`**. Documente dans le commit message : *"TransferManager.transfer() fails without NCCL — using copy_() fallback only, full Rust P2P routing tracked for V6.C.bis"*.

---

## Tâche V6.D — Hook lending pool dans `paged_attention` KV overflow

### Objectif

Quand le KV cache dépasse la VRAM d'un GPU pendant l'inférence, le path actuel le spille en DRAM (cpu_offload). V6.D fait le spillover **vers la VRAM du voisin via le lending pool** — ça utilise les ~22 GB libres restants sur le 3090 plutôt que la DRAM, ce qui devrait améliorer drastiquement le tok/s sur les contextes longs.

### Pourquoi

C'est le **cas d'usage primaire** documenté de la lending pool (voir docstring `core/vram_lending.py:1-35`). Aujourd'hui le pool n'est jamais appelé depuis `paged_attention` malgré sa raison d'être. V6.D corrige ça.

### Fichiers à investiger d'abord

```bash
ls core/paged_attention*.py
grep -n "overflow\|spill\|offload\|migrate" core/paged_attention.py core/paged_attention_offload.py | head -20
grep -n "lending_pool\|borrow\|lease" core/paged_attention*.py | head -10
```

Le path d'overflow KV existant utilise probablement `_DramDict` ou `PagedAttentionOffloader`. Lis ce code en entier avant de toucher quoi que ce soit.

### Stratégie

1. **Identifier le hook point** dans `paged_attention.py` ou `paged_attention_offload.py` où une page KV est évincée de la VRAM. Cherche `evict`, `migrate`, `_offload_page`.

2. **Avant d'évincer vers DRAM**, demander au pool s'il peut prêter de la VRAM voisine :
   ```python
   if hasattr(self, 'lending_pool') and self.lending_pool is not None:
       lease = self.lending_pool.borrow(
           borrower_gpu=current_gpu,
           size_bytes=page_size_bytes,
           purpose="kv_cache_overflow",
           priority=2,  # medium — KV overflow can be reclaimed
       )
       if lease is not None:
           leased_buffer = self.lending_pool.allocate_on_lease(
               lease, shape=page_shape, dtype=page_dtype,
           )
           if leased_buffer is not None:
               # Use V6.C helper to actually move the page
               self.lending_pool.transfer_into_lease(lease, page_tensor)
               # Track this lease so we can reclaim later
               self._kv_leases[page_id] = lease
               return  # Successfully spilled to lender VRAM
   # Fallback: existing DRAM offload path
   self._offload_to_dram(page_tensor, page_id)
   ```

3. **Côté reclaim** : quand une page KV doit revenir en active VRAM (cache hit), récupérer via `transfer_from_lease(lease, dst_gpu=current_gpu)` puis `pool.return_lease(lease)`.

4. **Ajouter un flag d'env** `VRM_KV_LEND=1` (default `0`, désactivé) pour gater le hook. Évite de casser des cas qui marchent déjà.

### Fichiers attendus à modifier

- `core/paged_attention.py` ou `core/paged_attention_offload.py` (selon où vit le path d'overflow)
- `core/inference_pipeline.py` — injecter `lending_pool` dans le `paged_attention` setup, similaire à comment ça a été fait pour `backend.lending_pool` en V6.B (cherche `setattr(self.backend, 'lending_pool'`)
- `tests/test_paged_attention_lending.py` (nouveau)

### Validation

1. **Test unitaire** : `tests/test_paged_attention_lending.py` qui simule un KV cache surchargé, vérifie que les pages partent vers la lending pool (pas DRAM) quand `VRM_KV_LEND=1`.

2. **Bench long-context** : un nouveau bench `benchmarks/bench_kv_lending_overflow.py` qui teste sur Qwen14B avec un prompt très long (16k tokens) pour saturer le KV cache et observer la différence tok/s avec/sans lending pour KV.

3. **Régression** : Qwen14B BF16 normal (32 tokens) doit rester dans la fourchette 13–15 tok/s — le hook KV ne doit pas s'activer sur des contextes courts.

### Commit

Format identique à V6.C mais avec `v6d:` en header et description du hook KV.

### Stop / blocker

V6.D est **risquée** parce qu'elle touche un hot path. Si après 4 heures de travail tu n'arrives pas à un état stable :

- Garde la modification minimale (ajout du hook gaté par `VRM_KV_LEND=1` mais inactif par défaut).
- Documente honnêtement dans le commit : *"V6.D: KV overflow hook implemented but not validated end-to-end — gate kept off by default, manual testing required"*.
- N'active pas le flag par défaut.

---

## Tâche optionnelle V6.E — Finir le sweep silent excepts

### Objectif

V5 P5 a migré 33 `except Exception: pass` vers `_logger.debug(..., exc_info=True)` dans 3 modules hot path. Il reste ~193 occurrences hors hot path. Cette tâche les sweep par batch.

### Comment

1. Lister les occurrences :
   ```bash
   grep -rn "except Exception:\s*pass\|except:\s*pass" --include='*.py' core/ | grep -v "_test\|tests/\|deprecated"
   ```

2. Par fichier, pour chaque occurrence :
   - Si l'exception est *vraiment* attendue et silencieuse (ex. `try: import optional_module except: pass`), garder mais ajouter un commentaire `# silence intentional: optional dep`.
   - Sinon, remplacer par `_logger.debug("descriptive message", exc_info=True)` (importer le logger si nécessaire).

3. Commits par groupe de fichiers cohérents. Ne fais pas un mega-commit.

### Validation

Tests complets :
```bash
VRM_MINIMAL_TEST=1 python -m pytest tests/ --no-cov -x -k "not gpu and not slow"
```

### Stop

Cette tâche est **éternellement reportable**. Fais une session de 1-2 heures, livre ce que tu as fait, et stop. Pas besoin de tout faire d'un coup.

---

## Tâche V6.F — Fix doublon `resultat_v5.md`

### Objectif

Il y a deux fichiers `resultat_v5.md` :
- `/home/jeremie/VRAMancer/VRAMancer/resultat_v5.md` (à jour, V6 inclus)
- `/home/jeremie/VRAMancer/VRAMancer/docs/reports/resultat_v5.md` (figé au 2026-05-06, dit `[BLOCKED]` pour P13)

Le second est obsolète mais peut induire en erreur (un futur reviewer ne saura pas lequel est canonique).

### Action

Soit :
- **Option A** : supprimer `docs/reports/resultat_v5.md`, ajouter une note dans `docs/reports/PLAN_ACTION_V5.md` qui pointe vers la racine.
- **Option B** : remplacer `docs/reports/resultat_v5.md` par un stub :
  ```markdown
  # resultat_v5.md déplacé
  Voir [/resultat_v5.md](../../resultat_v5.md) à la racine du repo pour le résultat à jour.
  ```

Préférence : **Option B** (plus safe, garde l'index si quelqu'un avait un lien externe).

### Validation

- Pas de `grep -r "docs/reports/resultat_v5.md"` qui ramène des références à du contenu (uniquement le pointeur).

### Commit

```
docs: dedupe resultat_v5.md, point docs/reports/ to root canonical version
```

---

## Référence rapide — où vit quoi

| Composant | Fichier(s) clés |
|-----------|-----------------|
| Lending pool core | `core/vram_lending.py` (1100+ lignes) |
| Pipeline init order (V6.B applied) | `core/inference_pipeline.py:156-345` (la méthode `load()`) |
| Backend HF + max_memory map | `core/backends.py:504-650` (class `HuggingFaceBackend`, `_build_compute_aware_memory_map`) |
| Transfer manager (P2P/ReBAR/CPU staging) | `core/transfer_manager.py:483-680` (méthode `transfer`, stratégies 0-4) |
| Paged attention (probable hook V6.D) | `core/paged_attention.py`, `core/paged_attention_offload.py` |
| Hetero config (compute capability detection) | `core/hetero_config.py` |
| Bench A/B lending | `benchmarks/bench_lending_hetero_real.py` |
| Bench DeepSeek (P13) | `benchmarks/bench_deepseek_engram.py` |
| Tests lending | `tests/test_vram_lending.py`, `tests/test_lending_stress.py`, `tests/test_rebar_lending.py` |

## API clés

### `VRAMLendingPool` (`core/vram_lending.py`)

- `__init__(policy, monitor=None, transfer_manager=None)`
- `register_gpu(gpu_id, total_bytes, model_bytes, device_name, pcie_gen, compute_capability, vendor=None)` → `GPUBudget`
- `update_gpu_usage(gpu_id, model_bytes=None, kv_cache_bytes=None)` → mise à jour post-load
- `borrow(borrower_gpu, size_bytes, purpose, priority, preferred_lender=None)` → `Optional[VRAMLease]`
- `allocate_on_lease(lease, shape, dtype=None)` → `torch.Tensor` sur le lender
- `return_lease(lease)` → `bool`
- `get_budget(gpu_id)` → `Optional[GPUBudget]`
- `pool_capacity()` → `Dict[str, Any]`
- `stats()` → `Dict[str, Any]`
- **V6.B nouveau** : `suggest_placement_budget(model_size_bytes, gpu_ids=None, runtime_headroom_ratio=0.05, cpu_overflow_gb=48.0)` → `Optional[Dict[Any, str]]` (format accelerate)

### `TransferManager` (`core/transfer_manager.py`)

Cherche `def transfer` (~ligne 483). Signature attendue : `transfer(source_gpu: int, target_gpu: int, tensor: torch.Tensor, stream: Optional[Any] = None) -> Tuple[TransportMethod, torch.Tensor]`.

Stratégies internes (0 → 4) :
- 0 : Cross-vendor bridge (AMD ↔ NVIDIA via DMA-BUF/ReBAR)
- 1 : Direct CUDA P2P
- 1.5 : Rust GpuPipeline (DtoD ≥ 512 KB)
- 1.7 : ReBAR pipelined (chunks 64 MB)
- 2 : NCCL send/recv
- 4 : CPU staging (fallback universel)

## Modèles HF disponibles localement

| Modèle | Taille | Status | Usage |
|--------|--------|--------|-------|
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 2 GB | full cache | Smoke tests bench (cuDNN crash en 2-GPU sur ce stack) |
| `Qwen/Qwen2.5-14B-Instruct` | 28 GB | full cache | Regression bench (LENDING_ON → 13.72 tok/s) |
| `Qwen/Qwen2.5-32B-Instruct` | 62 GB | full cache | OOM-débloque bench (V6.B prouve LENDING_ON OK à 0.39 tok/s) |
| `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8` | 33 GB | full cache | **Inutilisable** : Marlin kernel JIT compile fail (toolchain) |
| `deepseek-ai/DeepSeek-V4-Flash` | 149 GB | full cache | vLLM-only (P13) |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 0 | gated, no token | NE PAS essayer |
| `TheBloke/Llama-2-70B-GPTQ` | 28 KB | metadata only | NE PAS essayer (pas de poids) |

Pas de token HF configuré. Si tu as besoin d'un autre modèle, télécharge-le via :
```python
from huggingface_hub import snapshot_download
path = snapshot_download(repo_id='ORG/MODEL',
                         allow_patterns=['*.json', '*.safetensors', 'tokenizer*'])
```

## Format des résultats de bench

`benchmarks/results/bench_lending_hetero_real_<suffix>.json` schema :

```json
{
  "bench": "bench_lending_hetero_real",
  "phase": "...",
  "model": "Qwen/...",
  "num_gpus": 2,
  "max_new": 32,
  "warmup": 3,
  "topology": [{"index": 0, "name": "...", "sm": "...", "total_mb": ..., "free_mb": ...}, ...],
  "runs": [
    {
      "label": "LENDING_OFF" | "LENDING_ON",
      "lending_enabled_env": false | true,
      "vram_pre_load": {"gpu0": {"used_mb": ..., "total_mb": ...}, ...},
      "load_time_s": float,
      "pool_active": bool,
      "pool_registered_gpus": [{"gpu_id": int, "device_name": str, "lendable_bytes": int}, ...],
      "vram_post_load": {...},
      "tok_s": float,
      "total_tokens": int,
      "elapsed_s": float,
      "vram_post_bench": {...},
      "vram_delta_mb": {...},
      "pool_stats": {...},
      "status": "ok" | "OOM" | "error" | "timeout",
      "error": str (only if status != ok),
      "elapsed_subprocess_s": float,
      "returncode": int
    }
  ]
}
```

Si tu modifies le bench pour ajouter de nouveaux champs (par ex. `kv_lending_active` pour V6.D), maintiens la rétrocompatibilité.

---

## Ordre d'exécution recommandé

1. **V6.C** (data plane via TransferManager) — 2-4 h, indépendant, low risk
2. **V6.F** (dedupe resultat_v5.md) — 15 min, trivial
3. **V6.E** (silent excepts sweep) — 1-2 h par batch, infinitely deferrable
4. **V6.D** (KV overflow hook) — 4-6 h, plus risqué, à faire en dernier

Si tu n'as que 4 heures, fais V6.C + V6.F. V6.D nécessite une revue avant.

## Ce qui ne fait PAS partie de V6 (et que tu ne dois pas attaquer)

- Wirer la lending pool dans le path vLLM (`backend_type='vllm'`) — vLLM utilise un sous-process spawn avec son propre allocator, intégrer le pool nécessite des hooks vLLM internals que vLLM n'expose pas. **Tracé en V7+**.
- TURBO_KV_CUDAGRAPH (Phase 2 turbo_engine: StaticKVCache + CUDA Graph capture) — projet en soi.
- TURBO_KV_HMM_OFFLOAD (migrer `_DramDict` shim vers vrai `HierarchicalMemoryManager`) — projet en soi.
- VRM_TRANSFER_OVERLAP=1 mesures de gain — bench dédié séparé.
- Fix Marlin kernel JIT (`gptqmodel`) — toolchain CUDA, hors scope de la lending pool.

## Contact / blocage

Si tu es bloqué sur quelque chose qui n'est **pas** explicitement adressé dans ce document, **n'invente pas une solution**. Stop, écris ce que tu as compris du blocage dans un commentaire `# BLOCKED@V6.X: ...`, et demande à l'utilisateur (Jérémie). Mieux vaut une question qu'un fix qui casse un autre flow.

---

*Document écrit le 2026-05-08 par Opus 4.7 (1M context) après les commits V6.A `74dc904` et V6.B `73218f9`. Branch de référence : `feat/v6-lending-cooperative`. Bonne route.*
