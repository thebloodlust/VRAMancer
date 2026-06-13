# Plan V3 — Cleanup Honnêteté + Performance ReBAR + Onboarding

> **Pour l'agent exécutant :** ce plan est conçu pour être suivi à la lettre, étape par étape.  
> Lis chaque section dans l'ordre. Ne saute jamais une validation.  
> Après chaque tâche numérotée (V0.1, V1.1…), tu dois faire un commit ATOMIQUE.  
> Si une tâche échoue, ARRÊTE-TOI et demande à l'utilisateur — ne brute-force pas.

**Version :** v3.0  
**Date :** 5 mai 2026  
**Branche source :** `chore/sonnet-plan-v2` (HEAD `7286a62`)  
**Branche cible :** `chore/sonnet-plan-v3`  
**Auteur :** Architecte Claude Opus 4.7

---

## Vision

Trois objectifs simultanés :

1. **Honnêteté** — Mettre à jour le doc d'audit `.github/copilot-instructions.md` (5+ red flags du doc sont en réalité **déjà résolus**). Documenter explicitement les vrais stubs restants.
2. **Performance ReBAR** — L'utilisateur a modifié Proxmox pour exposer ReBAR sur les GPU passthrough. Les performances sont **proches du bare-metal** désormais. Re-benchmarker tout, activer Strategy 1.5 (P2P direct) qui était bloquée par IOMMU.
3. **Onboarding** — `pip install vramancer && vramancer serve <model>` doit fonctionner en 5 min sur une machine vierge.

---

## Règles globales (ABSOLUES)

1. **NE JAMAIS** modifier ces fichiers/dossiers :
   - `_deprecated/`
   - `tests/test_chaos_concurrency.py`
   - `csrc/paged_attention_kernel.cu`
   - `rust_core/src/` (sauf instruction explicite en P3)
   - `core/security/__init__.py`, `core/security/startup_checks.py`
   - `core/paged_attention.py`
2. **NE JAMAIS** renommer : `Synapse`, `synapses`, `_apply_neuroplasticity_score`, `HolographicKVManager`, `Connectome`.
3. **NE JAMAIS** push, merge, créer une PR, ni faire `git push --force`.
4. **NE JAMAIS** désactiver des tests existants pour faire passer la suite. Si un test casse → fix le code, pas le test.
5. **TOUJOURS** committer atomiquement (1 tâche = 1 commit), avec préfixe `[V<x>.<y>]`.
6. **TOUJOURS** valider la suite complète (V7.1) à la fin avant de signaler "terminé".
7. **TOUJOURS** copier-coller les résultats CHIFFRÉS dans `resultat_v3.md` (pas de "ça marche" — donne les nombres).

---

## Préparation (UNE FOIS, avant V0)

```bash
cd /home/jeremie/VRAMancer/VRAMancer
source .venv/bin/activate
git status   # doit être propre
git checkout chore/sonnet-plan-v2   # branche source
git checkout -b chore/sonnet-plan-v3
```

**Baseline tests à enregistrer dans `resultat_v3.md` au démarrage :**

```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 VRM_BACKEND_ALLOW_STUB=1 \
  pytest tests/ --ignore=tests/test_chaos_concurrency.py --tb=no --no-cov 2>&1 | tail -1
```

**Cible attendue baseline :** `1 failed, 1070 passed, 39 skipped` (1 failed pré-existant).

Crée `resultat_v3.md` avec section `[BASELINE]` au tout début.

---

# P0 — Mise à jour du doc d'audit

Le doc `.github/copilot-instructions.md` contient des red flags **déjà résolus** depuis longtemps. Il faut le mettre à jour pour ne plus mentir aux futurs agents.

## V0.1 — Vérifier l'état actuel des "red flags" du doc audit

**Commandes de vérification (à lancer une par une et noter le résultat) :**

```bash
# 1. software_cxl.cpp existe encore ?
ls csrc/ | grep -E "cxl|file_offload"
# Attendu: file_offload.cpp (renommé), pas software_cxl.cpp

# 2. batch_inference.py — où est-il ?
find . -name "batch_inference.py" -not -path "*/node_modules/*" 2>/dev/null
# Attendu: ./_deprecated/batch_inference.py uniquement

# 3. supervision_api NODES — hardcodé ou dynamique ?
grep -n "^NODES" core/network/supervision_api.py
# Attendu: NODES = []  (avec commentaire "Populated dynamically")

# 4. rust_core detect_best_transport — stub ou réel ?
sed -n '30,40p' rust_core/src/lib.rs
# Attendu: probe libibverbs.so.1 via libloading

# 5. aitp_receiver XDP — stub ?
sed -n '125,150p' core/network/aitp_receiver.py
# Attendu: getattr(socket, "AF_XDP", 44) avec fallback gracieux
```

Note les 5 résultats dans `resultat_v3.md` section `[V0.1]`.

## V0.2 — Mettre à jour `.github/copilot-instructions.md`

**Cible :** Section `### RED FLAGS (code qui ment)` (cherche "RED FLAGS" dans le fichier).

**Modifications EXACTES :**

1. **Red flag #1 (hierarchical_memory)** — déjà marqué CORRIGE. Laisser tel quel.
2. **Red flag #2 (transfer_manager Strategy 1.5)** — déjà CORRIGE. Laisser.
3. **Red flag #3 (block_router RemoteExecutor)** — V3.1 du plan V2 a corrigé la docstring. Mettre à jour :
   ```diff
   -3. **block_router.py RemoteExecutor** — label "zero-copy" = **FAUX** (safetensors serialise).
   +3. ~~**block_router.py RemoteExecutor**~~ **CORRIGE** (V2 plan, 2026-05) : docstring honnête (safetensors → TCP socket round-trip, pas zero-copy).
   ```
4. **Red flag #4 (software_cxl.cpp)** — déjà renommé. Mettre à jour :
   ```diff
   -4. **software_cxl.cpp** — nom "CXL" = **TROMPEUR** : c'est du file I/O simple (std::ofstream).
   +4. ~~**software_cxl.cpp**~~ **CORRIGE** : renommé `csrc/file_offload.cpp` avec header explicite "formerly software_cxl, plain file I/O".
   ```
5. **Red flag #5 (supervision_api NODES)** — déjà CORRIGE.
6. **Red flag #6 (batch_inference.py)** — déplacé dans `_deprecated/`. Mettre à jour :
   ```diff
   -6. **batch_inference.py** — `generate_batch_fn` **JAMAIS FOURNI** -> fallback toujours sequentiel.
   +6. ~~**batch_inference.py**~~ **DEPRECIE** : déplacé dans `_deprecated/batch_inference.py`. Use `core/continuous_batcher.py` à la place.
   ```
7. **Red flag #7 (backends_webgpu)** — TOUJOURS un red flag (POC marketing).
8. **Red flag #8 (aitp_receiver XDP)** — précision : le code utilise `getattr(socket, "AF_XDP", 44)` qui est valide en Linux moderne. Mettre à jour :
   ```diff
   -8. **aitp_receiver.py XDP** — `socket(44, SOCK_RAW, 0)` — famille 44 invalide, toujours False. Seul UDP marche.
   +8. **aitp_receiver.py XDP** — code défensif (`getattr(socket, "AF_XDP", 44)` + fallback gracieux). AF_XDP=44 est valide en Linux ≥4.18 mais nécessite CAP_NET_ADMIN. Sans permissions, retombe sur UDP. **Pas un red flag — defensive coding.**
   ```
9. **Red flag #9 (dashboard launcher)** — déjà CORRIGE.
10. **Red flag #10 (placement_engine neuroplasticity)** — précision :
    ```diff
    -10. **placement_engine.py** — `_apply_neuroplasticity_score()` = heuristique pseudo-scientifique non-deterministe.
    +10. **placement_engine.py** — `_apply_neuroplasticity_score()` utilise les poids synaptiques du Connectome (Hebbian learning sur latence/reliability mesurés). **Réel mais non-déterministe par design** — utilise les mesures live. Considérer de logger le poids appliqué pour traçabilité.
    ```

**Section "rust_core/" → ligne TransportTier :**

```diff
-| **TransportTier** | STUB | `detect_best_transport()` retourne toujours ZeroCopyTcp. Pas de detection RDMA reelle. |
+| **TransportTier** | REEL | `detect_best_transport()` probe `libibverbs.so.1` via `libloading::Library::new()`. Retourne `DirectRdma` si présent, sinon `ZeroCopyTcp`. |
```

**Validation :**

```bash
git diff .github/copilot-instructions.md | head -80
```

**Commit :**

```bash
git add .github/copilot-instructions.md
git commit -m "[V0.1+V0.2] update copilot-instructions: 5 red flags now resolved (audit truth-up)"
```

---

# P1 — Honnêteté code (red flags réels restants)

## V1.1 — Marquer `backends_webgpu.py` comme POC dans le module docstring

**Cible :** `core/backends_webgpu.py`, lignes 1-30 (docstring de module).

**Action :** Lis les 30 premières lignes, puis remplace le docstring de module pour qu'il dise EXPLICITEMENT que c'est un POC, pas production-ready, et que les claims "Speculative Decoding" et "Holographic Parity" sont du batching optimiste / parity XOR simple, pas des techniques avancées.

**Template du nouveau docstring :**

```python
"""WebGPU Backend — Experimental POC (NOT PRODUCTION-READY).

⚠️  Status: Proof-of-concept / template only.
    - WebSocket worker pool defined but `nodes` registry never populated automatically.
    - Class names referencing "Speculative Decoding" implement simple optimistic
      batching (not the formal speculative-decoding algorithm).
    - Class names referencing "Holographic Parity" implement plain XOR parity
      (not holographic memory).
    - Use only via `python -m dashboard.worker` for browser-based experiments.

For production WebGPU compute, see `dashboard/worker/` (matmul.wgsl + worker.js).
"""
```

**Si le fichier a déjà un docstring de module, le remplacer en gardant 3 lignes de contexte (imports etc) en dessous. Si pas de docstring, l'ajouter ligne 1.**

**Validation :**

```bash
python -c "import core.backends_webgpu; print(core.backends_webgpu.__doc__[:100])"
# doit afficher "WebGPU Backend — Experimental POC..."
```

**Commit :**

```bash
git add core/backends_webgpu.py
git commit -m "[V1.1] mark backends_webgpu.py as experimental POC in module docstring"
```

## V1.2 — Documenter `csrc/vtp_core.cpp` L3+ stubs explicitement

**Cible :** `csrc/vtp_core.cpp` ligne 6 (header existant).

**Le fichier dit déjà** :
```
//   STUB:  L3+ (RDMA, RAM, NVMe, Network) — returns src.clone(), no actual transport.
```

**Action :** Vérifier que c'est honnête dans le header ET ajouter un `TODO(VTP_L3)` au-dessus de la ligne `if (target_tier == L3_VRAM_REMOTE_RDMA)` pour qu'un futur dev puisse retrouver la dette technique.

```bash
grep -n "L3_VRAM_REMOTE_RDMA" csrc/vtp_core.cpp
```

Insérer juste avant la ligne `if (target_tier == L3_VRAM_REMOTE_RDMA)` (lignes ~52-54) :

```cpp
    // TODO(VTP_L3): Implement actual RDMA transport via libibverbs.
    // Current behavior: returns src.clone() — no remote transfer.
    // Track in: docs/reports/TECHNICAL_DEBT.md#vtp-l3-l7
```

**Commit :**

```bash
git add csrc/vtp_core.cpp
git commit -m "[V1.2] document VTP L3+ stub with TODO marker for technical debt tracking"
```

## V1.3 — Créer `docs/reports/TECHNICAL_DEBT.md`

**Action :** Créer un fichier listant TOUS les stubs/dette technique connus, pour que les futurs devs sachent quoi attaquer.

**Contenu attendu (template — adapte avec les vraies valeurs vérifiées) :**

```markdown
# Technical Debt — VRAMancer

> Dernière mise à jour : 2026-05  
> Maintenu manuellement à chaque PR qui ajoute ou résout un stub.

## Stubs réels (code en place mais incomplet)

| ID | Fichier | Ligne(s) | Description | Effort | Priorité |
|----|---------|----------|-------------|--------|----------|
| VTP_L3 | csrc/vtp_core.cpp | 52-62 | Router L3+ retourne `src.clone()` au lieu d'un vrai transport RDMA/Network | Moyen | Basse — VTP routing ne sert qu'en cluster multi-node, et le path AITP/RDMA via `core/network/llm_transport.py` est déjà fonctionnel séparément |
| DMABUF_WRITE | csrc/dmabuf_bridge.c | header | `vrm_dmabuf_transfer` step 4-5 : dst mmap write pas implémenté ; le caller Python doit faire la copie finale via torch pinned memory | Gros | Basse — les stratégies CUDA P2P (RTX→RTX) couvrent 99% des cas. DMA-BUF sert pour cross-vendor (NVIDIA↔AMD) qui est rare |
| NAT_HOLE_PUNCH | core/network/nat_traversal.py | hole_punch + relay | UDP hole punching et TURN relay sont des stubs ; STUN RFC 5389 est réel | Moyen | Basse — mode LAN/intranet n'en a pas besoin |
| WEBGPU_BACKEND | core/backends_webgpu.py | global | POC : nodes jamais peuplées automatiquement, "Speculative Decoding" = batching optimiste, "Holographic Parity" = XOR simple | Gros | Très basse — usage browser-only via dashboard/worker |
| BATCH_INFERENCE | _deprecated/batch_inference.py | — | Déprécié. `generate_batch_fn` n'a jamais été câblé. Remplacé par `core/continuous_batcher.py` | — | Aucune — déprécié |

## Limitations connues (non-bugs, by design)

| Limitation | Contournement |
|------------|---------------|
| VM Proxmox sans ReBAR : Strategy 1.5 (P2P direct) bloquée par IOMMU | Activer ReBAR shared dans Proxmox (cf `docs/reports/REBAR_PROXMOX_SETUP.md`) |
| BnB quantization multi-GPU = upstream bug accelerate 1.13.0 | VRAMancer force single-GPU si BnB |
| Continuous batcher backpressure max_waiting_queue=256 | Configurable via `VRM_BATCHER_QUEUE_MAX` (à ajouter si besoin) |
| Triton sampling fallback PyTorch toujours utilisé en pratique | Optimisation future — fuser top-k dans le kernel |
| `_apply_neuroplasticity_score()` non-déterministe | By design — utilise les poids synaptiques live du Connectome (Hebbian) |

## Stubs résolus depuis l'audit 2026-03 (pour traçabilité)

- ✅ `software_cxl.cpp` → `csrc/file_offload.cpp` (nom honnête)
- ✅ `supervision_api.NODES` → désormais dynamique (peuplé par heartbeat)
- ✅ `transfer_manager` Strategy 1.5 → `direct_vram_copy()` réel via PyO3
- ✅ `vram_lending` → testé réel RTX 3090 + 5070 Ti
- ✅ `hierarchical_memory` eviction/spill → réel (Rust cxl_direct_memory_dump)
- ✅ `block_router.RemoteExecutor` docstring → corrigée (n'est PAS zero-copy)
- ✅ `dashboard/launcher.launch_cli_dashboard()` → existe via alias
- ✅ `routes_ops.py` GPU detection → multi-vendor (CUDA/ROCm/MPS/XPU)
- ✅ `backends_ollama.py` aiohttp leak → supprimé, sync requests uniquement
- ✅ `backends_vllm.py` OOM retry no-op → fix (halve max_tokens uniquement)
- ✅ `stream_manager` async executor jamais join() → shutdown(wait=True)
- ✅ `batch_inference.py` fallback séquentiel → déplacé dans `_deprecated/`
- ✅ `rust_core.detect_best_transport()` → probe libibverbs réel (pas un stub)
```

**Commit :**

```bash
git add docs/reports/TECHNICAL_DEBT.md
git commit -m "[V1.3] add TECHNICAL_DEBT.md — single source of truth for known stubs and limitations"
```

## V1.4 — Documenter le stub UDP hole punch dans `nat_traversal.py`

**Cible :** `core/network/nat_traversal.py` (~250 LOC).

**Action :** Lis les 30 premières lignes (docstring de module) et ajoute un avertissement clair en tête du docstring si pas déjà présent :

```python
"""NAT Traversal — STUN client (RFC 5389) + UDP hole punch / TURN relay STUBS.

⚠️  Implementation status:
    - STUN RFC 5389 client: REAL (uses udp socket binding response).
    - UDP hole punching: STUB (returns success but no actual punch logic).
    - TURN relay: STUB (no fallback when hole punch fails).

This module is suitable for LAN / VPN / intranet topologies where NAT
traversal is not required. For internet-scale peer-to-peer scenarios,
consider using a dedicated TURN server (coturn) and treating this module
as a discovery layer only.
"""
```

**Si le fichier a déjà ce docstring, ne fais rien (idempotent).**

**Commit :**

```bash
git add core/network/nat_traversal.py
git commit -m "[V1.4] document NAT traversal stubs (hole punch + TURN relay) in module docstring"
```

---

# P2 — Performance benchmarks ReBAR (Proxmox modifié)

> **Contexte utilisateur :** "j'ai modifié Proxmox pour qu'il gère les cartes en partage ReBAR.  
> Les performances sont désormais plus proches du hardware pur."  
> ⚠️ **Cette section nécessite l'accès aux GPU réels (RTX 3090 + RTX 5070 Ti).**  
> Si l'agent n'a pas d'accès GPU, **arrête-toi à V2.1** et demande à l'utilisateur de lancer V2.2-V2.5 lui-même.

## V2.1 — Vérifier que ReBAR est actif via nvidia-smi

**Commandes :**

```bash
# Méthode 1 : ReBAR officiel via nvidia-smi
nvidia-smi -q | grep -A 3 "BAR1 Memory Usage"
# Attendu (si ReBAR actif) : BAR1 Total >= VRAM Total
#   Ex RTX 3090 : BAR1 Total = 24576 MiB (= 24 GB VRAM totale)
#   Sans ReBAR :  BAR1 Total = 256 MiB (legacy default)

# Méthode 2 : via lspci (BAR size resizable)
lspci -vvv -s $(lspci | grep -i nvidia | head -1 | awk '{print $1}') 2>&1 | grep -E "Region|Resizable BAR"

# Méthode 3 : via core/health.py
python -c "from core.health import gpu_health_check; import json; print(json.dumps(gpu_health_check(), indent=2))" 2>&1 | head -30
```

**Note les résultats EXACTS dans `resultat_v3.md` section `[V2.1]`. Inclure :**
- Sortie complète de `nvidia-smi -q | grep -A 3 BAR1`
- Pour CHAQUE GPU : `BAR1 Total = X MiB`, `VRAM Total = Y MiB`
- Verdict : `ReBAR ACTIF` ou `ReBAR INACTIF`

**Si ReBAR INACTIF :**
- Note-le et **passe directement à P3** (skip P2.2 - P2.5).
- Ajoute dans `resultat_v3.md` : "P2.2-P2.5 SKIP — ReBAR pas détecté."

**Commit (sans modification de code) — utiliser empty commit :**

```bash
git commit --allow-empty -m "[V2.1] ReBAR detection report (see resultat_v3.md)"
```

## V2.2 — Re-bench Qwen2.5-14B BF16 2-GPU

> **Pré-requis :** V2.1 confirme ReBAR ACTIF.

**Commande :**

```bash
python benchmarks/bench_14b_bigpu.py 2>&1 | tee bench_14b_rebar.log
```

**(Si le script n'existe pas, utiliser ce one-liner) :**

```bash
python -c "
import os, time
os.environ['VRM_QUANTIZATION'] = ''  # BF16
from core.inference_pipeline import get_pipeline
pipe = get_pipeline()
pipe.load('Qwen/Qwen2.5-14B', num_gpus=2)
prompt = 'Explain quantum entanglement in one paragraph.'
# Warmup
pipe.generate(prompt, max_new_tokens=10)
# Bench
t0 = time.perf_counter()
out = pipe.generate(prompt, max_new_tokens=200)
elapsed = time.perf_counter() - t0
toks = 200
print(f'tok/s = {toks/elapsed:.2f}')
print(f'elapsed = {elapsed:.2f}s')
" 2>&1 | tee bench_14b_rebar.log
```

**Comparer au baseline pré-ReBAR :**
- Baseline 2026-04 (sans ReBAR Proxmox) : **6.0 tok/s**
- Cible avec ReBAR : **≥ 9 tok/s** (espéré : +50% à +100% selon le bottleneck)

**Inscrire dans `resultat_v3.md` :**

```markdown
### [V2.2] Qwen 14B BF16 2-GPU avec ReBAR

| Métrique | Pré-ReBAR | Post-ReBAR | Delta |
|----------|-----------|-----------|-------|
| tok/s | 6.0 | XXX | +YYY% |
| TTFT | XXX | XXX | XXX |
| GPU0 VRAM | 21.7 GiB | XXX GiB | XXX |
| GPU1 VRAM | 14.2 GiB | XXX GiB | XXX |
```

## V2.3 — Bench Strategy 1.5 (P2P direct) maintenant disponible

**Action :** Tester que `transfer_manager.direct_vram_copy()` fonctionne en P2P direct (avant : bloqué IOMMU, fallback Strategy 4 CPU-staged).

```bash
python -c "
from core.transfer_manager import TransferManager
import torch
tm = TransferManager()
# Test P2P direct
src = torch.randn(1024, 1024, dtype=torch.float16, device='cuda:0')
dst = torch.empty_like(src, device='cuda:1')
import time
t0 = time.perf_counter()
for _ in range(100):
    tm.direct_vram_copy(src, dst)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
size_gb = src.numel() * 2 / 1e9 * 100
print(f'Strategy 1.5 P2P bandwidth: {size_gb/elapsed:.2f} GB/s')
" 2>&1 | tee bench_p2p_rebar.log
```

**Cible (RTX 3090 ↔ RTX 5070 Ti via PCIe 4.0 x16) :**
- Sans ReBAR (Strategy 4 CPU-staged) : ~12-15 GB/s
- Avec ReBAR (Strategy 1.5 P2P) : **≥ 20 GB/s** attendu

**Inscrire résultat dans `resultat_v3.md` section `[V2.3]`.**

## V2.4 — Bench transfer end-to-end (small + large tensors)

```bash
python -c "
import torch, time
from core.transfer_manager import TransferManager
tm = TransferManager()
sizes_mb = [1, 4, 16, 64, 256, 1024]
for mb in sizes_mb:
    n = mb * 1024 * 1024 // 2  # fp16
    src = torch.randn(n, dtype=torch.float16, device='cuda:0')
    dst = torch.empty_like(src, device='cuda:1')
    # Warmup
    tm.direct_vram_copy(src, dst); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        tm.direct_vram_copy(src, dst)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 20
    print(f'{mb:>5} MB | {dt*1000:>7.2f} ms | {mb/dt/1024:.2f} GB/s')
" 2>&1 | tee bench_p2p_sweep.log
```

**Inscrire la table complète dans `resultat_v3.md` section `[V2.4]`.**

## V2.5 — Documenter dans `docs/reports/REBAR_PROXMOX_BENCHMARK.md`

**Action :** Créer le fichier avec :
- Description de la modif Proxmox effectuée par l'utilisateur (à demander, ou laisser le placeholder)
- Tableau comparatif pré-ReBAR vs post-ReBAR (V2.2)
- Bandwidth sweep (V2.4)
- Recommandation : "Activer ReBAR sur tous les hôtes Proxmox VRAMancer"

**Template :**

```markdown
# ReBAR Proxmox Benchmark

> Date : 2026-05-XX  
> Hardware : RTX 3090 + RTX 5070 Ti (passthrough Proxmox)

## Configuration Proxmox

[À compléter par l'utilisateur — décrire les modifs grub/kernel/QEMU]

Ex (à valider) :
- `/etc/default/grub` : `pci=realloc=on`, `pcie_acs_override=downstream`
- `/etc/modprobe.d/vfio.conf` : options vfio_pci ids=…
- `qm set <vmid> -hostpci0 …,pcie=1,x-vga=1,resizable_bar=1`

## Verification

```
nvidia-smi -q | grep -A 3 "BAR1 Memory Usage"
[paste output]
```

## Benchmarks

### Qwen 14B BF16 2-GPU

| Métrique | Pré-ReBAR | Post-ReBAR | Δ |
|----------|-----------|-----------|---|

### P2P bandwidth sweep

| Taille | Latence | BW |
|--------|---------|----|

## Conclusion

- Strategy 1.5 (P2P direct) débloquée
- Speedup observé : XXX%
- Recommandation : activer ReBAR sur toute la flotte
```

**Commit :**

```bash
git add docs/reports/REBAR_PROXMOX_BENCHMARK.md bench_*.log 2>/dev/null
git commit -m "[V2.2-V2.5] ReBAR Proxmox benchmarks: P2P direct now functional, +XXX% speedup"
```

---

# P3 — Optimisations performance

## V3.1 — Améliorer le prefetch transfer overlap N+1

**Cible :** `core/stream_manager.py` (~544 LOC).

**Contexte :** Pendant que le compute exécute la couche N, le transfer du output de N→N+1 doit être lancé en async pour que la couche N+1 ait son input prêt sans wait. Vérifier que c'est bien le cas.

**Audit :**

```bash
grep -n "prefetch\|stream\|cuda.Stream\|wait_event\|record_event" core/stream_manager.py | head -20
grep -n "torch.cuda.Stream\|cuda.stream" core/inference_pipeline.py | head -10
```

**Si le prefetch existe mais n'est PAS overlappé** (i.e. transfer attend la fin du compute avant de démarrer), créer une issue :

Créer `docs/reports/PREFETCH_OVERLAP_AUDIT.md` :

```markdown
# Prefetch Overlap Audit

## Status actuel

[Décris ici ce que fait le code actuellement, ligne par ligne avec citations]

## Bottleneck identifié ?

[Yes/No + détails]

## Recommendation

[Si Yes : patch proposé. Si No : note "déjà optimal".]
```

**Si le prefetch est OK :** commit empty avec note "[V3.1] prefetch overlap audit — already optimal" et passe à V3.2.

**Si le prefetch est SUBOPTIMAL :** ne PAS modifier le code dans cette session — c'est une optimisation risquée. Documente dans `TECHNICAL_DEBT.md` et passe à V3.2.

**Commit :**

```bash
git add docs/reports/PREFETCH_OVERLAP_AUDIT.md
git commit -m "[V3.1] prefetch overlap audit (analysis only, no code changes)"
```

## V3.2 — CUDA Graph multi-GPU — audit faisabilité

**Cible :** `core/cuda_graph_decode.py` (~250 LOC).

**Contexte :** Actuellement CUDA Graph fonctionne single-GPU only. Étendre à multi-GPU est complexe (capture sur plusieurs streams, sync events).

**Action :** Faire un audit (lecture seule) et documenter la faisabilité dans `docs/reports/CUDA_GRAPH_MULTI_GPU_AUDIT.md`.

**Questions à répondre dans le doc :**
1. Le code actuel capture sur 1 stream / 1 device — confirmé ?
2. Est-ce qu'un capture multi-device est techniquement possible avec PyTorch 2.5+ ?
3. Quel serait le risque de fragilité (dynamic shapes, synchronization across devices) ?
4. Estimation effort : XS / S / M / L / XL ?
5. Estimation gain perf : XS / S / M / L ?

**Verdict attendu :** "L (gros effort) pour M (gain moyen) — basse priorité tant que le pipeline parallel est < 70% utilisation GPU."

**Commit :**

```bash
git add docs/reports/CUDA_GRAPH_MULTI_GPU_AUDIT.md
git commit -m "[V3.2] CUDA Graph multi-GPU feasibility audit (analysis only)"
```

## V3.3 — Profile sync points dans pipeline parallel

**Cible :** `core/inference_pipeline.py` méthode `infer()` ou équivalent.

**Action :** Ajouter (en option, derrière flag `VRM_PROFILE_SYNC=1`) un logger de sync points pour identifier les `torch.cuda.synchronize()` ou `.cpu()` qui bloquent le pipeline.

**Implémentation minimale :**

Cherche les sync points existants :

```bash
grep -nE "synchronize|\.cpu\(\)|\.item\(\)|\.tolist" core/inference_pipeline.py | head -20
```

Si > 5 sync points trouvés, créer `docs/reports/SYNC_POINTS_AUDIT.md` listant chaque sync avec sa ligne et sa justification (légitime / pas légitime).

**Commit :**

```bash
git add docs/reports/SYNC_POINTS_AUDIT.md
git commit -m "[V3.3] sync points audit in inference_pipeline (analysis only)"
```

---

# P4 — Onboarding UX

## V4.1 — Vérifier la CLI `vramancer` actuelle

**Commande :**

```bash
which vramancer && vramancer --help 2>&1 | head -30
ls vramancer/cli/ 2>&1
```

**Inscrire le résultat dans `resultat_v3.md` section `[V4.1]`.**

## V4.2 — Ajouter (ou vérifier) la commande `vramancer serve <model>`

**Cible :** `vramancer/cli/` (créer `serve.py` si absent).

**Action :**

1. Vérifier si la commande existe :
   ```bash
   grep -rn "def serve\|'serve'" vramancer/ 2>&1 | head
   ```

2. **Si elle existe** : tester `vramancer serve gpt2 --port 8000 &` (background), puis `curl http://localhost:8000/v1/models` (5 sec timeout). Note résultat dans `[V4.2]`.

3. **Si elle n'existe pas** : créer `vramancer/cli/serve.py` minimal :

```python
"""vramancer serve <model> — quickstart command."""
import argparse
import os


def main():
    parser = argparse.ArgumentParser(prog="vramancer serve")
    parser.add_argument("model", help="HuggingFace model id or local path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="0 = auto-detect")
    parser.add_argument("--quantization", default="",
                        choices=["", "nf4", "int8", "nvfp4"])
    args = parser.parse_args()

    if args.quantization:
        os.environ["VRM_QUANTIZATION"] = args.quantization

    from core.production_api import create_app
    from core.inference_pipeline import get_pipeline
    pipe = get_pipeline()
    pipe.load(args.model, num_gpus=args.num_gpus or None)
    app = create_app(pipeline=pipe)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
```

Et la cabler dans `vramancer/main.py` ou `vramancer/__main__.py` (selon ce qui existe — utilise `grep -n "def main\|argparse\|sub_parsers" vramancer/main.py vramancer/__main__.py`).

**Validation :**

```bash
vramancer serve --help 2>&1 | head -10
```

**Commit :**

```bash
git add vramancer/cli/serve.py vramancer/main.py
git commit -m "[V4.2] add 'vramancer serve <model>' quickstart CLI command"
```

## V4.3 — Créer `docs/QUICKSTART.md`

**Contenu cible :**

```markdown
# VRAMancer Quickstart — 5 minutes from zero to inference

## Prerequisites

- Python 3.10+ (3.12 recommended)
- NVIDIA GPU with CUDA 12.1+ (or AMD ROCm 6+, or Apple MPS)
- 16 GB RAM minimum

## Install

```bash
pip install vramancer
# Or from source:
git clone https://github.com/thebloodlust/VRAMancer
cd VRAMancer && pip install -e .
```

## Verify GPU detection

```bash
vramancer health
# Should list your GPU(s) with VRAM and compute capability.
```

## Run inference (3 commands)

### 1. Single-GPU, BF16

```bash
vramancer serve gpt2 --port 8000
```

### 2. Multi-GPU with auto-split

```bash
vramancer serve Qwen/Qwen2.5-7B --num-gpus 2 --port 8000
```

### 3. Quantized (NF4 — runs on 8 GB GPU)

```bash
vramancer serve mistralai/Mistral-7B-v0.1 --quantization nf4 --port 8000
```

## Test the API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt2","messages":[{"role":"user","content":"Hello"}]}'
```

OpenAI-compatible — all standard SDKs work.

## Next steps

- **Multi-node cluster** : `docs/CLUSTER_SETUP.md`
- **Production deployment** : `docs/PRODUCTION.md`
- **Backend matrix** : `docs/COMPATIBILITY.md`
- **Benchmark vs vLLM/TGI** : `docs/reports/BENCHMARKS.md`
```

**Commit :**

```bash
git add docs/QUICKSTART.md
git commit -m "[V4.3] add QUICKSTART.md — 5-minute onboarding guide"
```

---

# P5 — Benchmarks comparatifs (optionnel — nécessite GPU)

> ⚠️ Si l'agent n'a pas d'accès GPU réel, **skip P5 entièrement** et note "P5 SKIP — needs GPU access" dans `resultat_v3.md`.

## V5.1 — Bench vs vLLM (même modèle, mêmes GPU)

```bash
# 1. VRAMancer
vramancer serve Qwen/Qwen2.5-7B --num-gpus 1 --port 8000 &
sleep 30  # warmup
python benchmarks/bench_openai_api.py --url http://localhost:8000 --model Qwen/Qwen2.5-7B --requests 50 2>&1 | tee bench_vrm_qwen7b.log
kill %1

# 2. vLLM
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B --port 8001 &
sleep 60  # vLLM warmup is slow
python benchmarks/bench_openai_api.py --url http://localhost:8001 --model Qwen/Qwen2.5-7B --requests 50 2>&1 | tee bench_vllm_qwen7b.log
kill %1
```

**Si `benchmarks/bench_openai_api.py` n'existe pas, créer un script minimal qui envoie N requêtes parallèles et mesure :**
- TTFT (time to first token)
- ITL (inter-token latency)  
- Throughput (tok/s aggregated)
- p50, p95, p99 latency

## V5.2-V5.4 — Bench llama.cpp + TGI + tableau

[Idem V5.1 avec adapté]

**Commit :**

```bash
git add benchmarks/bench_openai_api.py bench_*_qwen7b.log docs/reports/BENCHMARKS.md
git commit -m "[V5.1-V5.4] comparative benchmarks vs vLLM, llama.cpp, TGI"
```

---

# P6 — Stress-test multi-user

> ⚠️ Skip si pas d'accès GPU.

## V6.1 — Continuous batcher concurrent

```bash
VRM_CONTINUOUS_BATCHING=1 vramancer serve Qwen/Qwen2.5-7B --port 8000 &
sleep 30
python -c "
import asyncio, aiohttp, time
async def req(session, i):
    async with session.post('http://localhost:8000/v1/chat/completions', json={
        'model': 'Qwen/Qwen2.5-7B',
        'messages': [{'role':'user','content': f'Tell me a joke {i}'}],
        'max_tokens': 50,
    }) as r:
        return await r.json()
async def main(n_users):
    async with aiohttp.ClientSession() as s:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[req(s, i) for i in range(n_users)])
        dt = time.perf_counter() - t0
        print(f'{n_users} users : {dt:.2f}s total, {n_users/dt:.2f} req/s')
for n in [1, 10, 50, 100]:
    asyncio.run(main(n))
" 2>&1 | tee stress_concurrent.log
kill %1
```

**Inscrire dans `resultat_v3.md` section `[V6.1]`.**

**Commit :**

```bash
git add stress_concurrent.log
git commit -m "[V6.1] continuous batcher stress test (1, 10, 50, 100 concurrent users)"
```

---

# P7 — Validation finale

## V7.1 — Suite complète

```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 VRM_BACKEND_ALLOW_STUB=1 \
  pytest tests/ --ignore=tests/test_chaos_concurrency.py --tb=no --no-cov 2>&1 | tail -3
```

**Cible :**
- ≥ 1070 passed (baseline V2)
- 1 failed (pré-existant, MÊME failure)
- 39 skipped

**Si une régression apparaît : ARRÊTE-TOI, debug, fix, re-run. NE COMMIT JAMAIS sur une régression.**

**Inscrire le résultat exact dans `resultat_v3.md` section `[V7.1]`.**

## V7.2 — Vérifier git log

```bash
git log --oneline chore/sonnet-plan-v2..HEAD
```

**Cible :** entre 10 et 20 commits, tous préfixés `[V<x>.<y>]`.

**Inscrire le log complet dans `resultat_v3.md` section `[V7.2]`.**

## V7.3 — Finaliser `resultat_v3.md`

Section `[SUMMARY]` à la fin :

```markdown
## [SUMMARY]

| Phase | Status | Notes |
|-------|--------|-------|
| P0 — Audit truth-up | ✅ ou ❌ | X red flags resolved/updated |
| P1 — Honnêteté code | ✅ ou ❌ | TECHNICAL_DEBT.md created |
| P2 — ReBAR benchmarks | ✅ / ⚠️ skip | If skip: GPU not available |
| P3 — Performance audits | ✅ ou ❌ | 3 audit docs created |
| P4 — Onboarding UX | ✅ ou ❌ | QUICKSTART.md + serve cmd |
| P5 — Comparatifs | ✅ / ⚠️ skip | |
| P6 — Stress test | ✅ / ⚠️ skip | |
| P7 — Validation | ✅ ou ❌ | XXXX passed |

### Commits ajoutés sur `chore/sonnet-plan-v3` :

[paste git log]

### Fichiers créés / modifiés :

[liste]

### Régressions :

[Aucune] ou [liste avec analyse]

### Suggestions pour V4 :

[3-5 next priorités, basées sur ce qui a été découvert]
```

**Commit final :**

```bash
git add resultat_v3.md
git commit -m "[V7.3] resultat_v3.md — execution log V3"
```

---

## Annexe A — Mapping rapide tâche → commit

| Tâche | Type | Commit prefix | Fichiers attendus |
|-------|------|---------------|-------------------|
| V0.1 | analyse | (combiné V0.2) | resultat_v3.md |
| V0.2 | edit | `[V0.1+V0.2]` | .github/copilot-instructions.md |
| V1.1 | edit | `[V1.1]` | core/backends_webgpu.py |
| V1.2 | edit | `[V1.2]` | csrc/vtp_core.cpp |
| V1.3 | create | `[V1.3]` | docs/reports/TECHNICAL_DEBT.md |
| V1.4 | edit | `[V1.4]` | core/network/nat_traversal.py |
| V2.1 | analyse | `[V2.1]` empty | resultat_v3.md |
| V2.2-V2.5 | bench | `[V2.2-V2.5]` | bench_*.log + REBAR_PROXMOX_BENCHMARK.md |
| V3.1 | analyse | `[V3.1]` | docs/reports/PREFETCH_OVERLAP_AUDIT.md |
| V3.2 | analyse | `[V3.2]` | docs/reports/CUDA_GRAPH_MULTI_GPU_AUDIT.md |
| V3.3 | analyse | `[V3.3]` | docs/reports/SYNC_POINTS_AUDIT.md |
| V4.1 | analyse | (combiné V4.2) | resultat_v3.md |
| V4.2 | create+edit | `[V4.2]` | vramancer/cli/serve.py + main.py |
| V4.3 | create | `[V4.3]` | docs/QUICKSTART.md |
| V5.1-V5.4 | bench | `[V5.1-V5.4]` | bench_*.log + BENCHMARKS.md |
| V6.1 | bench | `[V6.1]` | stress_concurrent.log |
| V7.1 | validation | (pas de commit) | resultat_v3.md |
| V7.2 | validation | (pas de commit) | resultat_v3.md |
| V7.3 | edit | `[V7.3]` | resultat_v3.md |

---

## Annexe B — Variables d'environnement utilisées

| Var | Default | Usage |
|-----|---------|-------|
| `VRM_MINIMAL_TEST=1` | — | tests CI sans torch |
| `VRM_DISABLE_RATE_LIMIT=1` | — | tests CI |
| `VRM_TEST_MODE=1` | — | tests CI |
| `VRM_BACKEND_ALLOW_STUB=1` | — | tests CI |
| `VRM_QUANTIZATION` | "" | nf4/int8/nvfp4 |
| `VRM_PARALLEL_MODE` | pp | pp/tp |
| `VRM_PROFILE_SYNC` | — | (V3.3) profile sync points |

---

## Annexe C — Critères de succès du Plan V3

Le plan V3 est **terminé avec succès** si :

1. ✅ Toutes les tâches P0, P1 sont commitées (V0.1 → V1.4) — **obligatoire**.
2. ✅ Toutes les tâches P3 (audits) sont commitées (V3.1 → V3.3) — **obligatoire**.
3. ✅ V4.2 et V4.3 sont commitées (onboarding) — **obligatoire**.
4. ⚠️ P2, P5, P6 sont commitées **OU** marquées `SKIP — GPU access required` dans `resultat_v3.md`.
5. ✅ V7.1 montre **≥ 1070 passed** sans nouvelle régression.
6. ✅ `resultat_v3.md` est complet, contient les chiffres réels, et est commité (V7.3).

**Si l'un de ces critères n'est PAS rempli, le plan est INCOMPLET.** Notifie l'utilisateur avec les critères manquants — ne déclare PAS "succès".
