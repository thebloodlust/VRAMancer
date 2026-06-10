# Propositions de chantiers futurs pour VRAMancer

Suite à l'audit complet du projet et à la mise à jour de `.github/copilot-instructions.md`, voici les propositions prioritaires pour assainir et améliorer l'architecture de VRAMancer, avec **les détails d'implémentation techniques**.

## Ordre de priorité

| Priorité | Chantier | Effort | Impact | Justification |
|----------|----------|--------|--------|---------------|
| **P0** | Quick Wins (nettoyage) | 1-2 jours | Élevé | Réduit le bruit, stabilise la CI, supprime le dead code |
| **P1** | Continuous Batcher | 2-3 jours | Élevé | Pure Python, meilleur ratio effort/impact, débloque le throughput multi-requêtes |
| **P2** | TransferManager Rust | 3-5 jours | Moyen | Dépend de Rust + CUDA FFI, risque plus élevé, mais nécessaire pour le multi-GPU performant |
| **P3** | Hierarchical Memory | 3-5 jours | Faible | 40 GB VRAM total (3090+5070Ti) suffisent pour la plupart des modèles ≤14B. Utile uniquement pour les modèles >30B ou les contextes très longs |

---

## Chantier 0 : Quick Wins — Nettoyage et stabilisation CI

**Problème actuel** : ~15 tests en échec pré-existants, du dead code jamais appelé, des modules Grade D qui polluent le codebase et trompent les développeurs.

**Actions concrètes :**
1. **Supprimer les fichiers junk à la racine** : `)`, `1`, `s`, `or)`, `et_device(0)`, `ion.cuda)`, `ynchronize()`, `tatus --short` — artefacts de terminaux mal fermés.
2. **Fixer les ~15 tests en échec** : principalement des tests Flask (`test_api_consolidation`, `test_integration_flask`, `test_multiprocess_flask`) et des edge cases de sérialisation. L'objectif est un **green CI** stable.
3. **Marquer explicitement les stubs** : ajouter un warning log au démarrage pour `hierarchical_memory.py`, `backends_webgpu.py`, `swarm_ledger.py`, `telemetry.py`, `batch_inference.py` — que personne ne les confonde avec du code fonctionnel.
4. **Supprimer le dead code dangereux** : `dashboard/launcher.py` importe `launch_cli_dashboard()` qui n'existe pas. `supervision_api.py` a des nodes hardcodés fictifs.

**Métriques de succès :**
- 0 tests en échec en CI (hors `test_pipeline_concurrent_load` deselect)
- 0 fichier junk à la racine
- Chaque module Grade D a un `logger.warning("STUB: ...")` au premier appel

**Tests de validation :**
```bash
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 pytest -q tests/ --deselect tests/test_chaos_concurrency.py::test_pipeline_concurrent_load
# Résultat attendu : 0 failures, 0 errors
```

---

## Chantier 1 : Débloquer le Continuous Batcher — PRIORITÉ HAUTE

**Problème actuel** : `core/continuous_batcher.py` promet du "Continuous Batching" façon vLLM. En réalité, le Lock Python (GIL) est maintenu pendant les lourdes opérations d'E/S du tokenizer, ce qui transforme un système supposément parallèle en un goulot d'étranglement complètement séquentiel.

**Pourquoi c'est prioritaire** : Pure Python, zero dépendance externe (pas de Rust/CUDA), impact direct sur le throughput multi-requêtes en production. Meilleur ratio effort/impact de tous les chantiers.

**Comment s'y prendre :**
1. **Dégripper le Compute et l'E/S (GIL Release)** :
   - Encapsuler systématiquement les appels `AutoTokenizer.encode()` et `decode()` via `asyncio.get_running_loop().run_in_executor(None, ...)`, basculant ainsi la charge de décodage CPU dans un `ThreadPoolExecutor` ou `ProcessPoolExecutor`.
   - Circonscrire strictement les blocs de sécurité de concurrence (`asyncio.Lock()`/`threading.Lock()`). Le mutex ne doit protéger que la structure de données de la Queue (l'ajout/retrait d'une requête) et **jamais** bloquer l'API pendant qu'un texte se parse.
2. **Instaurer une vraie Backpressure (API Control)** :
   - À l'heure actuelle, un trop-plein de requêtes fait s'écrouler le serveur (OOM).
   - Mettre en place un buffer fini (`asyncio.Queue(maxsize=MAX_QUEUE)`). Tout nouveau prompt rejette immédiatement la connexion entrante au niveau du endpoint web (`routes_ops.py`) avec un code HTTP `429 Too Many Requests`.
3. **Mise à l'échelle dynamique de la VRAM (Dynamic Sizing)** :
   - Brancher l'algorithme directement à notre instance de *PagedAttention* (qui est 100% fonctionnelle en CUDA). À chaque tick du moteur asynchrone, scruter l'API `gpu_interface` (mémoire *libérée* par des prompts finis) pour déterminer avec précision le nombre de tokens supplémentaires acceptables, au lieu de se plier à une taille de lot (batch_size) codée en dur.

**Métriques de succès :**
- Throughput ≥ 2x sur 4 requêtes concurrentes vs baseline séquentiel (bench avec GPT-2 ou TinyLlama)
- Latence P99 < 3x la latence P50 (preuve que le lock ne bloque plus)
- 0 OOM sur un burst de 20 requêtes courtes simultanées (backpressure fonctionne)

**Tests de validation :**
```bash
# Test unitaire : vérifier que le tokenizer ne bloque pas le lock
pytest tests/test_continuous_batching.py -v -k "test_concurrent"

# Bench throughput (à créer) :
VRM_CONTINUOUS_BATCHING=1 python benchmarks/bench_batcher_concurrent.py --model gpt2 --concurrent-requests 4
# Attendu : tok/s total > 2x le single-request

# Test backpressure :
pytest tests/test_continuous_batching.py -v -k "test_backpressure_429"
```

---

## Chantier 2 : Câbler le transfert Rust manquant (TransferManager) — PRIORITÉ MOYENNE

**Problème actuel** : `transfer_manager.py` implémente théoriquement une stratégie "1.5" (bypass VRAM à VRAM direct) via `vramancer_rust.direct_vram_copy()`. Mais la librairie Rust exportée (`libvramancer_rust.so`) ne contient pas cette fonction dans ses bindings PyO3. Le module prétend l'utiliser sans que ce soit effectif.

**Comment s'y prendre :**
1. **Côté Rust (`rust_core/src/lib.rs`)** :
   - Ajouter un attribut `#[pyfunction]` pour exposer explicitement `direct_vram_copy` à Python.
   - La fonction prendra des pointeurs bruts en entrée (`src_ptr: usize`, `dst_ptr: usize`, `size: usize`).
   - S'interfacer avec le driver CUDA via `libloading` (déjà partiellement en place) pour appeler dynamiquement `cuMemcpyPeerAsync`.
   - **Triple-buffering** : Câbler la logique de triple-buffering asynchrone existante en utilisant `cuEventRecord`/`cuEventSynchronize` pour masquer la latence PCIe. Ainsi, le transfert du bloc N+1 a lieu en tâche de fond pendant le calcul du bloc N.
2. **Côté Python (`core/transfer_manager.py`)** :
   - Récupérer les adresses mémoire des tenseurs PyTorch en C en utilisant `tensor.data_ptr()`.
   - Synchroniser le contexte CUDA pour ne pas corrompre le graphe PyTorch : s'accrocher au stream PyTorch courant (`torch.cuda.current_stream().cuda_stream`).
   - Créer un fallback robuste : intercepter les erreurs d'exécution (ex: pont IOMMU désactivé empêchant le P2P) et dégrader gracieusement vers la stratégie "4" (CPU-staged pinned memory).

**Métriques de succès :**
- Latence transfert GPU→GPU < 2x la latence théorique PCIe (mesurée avec `nvidia-smi nvlink` ou timing `torch.cuda.Event`)
- Fallback IOMMU fonctionne automatiquement en VM Proxmox sans crash
- Overhead triple-buffering < 5% par rapport au transfert synchrone simple

**Tests de validation :**
```bash
# Test Rust unitaire :
cd rust_core && cargo test -- --nocapture

# Test Python integration :
pytest tests/test_transfer_manager.py -v -k "test_strategy_1_5"

# Test fallback IOMMU (en VM Proxmox) :
VRM_TRANSFER_P2P=0 pytest tests/test_transfer_manager.py -v -k "test_fallback_cpu_staged"

# Bench transfert (à créer) :
python benchmarks/bench_transfer.py --size 1GB --src-gpu 0 --dst-gpu 1
# Attendu : ~10 GB/s en P2P natif, ~6 GB/s en CPU-staged
```

---

## Chantier 3 : Assainir la mémoire hiérarchique — PRIORITÉ BASSE

**Problème actuel** : `core/hierarchical_memory.py` (~1000 lignes) est classé *Red Flag*. Sa méthode `_evict_lru()` produit des logs rassurants ("physically offloaded") mais ne déplace aucune donnée en réalité, se contentant de gérer des métadonnées fictives.

**⚠️ Faut-il vraiment l'implémenter ?** Avec 40 GB de VRAM totale (RTX 3090 24GB + RTX 5070 Ti 16GB), la plupart des modèles ≤14B en BF16 tiennent en VRAM. Le NF4/GGUF permet même du 30B+ single-GPU. Ce chantier n'a de sens que pour :
- Des modèles >30B en BF16 sans quantification
- Des contextes très longs (>32K tokens) avec KV cache volumineux
- Un futur upgrade hardware avec GPU plus petites

**Alternative minimaliste** : plutôt que d'implémenter les 6 tiers, **purger le code stub** et ne garder que 2 tiers fonctionnels (VRAM + Pinned RAM). Honnêteté > ambition.

**Si on décide d'implémenter :**
1. **Implémentation DRAM (Pinned RAM)** :
   - Supprimer le code "stubs" des métadonnées et allouer physiquement la mémoire.
   - Utiliser la fonction native `torch.empty(size, pin_memory=True)` pour créer un tampon côté hôte (RAM verrouillée, évitant le swap OS et maximisant les débits PCIe DMA).
   - Mettre en place le transfert asynchrone : `tensor_dram.copy_(tensor_vram, non_blocking=True)`.
2. **Implémentation NVMe (Stockage disque)** :
   - Remplacer l'approche actuelle par un système sérialisant les tenseurs au format `safetensors`.
   - Introduire la lecture mémoire mappée (memory-mapping, `mmap`) pour lire directement du disque vers la RAM.
3. **Synchronisation du Block Router** :
   - Modifier `core/block_router.py` (qui assume actuellement que l'offload est instantané) pour qu'il synchronise les files d'attente d'exécutions GPU (compute_engine) et d'E/S physiques.

**Métriques de succès :**
- `_evict_lru()` déplace **réellement** les données (vérifiable via `torch.cuda.memory_allocated()` avant/après)
- Round-trip VRAM→RAM→VRAM < 500ms pour un tenseur de 1 GB
- Round-trip VRAM→NVMe→VRAM < 2s pour un tenseur de 1 GB (NVMe Gen4)
- Aucun log mensonger ("physically offloaded" uniquement si le transfert a eu lieu)

**Tests de validation :**
```bash
# Test que l'eviction libère vraiment de la VRAM :
pytest tests/test_hierarchical_memory.py -v -k "test_evict_actually_frees_vram"

# Test round-trip integrity (les données survivent à l'offload) :
pytest tests/test_hierarchical_memory.py -v -k "test_roundtrip_vram_ram"

# Bench latence offload :
python benchmarks/bench_offload.py --size 1GB --tiers vram,ram,nvme
```
