# VRAMancer Todo

## ✅ Complété

- ✅ **Wake On Inference** — `WakeOnInferenceManager` complet : register/unregister, WoL magic packets (subnet configurable), `wait_for_nodes()` avec polling ClusterDiscovery, stats. (`core/wake_on_inference.py`)
- ✅ **WebGPU** — Backend WebGPU fonctionnel : WebSocket server, sérialisation binaire, quantification INT8, speculative decoding, retry/fallback. (`core/backends_webgpu.py`, `core/network/webgpu_node.py`)
- ✅ **Déterminer/Forcer l'ordre des GPUs (L0 vs L1)** — `rank_gpus()` trie par compute capability × SM count (Blackwell CC 12.x en tête). Override via `VRM_GPU_ORDER=1,0`. (`core/model_splitter.py`)
- ✅ **Équilibrage de charge asymétrique (Load Imbalance)** — `_split_by_vram()` pondéré par compute_scores : VRAM × puissance GPU. Une 5070 Ti reçoit proportionnellement plus de couches qu'une 3090. (`core/model_splitter.py`)
- ✅ **Groupes Privés (Cercles de Confiance)** — Tables `groups` + `group_members` dans le Ledger SQLite. `create_group()` émet un invite_token (hash SHA-256). `join_group()`, `is_group_member()`, `get_group_members()`, `validate_group_token()`. (`core/swarm_ledger.py`)
- ✅ **Redondance K-Répétition (Fault Tolerance)** — `BlockReplicator` : réplication k-fold (GPU ou CPU pinned memory), `checkpoint()`/`get_checkpoint()` pour sauvegarder le hidden state intermédiaire, failover via `get_replica()`. (`core/gpu_fault_tolerance.py`)
- ✅ **Battery Aware / Wake-Lock** — Scoring battery-aware dans le task dispatcher WebGPU : batterie < 15% non-branché = exclu, seuils 30%/50% progressifs. Blackwell priorisé (score 120). (`core/network/webgpu_node.py`)
- ✅ **Continuous Batcher câblé** — `generate()` dans `inference_pipeline.py` route vers le `ContinuousBatcher` quand il tourne. Auto-start via `VRM_CONTINUOUS_BATCHING=1` au chargement API.
- ✅ **Benchmarks tok/s publiés** — GPT-2, TinyLlama-1.1B, Mistral-7B : overhead <1%, parfois +1-8% plus rapide que HF natif. Voir `benchmarks/BENCHMARK_RESULTS.md`.
- ✅ **Rust bypass corrigé** — Faux "6-14x" corrigé en vrais 1.3-1.6x dans `VRAMANCER_RUST_BYPASS.md`.
- ✅ **Nettoyage dead code** — 19 fichiers déplacés vers `_deprecated/`. 0 code mort dans `core/`.
- ✅ **Audit complet des fonctions** — 70 fichiers source dans `core/`, 48 production-ready (🟢), 22 utiles (🟡), 0 dead (🔴).

## ⚠️ À corriger (dette technique)

- ✅ **aitp_fec.py — Vrai Reed-Solomon GF(2^8)** : L'ancien audit prétendait "faux RS / simple XOR" mais le code implémente un *vrai* Cauchy RS sur GF(2^8) avec tables exp/log, matrice de Cauchy, et élimination de Gauss. La fast-path `decode()` est correcte (non-contiguous indices gérés). **Résolu — rien à changer.**
- ✅ **speculative_decoding.py — Câblé au pipeline** : `swarm_verify_callable` câblé à `pipeline.infer()` (forward brut → logits). Draft model : auto-mapping par famille (Qwen→0.5B, Llama→1B, GPT-2→distilgpt2) + env `VRM_DRAFT_MODEL`. Self-drafting supprimé (pas de speedup). Device placement corrigé.
- ✅ **holographic_memory.py → parity_memory.py** : Renommé honnêtement en `core/parity_memory.py`. Classe `ParityKVManager` avec alias backward-compat `HolographicKVManager`. Méthodes `encode()`/`heal()` + alias `encode_hologram()`/`heal_hologram()`. Docstring pointe vers `aitp_fec.FastFEC` pour multi-fault. `_deprecated/holographic_memory.py` redirige vers le nouveau module.

## 🔴 Manquant (bloque la crédibilité)

- **Benchmark multi-GPU** : Bloqué par TDR Xorg en VM (les deux GPUs sont attachées au display server). Résolution : `sudo nvidia-smi -pm 1` ou GPU headless. Sans ça, aucune preuve que le multi-GPU fonctionne en perf.
- **Test de charge continuous batcher** : Le batcher est câblé mais jamais testé avec N requêtes concurrentes. Script nécessaire : N=1,2,4,8,16,32 clients simultanés, mesurer tok/s total + P50/P95/P99.
- **Benchmark VRAMancer vs vLLM** : Aucune comparaison avec le concurrent direct. Même modèle, même GPU, mêmes paramètres. vLLM gagnera probablement en throughput pur (PagedAttention CUDA C++), mais VRAMancer peut gagner sur les cas hétérogènes.
- **CI avec GPU** : Aucun pipeline CI n'exécute les tests GPU. Les 638 tests passent en mode stub (`VRM_MINIMAL_TEST=1`).

## En cours / Futur

- Tests Multi OS PC et MAC (CI cross-platform)
- **Support WebGPU/WebNN Mobile :** Intégrer les smartphones au Swarm. Les puces récentes (Apple Neural Engine, Snapdragon NPU, Google Tensor) ont des capacités matricielles exceptionnelles.
- **Profilage Asymétrique Extrême :** Le `layer_profiler` doit pouvoir assigner dynamiquement des petites charges aux téléphones (ex: 1 couche d'attention) et des grosses charges aux serveurs (ex: 20 couches), selon leur bande passante (WiFi/5G) et leur VRAM mobile (UMa).
