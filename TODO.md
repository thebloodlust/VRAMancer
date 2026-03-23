# VRAMancer Todo

## ✅ Complété

- ✅ **Wake On Inference** — `WakeOnInferenceManager` complet : register/unregister, WoL magic packets (subnet configurable), `wait_for_nodes()` avec polling ClusterDiscovery, stats. (`core/wake_on_inference.py`)
- ✅ **WebGPU** — Backend WebGPU fonctionnel : WebSocket server, sérialisation binaire, quantification INT8, speculative decoding, retry/fallback. (`core/backends_webgpu.py`, `core/network/webgpu_node.py`)
- ✅ **Déterminer/Forcer l'ordre des GPUs (L0 vs L1)** — `rank_gpus()` trie par compute capability × SM count (Blackwell CC 12.x en tête). Override via `VRM_GPU_ORDER=1,0`. (`core/model_splitter.py`)
- ✅ **Équilibrage de charge asymétrique (Load Imbalance)** — `_split_by_vram()` pondéré par compute_scores : VRAM × puissance GPU. Une 5070 Ti reçoit proportionnellement plus de couches qu'une 3090. (`core/model_splitter.py`)
- ✅ **Groupes Privés (Cercles de Confiance)** — Tables `groups` + `group_members` dans le Ledger SQLite. `create_group()` émet un invite_token (hash SHA-256). `join_group()`, `is_group_member()`, `get_group_members()`, `validate_group_token()`. (`core/swarm_ledger.py`)
- ✅ **Redondance K-Répétition (Fault Tolerance)** — `BlockReplicator` : réplication k-fold (GPU ou CPU pinned memory), `checkpoint()`/`get_checkpoint()` pour sauvegarder le hidden state intermédiaire, failover via `get_replica()`. (`core/gpu_fault_tolerance.py`)
- ✅ **Battery Aware / Wake-Lock** — Scoring battery-aware dans le task dispatcher WebGPU : batterie < 15% non-branché = exclu, seuils 30%/50% progressifs. Blackwell priorisé (score 120). (`core/network/webgpu_node.py`)

## En cours / Futur

- Tests Multi OS PC et MAC (CI cross-platform)
- **Support WebGPU/WebNN Mobile :** Intégrer les smartphones au Swarm. Les puces récentes (Apple Neural Engine, Snapdragon NPU, Google Tensor) ont des capacités matricielles exceptionnelles.
- **Profilage Asymétrique Extrême :** Le `layer_profiler` doit pouvoir assigner dynamiquement des petites charges aux téléphones (ex: 1 couche d'attention) et des grosses charges aux serveurs (ex: 20 couches), selon leur bande passante (WiFi/5G) et leur VRAM mobile (UMa).
