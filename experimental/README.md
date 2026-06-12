# Experimental modules

Modules non validés sur matériel réel. APIs instables. Aucune garantie.
Contributions de validation bienvenues.

Importer un de ces modules émet un `UserWarning` tant que
`VRM_EXPERIMENTAL=1` n'est pas défini dans l'environnement.

| Module | Statut |
|---|---|
| `vram_lending.py` | En développement actif (branche `feat/v6-lending-cooperative`). Pas encore benchmarké de bout en bout sur matériel hétérogène réel. |
| `hierarchical_memory.py` | Mémoire hiérarchique 6 niveaux, non benchmarkée. |
| `cross_vendor_bridge.py` | Bridge AMD↔NVIDIA (`PipelinedTransport`, `SharedMemTransport`, `DMABufTransport`), non testable sans GPU AMD. `ReBarTransport` (Strategy 1.7) reste ici aussi : un run précédent (mai 2026, conf Proxmox différente) avait BAR1 ≥ VRAM et mesurait ~21-24 GB/s, mais sur la VM actuelle BAR0 = 8-16 MB (legacy window) → `ReBarTransport.available = False`, awaiting bare-metal/Proxmox-conf re-validation (T2.3, `bench_transfer_strategies.py`). `PipelinedTransport` (Strategy 2 fallback), elle, est validée et utilisée par `core/transfer_manager.py` même hors ReBAR (~178 Gbps vs ~16 Gbps CPU-staged sur 1 GB, mesuré). |
| `wake_on_inference.py` | Wake-on-Inference, hors du périmètre cœur (inférence single-node multi-GPU). |
| `aitp_protocol.py`, `aitp_fec.py` | Protocole AITP + FEC Reed-Solomon. Réinvente NCCL/QUIC ; gain non mesurable sur Ethernet grand public. Gelé (voir docs/history). |
| `nat_traversal.py` | NAT traversal / STUN, hors périmètre cœur. |
| `fibre_fastpath.py` | RDMA / GPUDirect, non testable sans matériel IB/RoCE. |
| `cluster_discovery.py` | Cluster discovery / Bully election, hors périmètre cœur sauf si le multi-nœud devient un axe assumé. |

Référence : `Fable.md` (plan de restructuration), Phase 2.
