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
