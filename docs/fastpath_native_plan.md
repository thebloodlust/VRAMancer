# Plan d'implémentation Fastpath Natif (RDMA / io_uring / Fibre)

## Objectifs
- Latence sous-milliseconde transfert blocs poids/tensors inter-nœuds
- Zero-copy userland→NIC (ou near-zero copies) via RDMA / io_uring
- Intégration transparente dans `FastHandle` (mêmes appels send/recv)

## Phases
1. Abstraction stable (FAIT) : `open_low_latency_channel()` + `FastHandle.send/recv`
2. Plugin io_uring
   - Extension C/Python (liburing) ou py-uring
   - File de SQEs pré-allouées (WRITE/READ) sur fichier partagé ou socket AF_XDP (optionnel)
3. Backend RDMA verbs
   - Utilisation `pyverbs` (ibv_context, pd, cq, qp)
   - Échange des QP attributes via canal de contrôle (UDP/TCP léger)
   - Post send/recv sur buffers pinned + MR registration
4. Fibre SFP+/PCIe custom
   - Wrapper userspace driver existant ou module kernel + ioctl
5. Benchmark & Sélection dynamique
   - `memory_benchmark` étendu: enregistre latence transport et priorise

## API proposée (extension)
```python
ch = open_low_latency_channel(prefer="rdma")
ch.capabilities  # {"method":"rdma", "bw_gbps":..., "lat_us":...}
```

## Sécurité
- Token déjà présent côté API mémoire
- Pour RDMA: option chiffrage application (libsodium/ChaCha20) avant post_send

## Métriques
- vramancer_fastpath_bytes_total{method="mmap|rdma|io_uring"}
- vramancer_fastpath_latency_seconds_bucket

## Tests
- Echo tests (payload 4KB/1MB)
- Détection fallback (si RDMA indispo → mmap)

