# Fastpath (USB4 / RDMA / SFP+ / Stub)

Le module fastpath fournit une abstraction de transport local/ultra‑bas‑niveau permettant d'évaluer ou brancher des interfaces à faible latence (USB4, RDMA, SFP+), avec fallback stub.

## Endpoints API

```
GET  /api/fastpath/capabilities   # Capacités + interface sélectionnée
GET  /api/fastpath/interfaces     # Interfaces détectées + benchmarks (cache TTL)
POST /api/fastpath/select         # {"interface": "eth0"} ou {"kind": "usb4"}
```

## Sélection d'interface
- Automatique: premier élément détecté ou réordonné si `VRM_FASTPATH_IF` défini.
- Manuelle: `POST /api/fastpath/select` (re-benchmark immédiat).

## Benchmarks
- Benchmarks synthétiques (mmap écriture/lecture) => latence moyenne par interface.
- Cache TTL par défaut: 30s (override `VRM_FASTPATH_BENCH_TTL`).
- Forcer recalcul côté serveur: utiliser `force=1` (à implémenter si besoin) ou attendre expiration TTL / re-select.

## Métriques Prometheus
- `vramancer_fastpath_interface_latency_seconds{interface,kind}` (Gauge)
- `vramancer_fastpath_bytes_total{method, direction}` (Counter)
- `vramancer_fastpath_latency_seconds{method, op}` (Histogram)

## Variables d'environnement
| Variable | Description | Défaut |
|----------|-------------|--------|
| `VRM_FASTPATH_IF` | Interface ou type à prioriser (`eth0`, `usb4`, `rdma`) | (none) |
| `VRM_FASTPATH_BENCH_TTL` | Durée cache bench en secondes | 30 |

## Extension future
- Intégration réelle RDMA (verbs) / GPUDirect.
- Détection active latence réseau (ping micro paquet custom).
- Sélection adaptative basée sur charge & jitter.
- Mode multi-voies (agrégation). 

---
© 2025 VRAMancer
