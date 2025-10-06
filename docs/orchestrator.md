# Orchestrateur VRAMancer

Ce document décrit l'architecture, les responsabilités et les points d'extension de l'orchestrateur de ressources mémoire/compute utilisé par VRAMancer.

## Objectifs
- Placement intelligent des blocs mémoire et sous-modèles sur une hiérarchie (L1→L6)
- Minimisation des copies (promotion/demotion adaptatives)
- Agrégation multi-GPU / multi-hôte (future extension)
- Résilience (journal de réplication HA optionnel)
- Observabilité (métriques Prometheus + traces optionnelles)

## Hiérarchie mémoire
Niveaux typiques (configurable):
| Niveau | Description | Latence relative | API interne |
|--------|-------------|------------------|-------------|
| L1 | VRAM locale GPU prioritaire | 1× | alloc_block(level="L1") |
| L2 | VRAM secondaire / autre GPU | 1.3× | ... |
| L3 | Host Pinned (pinned CPU) | 2–3× | ... |
| L4 | Host pageable | 4–6× | ... |
| L5 | Fichier mmap (SSD rapide) | 12–20× | ... |
| L6 | Stockage froid (archive) | >50× | ... |

Le moteur choisit la cible initiale selon: taille, fréquence d'accès anticipée (hotness prédictive), pression VRAM, priorité tâche.

## Hotness & Promotion
Chaque bloc possède:
- Compteur de hits pondéré (LFU partiel)
- Timestamp dernier accès (LRU partiel)
- Décroissance exponentielle configurable (demi-vie, défaut 60s)
Score hotness = f(LFU, LRU_decay). Un seuil supérieur déclenche une promotion, un seuil inférieur + pression déclenche une démotion.

## Éviction
Politique hybride:
1. Collecte candidats par niveau si pression > threshold
2. Score d'éviction = (coldness * taille) / (coût_recharge + priorité)
3. Compression opportuniste (zstd/lz4/gzip/none) si bloc déplacé vers L5/L6
4. Journalisation (delta) si réplication HA active

## Réplication HA
- Activée via `VRM_HA_REPLICATION=1`
- Pairs dans `VRM_HA_PEERS="host1:5010,host2:5010"`
- Journal append-only mémoire → disques (rotation si > `VRM_HA_JOURNAL_MAX`)
- Deltas signés HMAC (secret dérivé horaire + nonce anti-rejeu)
- Compression adaptative: essaye zstd, puis lz4, puis gzip, sinon brut
- Métriques: `vramancer_ha_journal_size_bytes`, `vramancer_ha_journal_rotations_total`

## Scheduler Opportuniste
L'orchestrateur collabore avec le scheduler pour insérer des tâches de maintenance (promotion batch, compaction, recompute hotness) dans les fenêtres d'inactivité GPU/CPU.

Tâches support:
- warmup: pré-chargement blocs chauds futurs
- compress: compression paresseuse L4→L5
- noop: instrumentation / tests

Endpoints:
- `POST /api/tasks/submit{,_batch}`
- `GET /api/tasks/{status,history}`

Métriques clés:
- `vramancer_tasks_submitted_total`
- `vramancer_tasks_completed_total`
- `vramancer_tasks_failed_total`
- `vramancer_tasks_running`

## Fastpath
Lorsqu'un transfert inter-niveau implique un périphérique réseau rapide ou USB4, l'orchestrateur peut déléguer au module fastpath (benchmarks TTL). Métriques: `vramancer_fastpath_interface_latency_seconds`.

## Observabilité
Exporter Prometheus (port configurable `VRM_METRICS_PORT`), labels ressources (`backend`, `device`).
Traces OpenTelemetry si `VRM_TRACING=1`. Spans: `memory.promote`, `memory.demote`, `memory.evict_cycle`, `orch.place_block`.

## Sécurité
Endpoints sensibles (éviction forcée, rotation HA) requièrent rôle (header `X-API-ROLE`). HMAC appliqué aux deltas HA. Rate limiting configurable (désactivable tests via `VRM_DISABLE_RATE_LIMIT=1`).

## Extension Points
| Point | Méthode / Hook | Usage |
|-------|----------------|-------|
| Placement custom | `OrchestratorStrategy.place(block)` | Surclasser heuristique par défaut |
| Score hotness | `HotnessModel.score(block)` | Ajuster la pondération LRU/LFU |
| Compression | `CompressionManager.compress(data, hint)` | Ajouter algorithme (e.g. brotli) |
| Fastpath | `FastInterface.send/recv` | Support RDMA réel, io_uring |
| Estimation runtime | `RuntimeEstimator.estimate(task)` | Priorisation scheduler |

## Bonnes pratiques production
- Activer HA + compression zstd
- Configurer rotation secret HMAC (par défaut active) – désactiver seulement en test
- Surveiller métriques d'éviction + latence promotions
- Définir alertes sur croissance anormale journal HA
- Limiter PyQt5 dans les environnements serveurs (utiliser build lite)

## Build Lite
Réduit dépendances UI lourdes. Installation:
```bash
pip install -r requirements-lite.txt
python -m vramancer.main --mode cli
```

## TODO Futurs
- Placement cross-node (graph partition)
- Préfetch probabiliste séquence tokens
- CGroup / MIG awareness
- Backpressure producteur/consommateur fastpath
- Surcouche gRPC streaming

---
© 2025 VRAMancer
