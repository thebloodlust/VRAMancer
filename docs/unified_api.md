# Unified API (Prototype évolué)

Expose en un seul service plusieurs fonctionnalités prototypes :
- Workflows no-code (création basique)
- Simulation Digital Twin (simulate / replay)
- Federated Learning (cycle round simplifié)

## Endpoints
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | /api/info | Statut rapide & nombre de workflows |
| GET | /api/version | Version courante |
| POST | /api/workflows | Crée un workflow (validation Pydantic) |
| GET | /api/workflows | Liste workflows (in-memory + persistence) |
| GET | /api/workflows/<id> | Récupère un workflow |
| POST | /api/twin/simulate | Simule une action (JSON) |
| GET | /api/twin/replay | Rejoue l'historique simulation |
| GET | /api/twin/state | Snapshot cluster + taille historique |
| POST | /api/federated/round/start | Démarre un round federated |
| POST | /api/federated/round/submit | Soumet une update (value, weight) |
| GET | /api/federated/round/aggregate | Agrège (moyenne pondérée) |
| POST | /api/quota/reset | Réinitialise compteurs quota (maintenance) |
| POST | /api/xai/explain | Explication XAI |
| GET | /api/xai/explainers | Liste explainers |
| GET | /api/marketplace/plugins | Plugins enregistrés + signatures |

## Lancement
```bash
python -m core.api.unified_api
```

## Sécurité & politiques
- Auth / intégrité via `install_security` (HMAC + rotation facultative)
- Rate limiting global (module security)
- Quota par token (`VRM_UNIFIED_API_QUOTA`, en-tête `X-API-TOKEN`)
- Mode read-only (`VRM_READ_ONLY=1`) bloque POST/PUT/DELETE → métrique `vramancer_api_read_only_blocked_total`
- Endpoint maintenance `/api/quota/reset` exclu du quota

## Métriques ajoutées
- `vramancer_api_quota_exceeded_total`
- `vramancer_api_read_only_blocked_total`

## Améliorations récentes
- Validation Pydantic des workflows (structure tâches)
- Agrégation fédérée pondérée (`weight`)
- Snapshot Digital Twin (`/api/twin/state`)
- Quotas & reset
- Intégration module sécurité (HMAC + rate limit + RBAC minimal)

## Limites restantes
- Pas de persistance durable des workflows / états twin
- Pas (encore) de secure aggregation chiffrée
- Pas de DAG complexe / dépendances inter-tâches
- Pondération fournie par client (non vérifiée)

## Prochaines étapes suggérées
- Ajouter auth HMAC (import de install_security)
- Étendre Digital Twin : latence, VRAM, fiabilité
- Ajout de pondération et secure aggregation FL
- Scheduler pour exécution asynchrone workflows
