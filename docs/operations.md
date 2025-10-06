# Guide Opérations VRAMancer

## Démarrage / Arrêt
API principale : `python -m vramancer.main`
Unified API (lite) : `python -m core.api.unified_api`

## Variables critiques
| Variable | Rôle |
|----------|------|
| VRM_API_PORT | Port API principal |
| VRM_METRICS_PORT | Port métriques Prometheus |
| VRM_UNIFIED_API_QUOTA | Quota requêtes/token |
| VRM_READ_ONLY | Bloque mutations (1=ON) |
| VRM_SQLITE_PATH | Active persistence SQLite |
| VRM_API_TOKEN | Secret HMAC (rotation) |
| VRM_RATE_MAX | Rate limit fenêtre |

## Maintenance
Rotation HMAC forcée : `POST /api/security/rotate`
Reset quotas : `POST /api/quota/reset`
Diagnostics : `python scripts/diagnostics.py`

## Persistence
Définir `VRM_SQLITE_PATH=state.db` avant lancement unified_api. Sauvegardes automatiques workflows et rounds FL.

## Sauvegarde / Restauration
Fichier SQLite + éventuels journaux HA (si activés). Copier `state.db` et réinjecter via simple redeploi.

## Observabilité
Métriques: `curl :9108/metrics | grep vramancer_`
Latence API: Histogram `vramancer_api_latency_seconds`.
XAI: `vramancer_xai_requests_total{kind=...}`

## Sécurité
Enprod: ne pas définir `VRM_TEST_MODE`; définir `VRM_API_TOKEN`; ne pas désactiver la rotation.
Limiter exposition unified_api derrière reverse proxy TLS.

## Plan de reprise
1. Restaurer state.db
2. Relancer services (API + scheduler)
3. Vérifier `scripts/diagnostics.py` sortie `health` ok
4. Consulter métriques HA / placements.

## Montée de version
1. Tag git (vx.y.z)
2. CI génère wheel & image
3. Déployer image runtime slim
4. Migration manuelle si schéma persistence évolue (table ajoutable sans downtime).

## Limites actuelles
- Pas de compression DB ni archivage cycles.
- Pas d’OIDC / RBAC avancé.
- Secure aggregation basique.
