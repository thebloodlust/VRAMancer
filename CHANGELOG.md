## Changelog

### 1.1.0 (2025-10-15) - Production-Ready Release 🚀

**Major Changes:**
- **API Production-Ready**: Nouveau `core/production_api.py` avec logging structuré, error handling robuste, validation complète
- **Logging Unifié**: Migration de `print()` vers logger structuré avec support JSON, rotation, multi-niveaux
- **Sécurité Renforcée**: Documentation complète sécurité production, validation automatique, alertes credentials par défaut
- **Scripts de Validation**: `scripts/check_production_ready.sh` - validation automatique configuration production
- **Migration Guide**: Documentation complète migration dev → production

**Security Enhancements:**
- ⚠️ **WARNING**: Credentials admin/admin par défaut documentés comme dangereux
- Ajout `VRM_DISABLE_DEFAULT_ADMIN` pour désactiver compte par défaut
- Validation obligatoire `VRM_AUTH_SECRET` (32+ caractères) en production
- Script de détection secrets hardcodés et debug mode activé
- Guide complet sécurisation: `SECURITY_PRODUCTION.md`

**API Improvements:**
- Health check endpoints: `/health`, `/ready` (Kubernetes/Docker ready)
- Error handlers structurés (404, 500) avec logging
- Middleware request/response logging (mode debug)
- Validation robuste GPU/System/Nodes endpoints
- Gestion d'erreurs complète avec fallbacks

**Logging:**
- Logger centralisé dans `core/logger.py` (existant, documenté)
- Support JSON structuré (ELK/Splunk ready)
- Rotation automatique logs (10 MB, 5 backups)
- Colored formatter pour développement
- Context logging pour traçabilité

**Documentation:**
- `SECURITY_PRODUCTION.md` (375 lignes) : Guide sécurité complet, checklist pré-production
- `MIGRATION_GUIDE.md` (420 lignes) : Guide migration dev → production
- `scripts/check_production_ready.sh` : Validation automatique (8 sections de vérification)

**Deployment:**
- Exemples systemd, Docker, Kubernetes
- Configuration `.env.production` type
- Scripts de rollback
- Monitoring Prometheus ready

**Breaking Changes:**
- **AUCUN** : 100% rétrocompatible via `api.py` wrapper
- Mode simple accessible via `VRM_PRODUCTION=0`
- Migration progressive recommandée

**Environment Variables (New):**
- `VRM_PRODUCTION` : Active mode production (défaut: 1)
- `VRM_DISABLE_DEFAULT_ADMIN` : Désactive admin/admin (défaut: 0)
- `VRM_LOG_JSON` : Format JSON pour logs (défaut: 0)
- `VRM_LOG_CONSOLE` : Active sortie console (défaut: 1)
- `VRM_LOG_DIR` : Répertoire logs (défaut: logs/)

**Migration Path:**
```bash
# Ancien (dev)
python api_simple.py

# Nouveau (production)
export VRM_AUTH_SECRET=$(openssl rand -hex 32)
export VRM_PRODUCTION=1
python api.py
```

**Files Added:**
- `core/production_api.py` : API production-ready
- `api.py` : Wrapper avec fallback
- `SECURITY_PRODUCTION.md` : Guide sécurité
- `MIGRATION_GUIDE.md` : Guide migration
- `scripts/check_production_ready.sh` : Script validation

**Files Preserved:**
- `api_simple.py` : Conservé pour développement/debug
- Tous les dashboards `*_simple.py` : Conservés (migration future)

**Testing:**
- Validation script testé sur Linux/macOS
- API production testée endpoints critiques
- Fallback api_simple.py fonctionnel

**Known Issues:**
- Dashboards `*_simple.py` toujours en mode prototype (migration à venir)
- `debug=True` présent dans certains fichiers dashboard (warning émis)

**Recommendations:**
1. Exécuter `./scripts/check_production_ready.sh` avant déploiement
2. Lire `SECURITY_PRODUCTION.md` intégralement
3. Changer mot de passe admin immédiatement
4. Définir `VRM_AUTH_SECRET` en production
5. Activer logging JSON en production

---

### 0.2.4 (2025-10-04)
Enhancements:
- Fallback tokenizer Python pur (`BasicTokenizer`) activable via `VRM_FORCE_BASIC_TOKENIZER=1` ou automatiquement si `transformers` indisponible.
- Support variable `USE_SLOW_TOKENIZER=1` pour forcer tentative `use_fast=False` avant fallback basic.
- Tests à prévoir (non inclus dans cette entrée) pour header `X-Request-ID` et secure aggregation (documentation mise à jour).
- Correction style requirements (espace accidentel supprimé avant pydantic si présent dans future commit).

Notes:
- Version interne incrémentée `0.2.4`.
- Aucun changement d'API cassant; uniquement robustesse et diagnostics.

### 0.2.3 (2025-10-04)
Windows & Robustness:
- Fallback ONNX (VRM_DISABLE_ONNX) + compute engine stubs sans crash.
- Auto-détection port API dans dashboard Qt (5030→5010) + retries configurables.
- Script `scripts/api_autodetect.py` pour diagnostics de connectivité.
- Ajout VRM_API_DEBUG (traces HTTP) & VRM_STRICT_IMPORT (fail fast optionnel).
- Dashboard Qt: indicateur visuel (pastille), bouton Reconnect, fallback 127.0.0.1, logs debug.

API & Observabilité:
- Endpoint `/api/env` exposant état runtime (features, flags, quotas, modes).

Docs & Packaging:
- README: section Windows tokenizers (wheel vs build), section Qt fiabilisation.
- Ajout `requirements-windows.txt` (transformers 4.46.2 + tokenizers 0.20.1) pour éviter compilation Rust/MSVC.
- Documentation variables supplémentaires (VRM_DISABLE_ONNX, VRM_API_DEBUG, VRM_STRICT_IMPORT).

Tests:
- Nouveau test `test_env_endpoint.py` validant `/api/env` + script autodetect JSON parse.

Hardening:
- Strict import mode pour environnements contrôlés CI/CD.

Note:
Port par défaut API interne: 5030 (le tableau env mentionne 5010: correction planifiée – rétrocompat maintenue via autodétection).

### 0.2.0 (2025-10-03)
Features:
- Unified API: quotas, read-only, XAI, FL weighted + clipping + noise, workflows listing, persistence (SQLite opt-in).
- Metrics: API latency, quota/read-only counters, XAI requests, placement instrumentation.
- CI: Multi-OS matrix, wheel build artifacts, security audit (pip-audit + bandit), Docker build smoke.
- Docker: Multi-stage (build + runtime slim).
- Persistence: workflows & federated rounds, listing endpoint.
- XAI: Feature attribution baseline endpoint.
- Marketplace: basic plugin registry & listing.
- Diagnostics script.

Post‑tag additions (to be included in next 0.2.x):
- Endpoint `/api/health` (Docker HEALTHCHECK).
- Secure aggregation prototype (`/api/federated/secure` GET/POST) avec masquage déterministe basique & retrait lors agrégation.
- Sandbox plugin `sandboxed_run` (expérimental, restrictions simples sur builtins/imports).
- Logging JSON (`VRM_LOG_JSON=1`) et logging HTTP (`VRM_REQUEST_LOG=1`).
- SocketIO désactivable (`VRM_DISABLE_SOCKETIO=1`) + fallback propre.
- Détection VRAM réelle via pynvml si disponible.

### 0.2.2 (2025-10-03)
Features:
- Auth forte (JWT access + refresh) + hash PBKDF2 + bootstrap admin.
- Endpoints: /api/login, /api/token/refresh.
Security:
- Protection endpoints non publics via Bearer token.
Docs:
- README endpoints section (JWT usage) ajouté.

Internal / Polish:
- Version bump central (__version__).
- Tests added (XAI, placement, quota reset, workflows list, persistence).
- Cleanup duplicated classes, improved error resilience.

### 0.1.0-dev
- Initial orchestrator, memory tiers, scheduler, security base, HA journal, systray enhancements.
