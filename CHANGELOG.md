## Changelog

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
