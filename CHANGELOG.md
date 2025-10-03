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

Internal / Polish:
- Version bump central (__version__).
- Tests added (XAI, placement, quota reset, workflows list, persistence).
- Cleanup duplicated classes, improved error resilience.

### 0.1.0-dev
- Initial orchestrator, memory tiers, scheduler, security base, HA journal, systray enhancements.
