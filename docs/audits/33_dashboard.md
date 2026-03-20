# Audit — dashboard/ (4 modules)

## dashboard/__init__.py (~20 LOC) — ✅ Bon
Entry point avec fallback gracieux si CLI indisponible.

## dashboard/cli_dashboard.py (~80 LOC) — ⚠️ Mixte
Dashboard terminal pour monitoring API en temps réel.
- 🟡 HTTP non chiffré (pas de TLS)
- 🟡 `requests.get()` bloquant (timeout 5s peut geler le dashboard)
- 🟡 Port non validé (environ `VRM_API_PORT`)
- 🟡 `except Exception` avale les vraies erreurs

## dashboard/dashboard_web.py (~350 LOC) — ⚠️ Mixte
Dashboard web Flask avec visualisation 3D (Tailwind, Alpine.js, 3D Force Graph).
- 🔴 **CORS wide open** : `cors_allowed_origins="*"`
- 🔴 `/api/debug/logs` expose tous les logs sans auth
- 🟡 Template HTML ~400 lignes inline (devrait être fichier séparé)
- 🟡 Données cluster mockées (3D viz montre faux cluster)
- 🟡 JavaScript assume endpoints non définis (`/api/swarm/status`)

## dashboard/launcher.py (~30 LOC) — ✅ Bon
Launcher CLI simple (argparse) pour choisir mode web/CLI.
