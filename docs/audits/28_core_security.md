# Audit — core/auth_strong.py, security/, security/__init__.py

## auth_strong.py (~130 LOC) — ✅ Production-ready
Authentification forte : pbkdf2 (120k itérations), JWT HS256, rotation refresh token, RBAC.
- ✅ `hmac.compare_digest()` contre timing attacks
- ✅ Sel unique par mot de passe (8 bytes hex)
- 🔴 **Admin par défaut** `admin/random_password` créé en dev mode
- 🟡 Pas de MFA, pas de politique de mot de passe
- 🟡 Base de données en mémoire (perdue au restart)

## security/startup_checks.py (~65 LOC) — ✅ Bon
Validation sécurité pré-démarrage. Crash si config insécurisée en production.
- ⚠️ `authenticate()` est un stub (retourne toujours True)
- ⚠️ Pas de validation d'entropie du secret

## security/remote_access.py (~180 LOC) — 🔴 Non sécurisé
Panneau de contrôle web avec login et session Flask.
- 🔴 **CRITIQUE : Credentials dev en clair dans les logs**
- 🔴 Pas de protection CSRF
- 🔴 MFA est un stub (non fonctionnel)
- 🟡 Pas de rate limiting sur `/login`

## security/__init__.py (~500+ LOC) — ✅ Solide
Middleware Flask unifié : token auth, HMAC, RBAC, CORS, rate limiting, read-only.
- 🔴 `VRM_TEST_ALL_OPEN` bypass toute la sécurité
- 🟡 Rate limiting in-process seulement (pas distribué)
- 🟡 Headers CSP très permissifs (`unsafe-inline`, `unsafe-eval`)
