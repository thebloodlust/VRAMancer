# Audit — core/config.py

## Résumé
Configuration centralisée avec résolution hiérarchique (defaults → YAML → env vars), support multi-OS, hot-reload et validation.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~500 |
| **Qualité** | ✅ Bon |
| **Sécurité** | 🟡 Risque moyen |
| **Performance** | ✅ Correct |

## Fonctions clés
- `get_config()` — singleton thread-safe
- `reload_config()` — hot reload
- `_load_yaml()`, `_coerce()`, `_env_overrides()`, `_validate()`

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🟡 MOYENNE | `api_token` default None — devrait être requis en production |
| 🟡 MOYENNE | `hmac_secret` peut être None en production |
| 🟡 MOYENNE | Pas de vérification de permissions fichier config (lisible par tous) |
| 🟡 MOYENNE | YAML accepte clés arbitraires : erreurs de frappe ignorées silencieusement |
| 🟢 BASSE | Coercion de type ne gère pas `int("foo")` proprement |
