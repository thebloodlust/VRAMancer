# Audit — core/transport_factory.py

## Résumé
Factory unifiée sélectionnant le transport optimal selon la localité (SAME_GPU / SAME_NODE / SAME_RACK / REMOTE). Gère les singletons TransferManager, FastHandle et VTP.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~350 |
| **Qualité** | ✅ Bon |
| **Sécurité** | 🟡 Risque moyen |
| **Performance** | 🟡 Lock sur chaque accès singleton |

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🟡 MOYENNE | **Détermination localité simpliste** : hardcodé via `VRM_SAME_RACK_NODES` env var |
| 🟡 MOYENNE | **Race condition double-check locking** : multiples TransferManagers possibles |
| 🟡 MOYENNE | VTP peut échouer silencieusement sans fallback |
| 🟡 MOYENNE | Pas de timeout sur envoi VTP/FastHandle |
| 🟢 BASSE | `reset_transport_factory()` ne ferme pas FastHandle |
| 🟢 BASSE | `device_str` property définie mais non utilisée |
