# Audit — core/stream_manager.py

## Résumé
Gestionnaire de streaming adaptatif pour inférence multi-GPU avec prefetch, swap et politiques d'éviction.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~500 |
| **Qualité** | ✅ Bon |
| **Sécurité** | 🟡 Risque faible |
| **Performance** | 🟡 Contention de lock |

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🔴 HAUTE | **ThreadPoolExecutor jamais shutdown** — fuite de threads |
| 🟡 MOYENNE | Lock tenu pendant allocation scheduler — bloque autres threads |
| 🟡 MOYENNE | `_estimate_layer_size()` hardcodé à 100MB par défaut |
| 🟡 MOYENNE | Éviction synchrone dans boucle monitor — peut bloquer l'inférence |
| 🟢 BASSE | Emoji "🧠 Anticipatory Brain" dans logs — non professionnel |
