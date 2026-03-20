# Audit — core/block_router.py

## Résumé
Routeur VRAM-aware pour blocs de modèle. Route intelligemment vers GPU/CPU/NVMe/remote selon disponibilité en temps réel.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~550 |
| **Qualité** | ⚠️ Mixte |
| **Sécurité** | 🟡 Risque moyen |
| **Performance** | 🔴 I/O réseau bloquant |

## Classes/Fonctions clés
- `BlockRouter` : `route()`, `register_remote_node()`, `_find_gpu_for_block()`, `_nvme_available()`, `_exec_on_device()`
- `RemoteExecutor` : exécution distante avec safetensors zero-copy, vérification HMAC, Wake-on-Inference

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🔴 HAUTE | **"default_insecure_token" hardcodé** comme fallback |
| 🔴 HAUTE | **Remote transport sans TLS** : seulement HMAC, pas de chiffrement |
| 🟡 MOYENNE | Race condition sur `os.makedirs()` dans `_nvme_available()` |
| 🟡 MOYENNE | `RemoteExecutor.forward()` 150+ lignes — mélange de responsabilités |
| 🟡 MOYENNE | I/O réseau synchrone bloque les threads GPU |
| 🟡 MOYENNE | Pas de connection pooling (nouveau socket par transfert) |
| 🟢 BASSE | Commentaires en français mélangés ("dégradé", "Niveau 2") |
