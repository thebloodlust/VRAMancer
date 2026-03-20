# Audit — core/utils.py

## Résumé
Détection de device GPU-aware, fallbacks tokenization et utilitaires. Détecte backend (CUDA/ROCm/MPS/TPU), énumère devices, fournit proxy tokenizer et sérialisation tensor.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~550 |
| **Qualité** | ✅ Excellent |
| **Sécurité** | 🟡 Risque moyen |
| **Performance** | 🟡 Pas de cache |

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🟡 MOYENNE | `detect_backend()` appelé sans cache — devrait être mémorisé |
| 🟡 MOYENNE | `BasicTokenizer` vocab sans limite de taille — fuite mémoire potentielle |
| 🟡 MOYENNE | Mapping device logique/physique jamais invalidé |
| 🟡 MOYENNE | `VRM_FORCE_BASIC_TOKENIZER` permet attaque de downgrade |
| 🟢 BASSE | Format sérialisation tensor sans header/magic |
