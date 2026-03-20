# Audit — core/model_splitter.py

## Résumé
Charge des modèles HuggingFace et les divise sur plusieurs GPU proportionnellement à la VRAM, avec placement DP-optimal optionnel via profiler.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~450 |
| **Qualité** | ✅ Bon |
| **Sécurité** | 🟡 Risque moyen |
| **Performance** | 🟡 Charge modèle complet en mémoire |

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🟡 MOYENNE | **Modèle chargé entièrement en DRAM** avant split — risque OOM |
| 🟡 MOYENNE | Pattern matching fragile (`_LAYER_CANDIDATES` hardcodé) |
| 🟡 MOYENNE | `AutoModel.from_pretrained()` sans validation de checksum |
| 🟡 MOYENNE | VLM handling incomplet (seulement vision_model + language_model) |
| 🟡 MOYENNE | pynvml initialisé à chaque appel |
| 🟢 BASSE | MoE désactive profiler sans explication |
