# Audit — core/compressor.py

## Résumé
Compression multi-stratégies : codecs byte-level (zstd/lz4/gzip) et quantification (INT8/INT4) pour spill NVMe et transferts réseau.

| Critère | Évaluation |
|---------|------------|
| **LOC** | ~400 |
| **Qualité** | ✅ Bon |
| **Sécurité** | 🟡 Risque moyen |
| **Performance** | 🟡 Auto-détection lente |

## Problèmes détectés
| Sévérité | Description |
|----------|-------------|
| 🟡 MOYENNE | **Pas de versioning de format** : données compressées sans header codec/version |
| 🟡 MOYENNE | **DoS via décompression** : fichiers craftés peuvent causer allocation mémoire illimitée |
| 🟡 MOYENNE | `decompress_bytes()` essaie tous les codecs séquentiellement |
| 🟡 MOYENNE | INT4 quantification simpliste (min/max naïf) |
| 🟡 MOYENNE | Pas de checksum/signature sur données compressées |
| 🟢 BASSE | API legacy `compress()` retourne taille estimée, pas compression réelle |
