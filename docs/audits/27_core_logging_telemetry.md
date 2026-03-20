# Audit — core/logger.py, telemetry.py, tracing.py

## logger.py (~75 LOC) — ✅ Bon
Logger centralisé avec sortie JSON optionnelle, formatage Rich et logging fichier.
- ⚠️ Chemin fichier log `VRAMANCER_LOG_FILE` non assaini
- ⚠️ I/O fichier non bufferisé

## telemetry.py (~90 LOC) — ✅ Bon
Format binaire/texte compact pour télémétrie cluster. Header 17B + ID variable NUL-terminated.
- ⚠️ Pas d'authentification/intégrité sur les paquets
- ⚠️ `decode_stream()` sans limite de taille max — boucle infinie possible

## tracing.py (~85 LOC) — ✅ Excellent
Intégration OpenTelemetry optionnelle avec no-op fallback.
- ⚠️ Pas de validation certificat OTLP (MITM vulnérable)
- ⚠️ Pas de sampling — tous les spans exportés si activé
- ⚠️ `start_tracing()` ne shutdown jamais le provider
