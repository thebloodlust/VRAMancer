# Audit — vramancer/ (Entry point + CLI)

## vramancer/__init__.py — ✅ Vide (minimal)
## vramancer/__main__.py (~10 LOC) — ✅ Bon
Délègue à `vramancer.main.main()`.

## vramancer/main.py (~550 LOC) — ⚠️ Mixte
Point d'entrée CLI unifié avec sous-commandes : serve, generate, benchmark, discover, split, hub, health, auth, status, version.
- ✅ Structure argparse bien organisée
- ✅ Fallback gracieux si `rich` absent
- 🟡 **Fichier trop long** (~550 lignes) — devrait être splitté
- 🟡 Version hardcodée '0.2.4' vs `core.__version__`
- 🟡 Détection auto GPU dupliquée dans 3 commandes
- 🟡 `_cmd_serve()` fait le pre-load avant `install_security()`
- 🟡 Utilise `print()` au lieu du logger

## vramancer/cli/swarm_cli.py (~25 LOC) — ✅ Bon
Génération d'identités Swarm Ledger et API keys.

## vramancer/cli/dashboard_cli.py (~15 LOC) — ✅ Bon
Shim de compatibilité vers `dashboard.cli_dashboard`.

## vramancer/cli/telemetry_cli.py (~35 LOC) — ⚠️ Mixte
Décodage stream télémétrie binaire.
- 🟡 HTTP non chiffré pour télémétrie
- 🟡 Pas de pagination (tout le stream en mémoire)
- 🟡 Port hardcodé `localhost:5010`
