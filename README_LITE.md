# VRAMancer Lite (CLI only)

Version ultra-légère, sans dashboard graphique ni modules premium.

## Installation

```bash
# Extraction du package Lite
make -f Makefile.lite lite
cd dist_lite
pip install -r requirements-lite.txt
```

## Utilisation

```bash
python vramancer.py --help
python cli/dashboard_cli.py --help
```

## Modules inclus
- CLI (dashboard_cli)
- Orchestrateur (vramancer.py, launcher.py)
- Core, utils, scripts

## Modules exclus
- dashboard/ (graphique)
- premium/
- marketplace/
- onboarding vidéo

## Pour la version complète
Consultez le README principal.
