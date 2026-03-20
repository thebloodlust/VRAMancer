#!/bin/bash
# VRAMancer macOS/Linux validation wrapper

# Définition du chemin absolu
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Exécution du script Python
python3 "$DIR/scripts/validate_platform.py" "$@"
