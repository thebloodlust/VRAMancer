#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
#  VRAMancer — Quick Installer (Linux / macOS)
#  Trouve Python, puis délègue à install.py qui gère tout.
#
#  Usage:
#    bash Install.sh              # Standard (venv + PyTorch auto)
#    bash Install.sh --full       # Tout (GUI, tracing, etc.)
#    bash Install.sh --lite       # CLI minimal
#    bash Install.sh --docker     # Build + lance Docker Compose
#    bash Install.sh --dev        # Mode développement
#    curl -sSL https://raw.githubusercontent.com/…/Install.sh | bash
# ═══════════════════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Trouver Python 3.10+ ─────────────────────────────────────────────
find_python() {
    for cmd in python3.12 python3.11 python3.10 python3 python; do
        if command -v "$cmd" >/dev/null 2>&1; then
            local ver
            ver=$("$cmd" -c "import sys; v=sys.version_info; print(f'{v.major}.{v.minor}')" 2>/dev/null || true)
            local major minor
            major=$(echo "$ver" | cut -d. -f1)
            minor=$(echo "$ver" | cut -d. -f2)
            if [ "${major:-0}" -ge 3 ] && [ "${minor:-0}" -ge 10 ]; then
                echo "$cmd"
                return 0
            fi
        fi
    done
    return 1
}

# ── Installer Python si absent ────────────────────────────────────────
install_python_hint() {
    local os_type
    os_type="$(uname -s)"
    echo ""
    echo "  Python 3.10+ est requis mais n'a pas été trouvé."
    echo ""
    case "$os_type" in
        Linux*)
            if command -v apt >/dev/null 2>&1; then
                echo "  Installer avec apt:"
                echo "    sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
            elif command -v dnf >/dev/null 2>&1; then
                echo "  Installer avec dnf:"
                echo "    sudo dnf install -y python3 python3-pip"
            elif command -v pacman >/dev/null 2>&1; then
                echo "  Installer avec pacman:"
                echo "    sudo pacman -S python python-pip"
            fi
            ;;
        Darwin*)
            echo "  Installer avec Homebrew:"
            echo "    brew install python@3.12"
            echo ""
            echo "  Ou télécharger depuis: https://www.python.org/downloads/"
            ;;
    esac
    echo ""
}

# ── Main ──────────────────────────────────────────────────────────────
PYTHON=$(find_python) || true

if [ -z "$PYTHON" ]; then
    install_python_hint
    exit 1
fi

echo ""
echo "  Utilisation de: $PYTHON ($($PYTHON --version 2>&1))"
echo ""

# Déléguer à install.py
exec "$PYTHON" "$SCRIPT_DIR/install.py" "$@"
