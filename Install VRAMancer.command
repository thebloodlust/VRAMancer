#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  VRAMancer — Installeur macOS One-Click
#  Double-cliquez sur ce fichier pour installer VRAMancer
# ═══════════════════════════════════════════════════════════
set -e

# Couleurs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

INSTALL_DIR="$HOME/VRAMancer"
REPO_URL="https://github.com/thebloodlust/VRAMancer.git"

info()  { echo -e "  ${GREEN}✓${NC} $1"; }
warn()  { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail()  { echo -e "  ${RED}✗${NC} $1"; }
step()  { echo -e "\n${BOLD}${CYAN}▸ $1${NC}"; }

cat << 'BANNER'

╔══════════════════════════════════════════════════════════╗
║            VRAMancer — macOS Installer                   ║
║     Multi-GPU LLM Inference for Heterogeneous Hardware   ║
╚══════════════════════════════════════════════════════════╝

BANNER

# ── Vérif macOS ───────────────────────────────────────────
step "Vérification du système"

if [[ "$(uname)" != "Darwin" ]]; then
    fail "Ce script est pour macOS uniquement."
    exit 1
fi

ARCH=$(uname -m)
MACOS_VER=$(sw_vers -productVersion)
info "macOS $MACOS_VER ($ARCH)"

if [[ "$ARCH" == "arm64" ]]; then
    info "Apple Silicon détecté — support MPS (Metal Performance Shaders)"
else
    warn "Intel Mac détecté — mode CPU uniquement (pas de MPS)"
fi

# ── Xcode Command Line Tools ─────────────────────────────
step "Xcode Command Line Tools"

if xcode-select -p &>/dev/null; then
    info "Déjà installé"
else
    info "Installation des Xcode CLI tools..."
    xcode-select --install 2>/dev/null || true
    echo ""
    echo "  Une fenêtre va s'ouvrir pour installer les outils Xcode."
    echo "  Cliquez 'Installer' puis relancez ce script après."
    echo ""
    read -p "  Appuyez sur Entrée une fois l'installation terminée..."
    if ! xcode-select -p &>/dev/null; then
        fail "Xcode CLI tools non installés. Réessayez."
        exit 1
    fi
fi

# ── Homebrew ──────────────────────────────────────────────
step "Homebrew"

if command -v brew &>/dev/null; then
    info "Homebrew trouvé: $(brew --prefix)"
else
    info "Installation de Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Ajouter au PATH pour cette session
    if [[ "$ARCH" == "arm64" ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    info "Homebrew installé"
fi

# ── Python 3.12 ──────────────────────────────────────────
step "Python"

# Chercher Python 3.10+
PYTHON_CMD=""
for py in python3.12 python3.11 python3.10 python3; do
    if command -v "$py" &>/dev/null; then
        PY_VER=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [[ "$PY_MAJOR" -ge 3 ]] && [[ "$PY_MINOR" -ge 10 ]]; then
            PYTHON_CMD="$py"
            break
        fi
    fi
done

if [[ -n "$PYTHON_CMD" ]]; then
    info "Python trouvé: $PYTHON_CMD ($PY_VER)"
else
    info "Installation de Python 3.12 via Homebrew..."
    brew install python@3.12
    PYTHON_CMD="python3.12"
    PY_VER="3.12"
    info "Python 3.12 installé"
fi

# ── Git ───────────────────────────────────────────────────
step "Git"

if command -v git &>/dev/null; then
    info "Git trouvé: $(git --version | head -1)"
else
    info "Installation de Git via Homebrew..."
    brew install git
    info "Git installé"
fi

# ── Clone / Update VRAMancer ─────────────────────────────
step "VRAMancer"

# Si on est lancé depuis le repo lui-même
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$SCRIPT_DIR/install.py" ]] && [[ -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    info "Lancé depuis le repo VRAMancer"
    INSTALL_DIR="$SCRIPT_DIR"
elif [[ -d "$INSTALL_DIR" ]] && [[ -f "$INSTALL_DIR/install.py" ]]; then
    info "VRAMancer déjà présent dans $INSTALL_DIR"
    cd "$INSTALL_DIR"
    info "Mise à jour depuis GitHub..."
    git pull --ff-only 2>/dev/null || warn "git pull échoué (modifications locales?)"
else
    info "Clonage de VRAMancer..."
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
info "Répertoire: $INSTALL_DIR"

# ── Lancer install.py ────────────────────────────────────
step "Installation automatique"
info "Lancement de install.py..."
echo ""

"$PYTHON_CMD" install.py --yes

# ── Créer un alias global ────────────────────────────────
step "Configuration du PATH"

VENV_BIN="$INSTALL_DIR/.venv/bin"
SHELL_RC=""

if [[ -f "$HOME/.zshrc" ]]; then
    SHELL_RC="$HOME/.zshrc"
elif [[ -f "$HOME/.bash_profile" ]]; then
    SHELL_RC="$HOME/.bash_profile"
elif [[ -f "$HOME/.bashrc" ]]; then
    SHELL_RC="$HOME/.bashrc"
fi

if [[ -n "$SHELL_RC" ]]; then
    # Vérifier si déjà ajouté
    if ! grep -q "VRAMancer" "$SHELL_RC" 2>/dev/null; then
        echo "" >> "$SHELL_RC"
        echo "# VRAMancer" >> "$SHELL_RC"
        echo "export PATH=\"$VENV_BIN:\$PATH\"" >> "$SHELL_RC"
        info "PATH ajouté à $SHELL_RC"
    else
        info "PATH déjà configuré dans $SHELL_RC"
    fi
fi

# ── Résumé ────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  Installation terminée !${NC}"
echo -e "${BOLD}${GREEN}═══════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}Dossier${NC}:  $INSTALL_DIR"
echo -e "  ${BOLD}Python${NC}:   $PYTHON_CMD ($PY_VER)"
echo -e "  ${BOLD}venv${NC}:     $INSTALL_DIR/.venv/"
echo ""
echo -e "  ${BOLD}Démarrage rapide${NC}:"
echo -e "    ${CYAN}source $INSTALL_DIR/.venv/bin/activate${NC}"
echo -e "    ${CYAN}vramancer serve${NC}"
echo ""
echo -e "  ${BOLD}Ou directement${NC}:"
echo -e "    ${CYAN}$VENV_BIN/vramancer serve${NC}"
echo ""
echo -e "  ${BOLD}API${NC}: http://localhost:5030"
echo -e "  ${BOLD}Health${NC}: curl http://localhost:5030/health"
echo ""

# Garder le terminal ouvert
read -p "Appuyez sur Entrée pour fermer..."
