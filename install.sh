#!/usr/bin/env bash
# VRAMancer — installateur une-ligne (S4).
#
#   curl -fsSL https://raw.githubusercontent.com/thebloodlust/VRAMancer/main/install.sh | bash
#
# Bootstrap mince : vérifie les prérequis, récupère le repo, délègue au
# `install.py` éprouvé (détecte GPU/CUDA, venv isolé, bon wheel torch, Rust),
# puis pose la commande `vramancer` dans ~/.local/bin. Idempotent.
#
# Variables d'env :
#   VRAMANCER_HOME    répertoire d'install (défaut: ~/.vramancer)
#   VRAMANCER_MODE    mode install.py: standard|full|lite|dev (défaut: standard)
#   VRAMANCER_REPO    URL git (défaut: dépôt officiel)
#   VRAMANCER_DRY_RUN 1 = ne lance pas install.py (test du bootstrap)
set -euo pipefail

REPO="${VRAMANCER_REPO:-https://github.com/thebloodlust/VRAMancer.git}"
HOME_DIR="${VRAMANCER_HOME:-$HOME/.vramancer}"
MODE="${VRAMANCER_MODE:-standard}"
BIN_DIR="$HOME/.local/bin"
DRY="${VRAMANCER_DRY_RUN:-0}"

c() { [ -t 1 ] && printf "\033[%sm%s\033[0m" "$1" "$2" || printf "%s" "$2"; }
info() { printf "  %s %s\n" "$(c 32 ✓)" "$1"; }
warn() { printf "  %s %s\n" "$(c 33 ⚠)" "$1"; }
step() { printf "\n%s\n" "$(c '1;36' "▸ $1")"; }
die()  { printf "  %s %s\n" "$(c 31 ✗)" "$1" >&2; exit 1; }

step "VRAMancer — installateur une-ligne"

# 1) Prérequis
command -v python3 >/dev/null 2>&1 || die "python3 introuvable (≥ 3.10 requis)."
PYV=$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])')
python3 -c 'import sys;sys.exit(0 if sys.version_info[:2]>=(3,10) else 1)' \
  || die "Python $PYV trop ancien (≥ 3.10 requis)."
info "Python $PYV"
if command -v nvidia-smi >/dev/null 2>&1; then
  info "Driver NVIDIA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
else
  warn "nvidia-smi absent — install CPU/ROCm (install.py choisira le bon wheel)."
fi

# 2) Récupérer les sources (checkout local si on lance depuis le repo, sinon clone/pull)
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || true)"
if [ -n "$SELF_DIR" ] && [ -f "$SELF_DIR/install.py" ]; then
  SRC="$SELF_DIR"
  info "Sources locales détectées: $SRC"
else
  command -v git >/dev/null 2>&1 || die "git introuvable (nécessaire pour cloner)."
  if [ -d "$HOME_DIR/.git" ]; then
    step "Mise à jour du dépôt ($HOME_DIR)"
    git -C "$HOME_DIR" pull --ff-only --quiet && info "À jour"
  else
    step "Clonage de $REPO"
    git clone --depth 1 "$REPO" "$HOME_DIR" --quiet && info "Cloné dans $HOME_DIR"
  fi
  SRC="$HOME_DIR"
fi

# 3) Déléguer au Universal Auto-Installer
step "Installation (install.py, mode=$MODE)"
MODE_FLAG=""
case "$MODE" in
  full) MODE_FLAG="--full" ;;
  lite) MODE_FLAG="--lite" ;;
  dev)  MODE_FLAG="--dev" ;;
  standard) MODE_FLAG="" ;;
  *) warn "Mode inconnu '$MODE' → standard" ;;
esac
if [ "$DRY" = "1" ]; then
  warn "DRY_RUN: install.py non lancé (test du bootstrap)."
else
  ( cd "$SRC" && python3 install.py $MODE_FLAG )
fi

# 4) Poser la commande `vramancer` dans ~/.local/bin
step "Commande vramancer"
VENV_PY="$SRC/.venv/bin/python"
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/vramancer" <<EOF
#!/usr/bin/env bash
# Wrapper VRAMancer (généré par install.sh)
exec "$VENV_PY" -m vramancer "\$@"
EOF
chmod +x "$BIN_DIR/vramancer"
info "Installée: $BIN_DIR/vramancer"

# 5) PATH + étapes suivantes
case ":$PATH:" in
  *":$BIN_DIR:"*) : ;;
  *) warn "$BIN_DIR n'est pas dans le PATH. Ajoute :  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac

step "Terminé"
printf "  Essaie :  %s\n" "$(c 36 'vramancer quickstart code-assistant')"
printf "  Puis   :  %s\n" "$(c 36 'vramancer quickstart code-assistant --run')"
