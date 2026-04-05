#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  VRAMancer — Build macOS DMG Installer
#  Run this ON macOS to create VRAMancer-Installer.dmg
#
#  Usage:  ./scripts/build_dmg.sh
#  Output: dist/VRAMancer-Installer.dmg
# ═══════════════════════════════════════════════════════════
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERSION=$(python3 -c "
import re
with open('$REPO_DIR/core/__init__.py') as f:
    m = re.search(r'__version__\s*=\s*[\"'\''](.*?)[\"'\'']', f.read())
    print(m.group(1) if m else '1.5.0')
" 2>/dev/null || echo "1.5.0")

DMG_NAME="VRAMancer-${VERSION}-macOS"
BUILD_DIR="$REPO_DIR/dist/dmg_build"
DMG_OUTPUT="$REPO_DIR/dist/${DMG_NAME}.dmg"

echo "╔══════════════════════════════════════════════════════════╗"
echo "║            VRAMancer — DMG Builder                       ║"
echo "║  Version: $VERSION                                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Nettoyage ─────────────────────────────────────────────
echo "▸ Préparation..."
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
mkdir -p "$REPO_DIR/dist"

# ── Copier les fichiers nécessaires ──────────────────────
echo "▸ Copie des fichiers..."

# Le repo complet (sans les fichiers lourds)
rsync -a --progress \
    --exclude='.venv/' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.eggs/' \
    --exclude='*.egg-info/' \
    --exclude='dist/' \
    --exclude='build/' \
    --exclude='*.so' \
    --exclude='rust_core/target/' \
    --exclude='node_modules/' \
    --exclude='.tmp/' \
    "$REPO_DIR/" "$BUILD_DIR/VRAMancer/"

# L'installeur double-cliquable au premier niveau
cp "$REPO_DIR/Install VRAMancer.command" "$BUILD_DIR/Install VRAMancer.command"
chmod +x "$BUILD_DIR/Install VRAMancer.command"

# README pour le DMG
cat > "$BUILD_DIR/LISEZMOI.txt" << 'EOF'
╔══════════════════════════════════════════════════════════╗
║            VRAMancer — Installation macOS                ║
╚══════════════════════════════════════════════════════════╝

1. Double-cliquez sur "Install VRAMancer.command"
2. C'est tout ! Le script installera automatiquement :
   - Homebrew (si absent)
   - Python 3.12 (si absent)
   - PyTorch avec support MPS (Apple Silicon)
   - VRAMancer et toutes ses dépendances

Après installation :
   source ~/VRAMancer/.venv/bin/activate
   vramancer serve

API : http://localhost:5030
─────────────────────────────────────────────────────────
https://github.com/thebloodlust/VRAMancer
EOF

# ── Créer le DMG ─────────────────────────────────────────
echo "▸ Création du DMG..."

# Supprimer l'ancien si existant
rm -f "$DMG_OUTPUT"

# Créer le DMG avec hdiutil
hdiutil create \
    -volname "VRAMancer Installer" \
    -srcfolder "$BUILD_DIR" \
    -ov \
    -format UDZO \
    -imagekey zlib-level=9 \
    "$DMG_OUTPUT"

# ── Nettoyage ─────────────────────────────────────────────
echo "▸ Nettoyage..."
rm -rf "$BUILD_DIR"

# ── Résultat ──────────────────────────────────────────────
DMG_SIZE=$(du -h "$DMG_OUTPUT" | cut -f1)
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  DMG créé : $DMG_OUTPUT"
echo "  Taille   : $DMG_SIZE"
echo "═══════════════════════════════════════════════════════"
echo ""
echo "  Pour distribuer : copiez le .dmg sur une clé USB"
echo "  ou partagez-le via AirDrop / réseau local."
echo ""
