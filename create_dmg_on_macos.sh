#!/bin/bash
# Script à exécuter sur macOS pour créer le .dmg final

set -e

ARCHIVE="VRAMancer-1.0.0-macOS.tar.gz"
BUILD_DIR="build_macos"

if [ ! -f "$ARCHIVE" ]; then
    echo "❌ Archive $ARCHIVE non trouvée"
    exit 1
fi

echo "📦 Extraction de l'archive..."
tar -xzf "$ARCHIVE"

echo "💿 Création du DMG..."
DMG_DIR="${BUILD_DIR}/dmg"
FINAL_DMG="VRAMancer-1.0.0-macOS.dmg"

hdiutil create -srcfolder "${DMG_DIR}" -volname "VRAMancer" \
    -fs HFS+ -fsargs "-c c=64,a=16,e=16" -format UDZO \
    -imagekey zlib-level=9 "${FINAL_DMG}"

echo "✅ DMG créé: ${FINAL_DMG}"
echo "📦 Taille: $(du -h "${FINAL_DMG}" | cut -f1)"
