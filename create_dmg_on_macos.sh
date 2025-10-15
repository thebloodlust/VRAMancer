#!/bin/bash
# Script à exécuter sur macOS pour créer le .dmg final

set -e

# Détection automatique de la version
if [ -f "VRAMancer-1.1.0-macOS.tar.gz" ]; then
    ARCHIVE="VRAMancer-1.1.0-macOS.tar.gz"
    VERSION="1.1.0"
elif [ -f "VRAMancer-1.0.0-macOS.tar.gz" ]; then
    ARCHIVE="VRAMancer-1.0.0-macOS.tar.gz"
    VERSION="1.0.0"
else
    # Chercher n'importe quelle version
    ARCHIVE=$(ls VRAMancer-*-macOS.tar.gz 2>/dev/null | head -1)
    if [ -z "$ARCHIVE" ]; then
        echo "❌ Aucune archive VRAMancer-*-macOS.tar.gz trouvée"
        echo ""
        echo "📍 Répertoire actuel : $(pwd)"
        echo "📁 Fichiers .tar.gz disponibles :"
        ls -1 *.tar.gz 2>/dev/null || echo "   Aucun fichier .tar.gz trouvé"
        echo ""
        echo "💡 Solutions :"
        echo "   1. Télécharger VRAMancer-1.1.0-macOS.tar.gz depuis :"
        echo "      https://github.com/thebloodlust/VRAMancer/releases"
        echo "   2. Placer l'archive dans ce répertoire"
        echo "   3. Relancer : ./create_dmg_on_macos.sh"
        exit 1
    fi
    VERSION=$(echo "$ARCHIVE" | sed 's/VRAMancer-\(.*\)-macOS.tar.gz/\1/')
fi

BUILD_DIR="build_macos"
echo "📦 Version détectée : $VERSION"
echo "📦 Archive : $ARCHIVE"

echo "📦 Extraction de l'archive..."
tar -xzf "$ARCHIVE"

echo "💿 Création du DMG..."
DMG_DIR="${BUILD_DIR}/dmg"
FINAL_DMG="VRAMancer-${VERSION}-macOS.dmg"

hdiutil create -srcfolder "${DMG_DIR}" -volname "VRAMancer" \
    -fs HFS+ -fsargs "-c c=64,a=16,e=16" -format UDZO \
    -imagekey zlib-level=9 "${FINAL_DMG}"

echo "✅ DMG créé: ${FINAL_DMG}"
echo "📦 Taille: $(du -h "${FINAL_DMG}" | cut -f1)"
