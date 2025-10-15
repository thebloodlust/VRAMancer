#!/bin/bash
# Script de nettoyage et création DMG pour macOS
# Résout les conflits de fichiers/dossiers lors de l'extraction

set -e

echo "🧹 Nettoyage complet forcé..."
cd ~/Downloads/VRAMancer-main

# Supprimer TOUT avec force (nécessite sudo pour les permissions)
sudo rm -rf build_macos dmg VRAMancer-1.1.0-macOS.dmg 2>/dev/null || true
find . -name "VRAMancer.app" -exec sudo rm -rf {} + 2>/dev/null || true

echo "📦 Extraction de l'archive..."
tar -xzf VRAMancer-1.1.0-macOS.tar.gz

echo "💿 Création du DMG..."
hdiutil create -volname "VRAMancer" -srcfolder build_macos/dmg -ov -format UDZO VRAMancer-1.1.0-macOS.dmg

echo ""
echo "✅ DMG créé : ~/Downloads/VRAMancer-main/VRAMancer-1.1.0-macOS.dmg"
echo "📂 Ouverture..."
open VRAMancer-1.1.0-macOS.dmg

echo ""
echo "🎉 INSTALLATION :"
echo "   1. Glisser VRAMancer.app dans Applications"
echo "   2. Ctrl+clic > Ouvrir (pour Gatekeeper)"
echo "   3. python3 launch_vramancer.py"
echo ""
echo "📖 Voir LISEZMOI_MACOS.txt pour plus d'aide"
