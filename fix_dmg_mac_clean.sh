#!/bin/bash
# Installation DMG macOS - Méthode dossier propre
# Contourne les conflits en extrayant dans un dossier temporaire vide

set -e

echo "🧹 Préparation d'un dossier propre..."
cd ~/Downloads
rm -rf vramancer_install_clean
mkdir vramancer_install_clean
cd vramancer_install_clean

echo "📦 Extraction de l'archive..."
tar -xzf ~/Downloads/VRAMancer-main/VRAMancer-1.1.0-macOS.tar.gz

echo "💿 Création du DMG..."
hdiutil create -volname "VRAMancer" -srcfolder build_macos/dmg -ov -format UDZO VRAMancer-1.1.0-macOS.dmg

echo "📂 Déplacement du DMG..."
mv VRAMancer-1.1.0-macOS.dmg ~/Downloads/

echo ""
echo "✅ DMG créé : ~/Downloads/VRAMancer-1.1.0-macOS.dmg"
echo "📂 Ouverture..."
open ~/Downloads/VRAMancer-1.1.0-macOS.dmg

echo ""
echo "🎉 INSTALLATION :"
echo "   1. Glisser VRAMancer.app dans Applications"
echo "   2. Ctrl+clic > Ouvrir (pour Gatekeeper)"
echo "   3. Lancer depuis Applications ou Terminal"
echo ""
echo "🗑️  Nettoyage du dossier temporaire..."
cd ~/Downloads
rm -rf vramancer_install_clean

echo "✅ Installation terminée !"
