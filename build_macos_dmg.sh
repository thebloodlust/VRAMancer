#!/bin/bash
#############################################################
# VRAMancer macOS .dmg Builder
# Crée un installateur .dmg pour macOS avec l'application bundle
#############################################################

set -e

echo "🍎 Construction de l'installateur VRAMancer.dmg pour macOS"
echo "=========================================================="
echo ""

# Variables
APP_NAME="VRAMancer"
VERSION="1.1.0"
DMG_NAME="${APP_NAME}-${VERSION}-macOS"
BUILD_DIR="build_macos"
APP_BUNDLE="${BUILD_DIR}/${APP_NAME}.app"
DMG_DIR="${BUILD_DIR}/dmg"
FINAL_DMG="${DMG_NAME}.dmg"

# Nettoyer les builds précédents
echo "🧹 Nettoyage des builds précédents..."
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
mkdir -p "${DMG_DIR}"

# Créer la structure de l'app bundle
echo "📦 Création de la structure ${APP_NAME}.app..."
mkdir -p "${APP_BUNDLE}/Contents/MacOS"
mkdir -p "${APP_BUNDLE}/Contents/Resources"
mkdir -p "${APP_BUNDLE}/Contents/Frameworks"

# Copier les modules Python
echo "📋 Copie des modules Python..."
cp -r core "${APP_BUNDLE}/Contents/MacOS/"
cp -r dashboard "${APP_BUNDLE}/Contents/MacOS/"
cp -r cli "${APP_BUNDLE}/Contents/MacOS/"
cp -r utils "${APP_BUNDLE}/Contents/MacOS/"
cp -r config "${APP_BUNDLE}/Contents/MacOS/"
cp -r vramancer "${APP_BUNDLE}/Contents/MacOS/" 2>/dev/null || echo "⚠️  Dossier vramancer non trouvé"

# Copier les interfaces mobiles/edge (si présentes)
[ -d "mobile" ] && cp -r mobile "${APP_BUNDLE}/Contents/MacOS/" || echo "⚠️  mobile/ non trouvé"
[ -d "edge" ] && cp -r edge "${APP_BUNDLE}/Contents/MacOS/" || echo "⚠️  edge/ non trouvé"

# Copier les scripts principaux
echo "📋 Copie des scripts principaux..."
cp api_simple.py "${APP_BUNDLE}/Contents/MacOS/"
cp systray_vramancer.py "${APP_BUNDLE}/Contents/MacOS/"
cp launcher_auto.py "${APP_BUNDLE}/Contents/MacOS/"
cp vrm_start.sh "${APP_BUNDLE}/Contents/MacOS/"
cp launch_vramancer.py "${APP_BUNDLE}/Contents/MacOS/" 2>/dev/null || echo "⚠️  launch_vramancer.py non trouvé"

# Copier les dashboards spécifiques
for script in dashboard_*.py debug_*.py; do
    [ -f "$script" ] && cp "$script" "${APP_BUNDLE}/Contents/MacOS/" || true
done

# Copier les fichiers de configuration
echo "📋 Copie des fichiers de configuration..."
cp requirements.txt "${APP_BUNDLE}/Contents/MacOS/"
cp requirements-lite.txt "${APP_BUNDLE}/Contents/MacOS/" 2>/dev/null || true
cp config.yaml "${APP_BUNDLE}/Contents/MacOS/" 2>/dev/null || echo "⚠️  config.yaml non trouvé"
cp setup.py "${APP_BUNDLE}/Contents/MacOS/"
cp pyproject.toml "${APP_BUNDLE}/Contents/MacOS/"
cp setup.cfg "${APP_BUNDLE}/Contents/MacOS/" 2>/dev/null || true

# Copier la documentation
echo "📚 Copie de la documentation..."
cp README.md "${APP_BUNDLE}/Contents/MacOS/"
cp MANUEL_FR.md "${APP_BUNDLE}/Contents/MacOS/" 2>/dev/null || true
cp MANUAL_EN.md "${APP_BUNDLE}/Contents/MacOS/" 2>/dev/null || true
cp CHANGELOG.md "${APP_BUNDLE}/Contents/MacOS/" 2>/dev/null || true

# Copier les exemples
[ -d "examples" ] && cp -r examples "${APP_BUNDLE}/Contents/MacOS/" || echo "⚠️  examples/ non trouvé"

# Copier l'icône
echo "🎨 Copie de l'icône..."
if [ -f "vramancer.png" ]; then
    cp vramancer.png "${APP_BUNDLE}/Contents/Resources/icon.png"
    # Si sips est disponible (macOS), créer un icns
    if command -v sips &> /dev/null; then
        echo "🎨 Conversion PNG → ICNS..."
        mkdir -p "${APP_BUNDLE}/Contents/Resources/icon.iconset"
        sips -z 16 16     vramancer.png --out "${APP_BUNDLE}/Contents/Resources/icon.iconset/icon_16x16.png" 2>/dev/null || true
        sips -z 32 32     vramancer.png --out "${APP_BUNDLE}/Contents/Resources/icon.iconset/icon_16x16@2x.png" 2>/dev/null || true
        sips -z 32 32     vramancer.png --out "${APP_BUNDLE}/Contents/Resources/icon.iconset/icon_32x32.png" 2>/dev/null || true
        sips -z 64 64     vramancer.png --out "${APP_BUNDLE}/Contents/Resources/icon.iconset/icon_32x32@2x.png" 2>/dev/null || true
        sips -z 128 128   vramancer.png --out "${APP_BUNDLE}/Contents/Resources/icon.iconset/icon_128x128.png" 2>/dev/null || true
        sips -z 256 256   vramancer.png --out "${APP_BUNDLE}/Contents/Resources/icon.iconset/icon_128x128@2x.png" 2>/dev/null || true
        sips -z 256 256   vramancer.png --out "${APP_BUNDLE}/Contents/Resources/icon.iconset/icon_256x256.png" 2>/dev/null || true
        sips -z 512 512   vramancer.png --out "${APP_BUNDLE}/Contents/Resources/icon.iconset/icon_256x256@2x.png" 2>/dev/null || true
        sips -z 512 512   vramancer.png --out "${APP_BUNDLE}/Contents/Resources/icon.iconset/icon_512x512.png" 2>/dev/null || true
        sips -z 1024 1024 vramancer.png --out "${APP_BUNDLE}/Contents/Resources/icon.iconset/icon_512x512@2x.png" 2>/dev/null || true
        iconutil -c icns "${APP_BUNDLE}/Contents/Resources/icon.iconset" -o "${APP_BUNDLE}/Contents/Resources/icon.icns" 2>/dev/null || true
        rm -rf "${APP_BUNDLE}/Contents/Resources/icon.iconset"
    fi
else
    echo "⚠️  vramancer.png non trouvé"
fi

# Créer Info.plist
echo "📝 Création de Info.plist..."
cat > "${APP_BUNDLE}/Contents/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>VRAMancer</string>
    <key>CFBundleExecutable</key>
    <string>VRAMancer</string>
    <key>CFBundleIconFile</key>
    <string>icon.icns</string>
    <key>CFBundleIdentifier</key>
    <string>com.vramancer.app</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>VRAMancer</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSPrincipalClass</key>
    <string>NSApplication</string>
    <key>LSUIElement</key>
    <false/>
</dict>
</plist>
EOF

# Créer le script de lancement principal
echo "🚀 Création du launcher macOS..."
cat > "${APP_BUNDLE}/Contents/MacOS/VRAMancer" << 'LAUNCHER_EOF'
#!/bin/bash

# VRAMancer macOS Launcher
# Ce script est le point d'entrée de l'application macOS

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Fonction pour afficher des notifications macOS
notify() {
    osascript -e "display notification \"$2\" with title \"VRAMancer\" subtitle \"$1\""
}

# Fonction pour afficher des dialogues
dialog() {
    osascript -e "display dialog \"$1\" buttons {\"OK\"} default button 1 with title \"VRAMancer\" $2"
}

# Vérifier Python 3
if command -v python3 &> /dev/null; then
    PYTHON="python3"
elif command -v python &> /dev/null; then
    PYTHON="python"
else
    dialog "Python 3 est requis pour VRAMancer.\n\nVeuillez installer Python 3 depuis:\nhttps://www.python.org/downloads/macos/" "with icon stop"
    exit 1
fi

echo "✅ Python trouvé: $($PYTHON --version)"

# Créer un environnement virtuel s'il n'existe pas
if [ ! -d ".venv" ]; then
    notify "Installation" "Première installation, création de l'environnement..."
    $PYTHON -m venv .venv
fi

# Activer l'environnement virtuel
source .venv/bin/activate

# Mettre à jour pip
pip install --upgrade pip --quiet 2>/dev/null || true

# Vérifier et installer les dépendances
echo "📦 Vérification des dépendances..."
if ! $PYTHON -c "import flask" 2>/dev/null; then
    notify "Installation" "Installation des dépendances (quelques minutes)..."
    pip install -r requirements.txt --quiet || {
        dialog "Erreur lors de l'installation des dépendances.\n\nConsultez le terminal pour plus de détails." "with icon stop"
        exit 1
    }
fi

# Créer les logs
LOG_DIR="$HOME/.vramancer/logs"
mkdir -p "$LOG_DIR"
API_LOG="$LOG_DIR/api.log"
APP_LOG="$LOG_DIR/app.log"

# Démarrer l'API en arrière-plan
notify "Démarrage" "Lancement de l'API VRAMancer..."
$PYTHON api_simple.py > "$API_LOG" 2>&1 &
API_PID=$!
echo "API PID: $API_PID" > "$LOG_DIR/pids.txt"

# Attendre que l'API soit prête
echo "⏳ Attente du démarrage de l'API..."
sleep 5

# Vérifier si l'API répond
for i in {1..10}; do
    if curl -s http://localhost:5030/health > /dev/null 2>&1; then
        echo "✅ API prête!"
        break
    fi
    sleep 1
done

# Lancer l'interface appropriée
notify "VRAMancer" "Lancement de l'interface..."

# Vérifier si PyQt5 est disponible pour le system tray
if $PYTHON -c "import PyQt5" 2>/dev/null; then
    echo "🎨 Lancement du System Tray..."
    $PYTHON systray_vramancer.py >> "$APP_LOG" 2>&1
else
    # Fallback: lancer le launcher auto ou ouvrir le dashboard web
    if [ -f "launcher_auto.py" ]; then
        echo "🚀 Lancement du launcher automatique..."
        $PYTHON launcher_auto.py
    else
        # Ouvrir le dashboard web dans le navigateur
        notify "Dashboard Web" "Ouverture du dashboard dans votre navigateur..."
        sleep 2
        open http://localhost:5030
        
        # Lancer le dashboard web avancé s'il existe
        if [ -f "dashboard/dashboard_web_advanced.py" ]; then
            $PYTHON dashboard/dashboard_web_advanced.py >> "$APP_LOG" 2>&1
        else
            dialog "Dashboard web disponible sur:\nhttp://localhost:5030\n\nAppuyez sur OK pour arrêter l'API." ""
        fi
    fi
fi

# Cleanup à la fermeture
echo "🛑 Arrêt de VRAMancer..."
if [ -f "$LOG_DIR/pids.txt" ]; then
    API_PID=$(cat "$LOG_DIR/pids.txt" | grep "API PID" | cut -d: -f2 | xargs)
    [ -n "$API_PID" ] && kill $API_PID 2>/dev/null || true
fi

notify "VRAMancer" "Application fermée"
LAUNCHER_EOF

chmod +x "${APP_BUNDLE}/Contents/MacOS/VRAMancer"

# Créer le script vrm_start.sh dans le bundle
chmod +x "${APP_BUNDLE}/Contents/MacOS/vrm_start.sh"

# Créer le README pour le DMG
echo "📄 Création du README..."
cat > "${DMG_DIR}/README.txt" << 'README_EOF'
╔═══════════════════════════════════════════════════════════════════╗
║                   VRAMancer pour macOS                            ║
║              Orchestrateur IA Multi-GPU                           ║
╚═══════════════════════════════════════════════════════════════════╝

📦 INSTALLATION
───────────────

1. Glissez l'icône VRAMancer.app dans le dossier Applications
2. Lancez VRAMancer depuis Applications ou Launchpad
3. Au premier lancement, l'application installera les dépendances Python

⚙️ PRÉREQUIS
─────────────

• Python 3.9 ou supérieur (installé par défaut sur macOS 10.15+)
• Si Python n'est pas installé : https://www.python.org/downloads/macos/

🚀 PREMIER LANCEMENT
────────────────────

• L'installation des dépendances prend 2-5 minutes
• L'API démarre automatiquement sur http://localhost:5030
• Le System Tray apparaît dans la barre de menu (si PyQt5 installé)
• Sinon, le dashboard web s'ouvre dans votre navigateur

🎮 INTERFACES DISPONIBLES
──────────────────────────

1. System Tray (barre de menu) - Recommandé
2. Dashboard Web - http://localhost:5030
3. Dashboard Qt - Interface graphique native
4. Dashboard Mobile - Responsive design

Pour installer PyQt5 et activer le System Tray :
  python3 -m pip install PyQt5

📚 DOCUMENTATION
────────────────

• README.md : Documentation complète (dans VRAMancer.app/Contents/MacOS/)
• MANUEL_FR.md : Guide en français
• MANUAL_EN.md : English manual
• http://localhost:5030/docs : Documentation API

🛠️ LANCEMENT MANUEL
────────────────────

Terminal :
  cd /Applications/VRAMancer.app/Contents/MacOS
  source .venv/bin/activate
  python3 api_simple.py &
  python3 systray_vramancer.py

Ou :
  ./vrm_start.sh

🔍 DÉPANNAGE
─────────────

• Logs : ~/.vramancer/logs/
• API ne démarre pas : vérifier le port 5030 (lsof -i :5030)
• PyQt5 non trouvé : pip3 install PyQt5
• Python non trouvé : installer depuis python.org

📧 SUPPORT
──────────

• GitHub : https://github.com/thebloodlust/VRAMancer
• Issues : https://github.com/thebloodlust/VRAMancer/issues

Version 1.0.0 - 2025
Licence MIT
README_EOF

# Copier l'app dans le dossier DMG
echo "📦 Préparation du contenu du DMG..."
cp -r "${APP_BUNDLE}" "${DMG_DIR}/"

# Créer un lien symbolique vers Applications
ln -s /Applications "${DMG_DIR}/Applications"

# Créer un fichier .DS_Store pour une présentation élégante (optionnel)
# Cela nécessiterait macOS, donc on le laisse pour l'instant

# Créer le DMG
echo "💿 Création du fichier .dmg..."

if command -v hdiutil &> /dev/null; then
    # Sur macOS - Création du DMG
    echo "✅ hdiutil trouvé, création du DMG..."
    
    # Créer une image temporaire
    TMP_DMG="${BUILD_DIR}/tmp.dmg"
    hdiutil create -srcfolder "${DMG_DIR}" -volname "${APP_NAME}" -fs HFS+ \
        -fsargs "-c c=64,a=16,e=16" -format UDRW -size 500m "${TMP_DMG}"
    
    # Monter l'image
    MOUNT_DIR="/Volumes/${APP_NAME}"
    hdiutil attach "${TMP_DMG}" -mountpoint "${MOUNT_DIR}"
    
    # Personnaliser l'apparence (optionnel)
    # echo '
    #   tell application "Finder"
    #     tell disk "'${APP_NAME}'"
    #       open
    #       set current view of container window to icon view
    #       set toolbar visible of container window to false
    #       set the bounds of container window to {400, 100, 920, 440}
    #       set viewOptions to the icon view options of container window
    #       set arrangement of viewOptions to not arranged
    #       set icon size of viewOptions to 72
    #       set position of item "'${APP_NAME}'.app" of container window to {160, 205}
    #       set position of item "Applications" of container window to {360, 205}
    #       close
    #       open
    #       update without registering applications
    #       delay 2
    #     end tell
    #   end tell
    # ' | osascript || true
    
    # Démonter
    hdiutil detach "${MOUNT_DIR}"
    
    # Convertir en DMG compressé final
    hdiutil convert "${TMP_DMG}" -format UDZO -imagekey zlib-level=9 -o "${FINAL_DMG}"
    
    # Nettoyer
    rm -f "${TMP_DMG}"
    
    echo "✅ DMG créé avec succès: ${FINAL_DMG}"
    echo "📦 Taille: $(du -h "${FINAL_DMG}" | cut -f1)"
    
else
    # Pas sur macOS - Créer une archive tar.gz
    echo "⚠️  hdiutil non disponible (nécessite macOS pour créer un .dmg)"
    echo "📦 Création d'une archive .tar.gz à la place..."
    
    ARCHIVE="${DMG_NAME}.tar.gz"
    tar -czf "${ARCHIVE}" -C "${BUILD_DIR}" dmg
    
    echo "✅ Archive créée: ${ARCHIVE}"
    echo "📦 Taille: $(du -h "${ARCHIVE}" | cut -f1)"
    echo ""
    echo "💡 Pour créer le .dmg, transférez cette archive sur macOS et exécutez:"
    echo ""
    
    # Créer un script helper pour macOS
    cat > "create_dmg_on_macos.sh" << 'HELPER_EOF'
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
HELPER_EOF
    
    chmod +x create_dmg_on_macos.sh
    echo "   ./create_dmg_on_macos.sh"
fi

echo ""
echo "═════════════════════════════════════════════════════════"
echo "✅ Build terminé avec succès!"
echo "═════════════════════════════════════════════════════════"
echo ""
echo "📁 Contenu:"
echo "   • ${APP_NAME}.app - Application macOS bundle"
echo "   • README.txt - Instructions d'installation"
echo "   • Lien Applications - Pour drag & drop"
echo ""

if [ -f "${FINAL_DMG}" ]; then
    echo "🎁 Fichier de distribution: ${FINAL_DMG}"
    echo ""
    echo "🚀 Pour installer:"
    echo "   1. Double-cliquez sur ${FINAL_DMG}"
    echo "   2. Glissez VRAMancer.app dans Applications"
    echo "   3. Lancez VRAMancer depuis Launchpad"
else
    echo "🎁 Archive de distribution: ${DMG_NAME}.tar.gz"
    echo ""
    echo "🚀 Pour créer le DMG sur macOS:"
    echo "   ./create_dmg_on_macos.sh"
fi

echo ""
echo "📊 Structure du bundle:"
find "${APP_BUNDLE}" -maxdepth 3 -type d | head -20

echo ""
echo "✨ Prêt pour la distribution!"
