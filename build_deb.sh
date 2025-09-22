#!/bin/bash

APP_NAME="vramancer"
VERSION="0.1.0"
DEB_DIR="${APP_NAME}-deb"
BIN_DIR="${DEB_DIR}/usr/local/bin"
ICON_DIR="${DEB_DIR}/usr/share/icons/hicolor/128x128/apps"
DESKTOP_DIR="${DEB_DIR}/usr/share/applications"
CONTROL_DIR="${DEB_DIR}/DEBIAN"

echo "📦 Construction du paquet .deb pour $APP_NAME..."

# Nettoyage
rm -rf "$DEB_DIR"
mkdir -p "$BIN_DIR" "$ICON_DIR" "$DESKTOP_DIR" "$CONTROL_DIR"

# Fichier de contrôle
cat <<EOF > "${CONTROL_DIR}/control"
Package: $APP_NAME
Version: $VERSION
Section: utils
Priority: optional
Architecture: all
Maintainer: Jérémie
Description: Optimisation VRAM multi-GPU pour IA locale
EOF

# Copie du lanceur
cp run.sh "${BIN_DIR}/${APP_NAME}"
chmod +x "${BIN_DIR}/${APP_NAME}"

# Icône (à placer dans le projet sous le nom vramancer.png)
cp vramancer.png "${ICON_DIR}/${APP_NAME}.png"

# Fichier .desktop pour menu Ubuntu
cat <<EOF > "${DESKTOP_DIR}/${APP_NAME}.desktop"
[Desktop Entry]
Name=VRAMancer
Comment=Optimisation VRAM multi-GPU pour IA locale
Exec=/usr/local/bin/${APP_NAME}
Icon=${APP_NAME}
Terminal=true
Type=Application
Categories=Development;
EOF

# Construction du paquet
dpkg-deb --build "$DEB_DIR"
echo "✅ Paquet créé : ${DEB_DIR}.deb"
