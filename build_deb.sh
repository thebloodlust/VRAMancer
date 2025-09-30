#!/bin/bash


APP_NAME="vramancer"
VERSION="0.1.0"
DEB_DIR="${APP_NAME}_deb"
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


cp scripts/vramancer-launcher.sh "${BIN_DIR}/${APP_NAME}-launcher.sh"
chmod +x "${BIN_DIR}/${APP_NAME}-launcher.sh"
mkdir -p "${DEB_DIR}/usr/local/share/vramancer"
cp vramancer.png "${DEB_DIR}/usr/local/share/vramancer/vramancer.png"
cat <<EOF > "${DESKTOP_DIR}/${APP_NAME}.desktop"
[Desktop Entry]
Name=VRAMancer
Comment=Optimisation VRAM multi-GPU pour IA locale
Exec=/usr/local/bin/${APP_NAME}-launcher.sh
Icon=/usr/local/share/vramancer/vramancer.png
Terminal=true
Type=Application
Categories=Development;
EOF


# Construction du paquet
dpkg-deb --build "$DEB_DIR"
echo "✅ Paquet créé : ${DEB_DIR}.deb"
