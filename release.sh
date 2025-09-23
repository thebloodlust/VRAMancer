#!/bin/bash
echo "🚀 Lancement du build complet de VRAMancer..."

# Nettoyage
echo "🧹 Nettoyage..."
find . -type d -name "__pycache__" -exec rm -r {} +

# Installation
echo "📦 Installation des dépendances..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Tests
echo "🧪 Lancement des tests..."
pytest tests/

# Build .deb
echo "📦 Construction du paquet .deb..."
chmod +x debian/postinst
dpkg-deb --build debian vramancer_1.0.deb

echo "✅ VRAMancer est prêt pour distribution !"
echo "📦 Paquet généré : vramancer_1.0.deb"
