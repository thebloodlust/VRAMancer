#!/bin/bash
echo "🔧 Installation de VRAMancer..."

# Crée un venv si absent
if [ ! -d ".venv" ]; then
  echo "📦 Création d’un environnement virtuel..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "📦 Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Installation terminée. Activez l’environnement avec :"
echo "source .venv/bin/activate"
