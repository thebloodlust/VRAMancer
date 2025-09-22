#!/bin/bash

echo "🧠 VRAMancer — Lancement global"

# Active le venv si présent
if [ -d ".venv" ]; then
  source .venv/bin/activate
else
  echo "⚠️ Aucun environnement virtuel détecté. Lancez ./install.sh d’abord."
  exit 1
fi

# Vérifie que les dépendances sont installées
REQUIRED=("torch" "flask" "psutil" "pyyaml" "numpy")
for pkg in "${REQUIRED[@]}"; do
  pip show "$pkg" > /dev/null || {
    echo "❌ Dépendance manquante : $pkg"
    echo "➡️  Lancez : pip install -r requirements.txt"
    exit 1
  }
done

# Lancement CLI + dashboard
echo "🚀 Lancement de VRAMancer avec dashboard web..."
python launcher.py --config config.yaml --dashboard --port 5000
