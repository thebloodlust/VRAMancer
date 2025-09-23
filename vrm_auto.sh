#!/bin/bash
echo "🤖 VRAMancer auto-adaptatif"

# Vérifie si un GPU est disponible
has_gpu=$(python3 -c "import torch; print(torch.cuda.is_available())")

# Vérifie si une interface graphique est disponible
has_display=${DISPLAY:-""}

# Mode par défaut
mode="cli"

if [ "$has_gpu" = "True" ]; then
  if [ -n "$has_display" ]; then
    # Si GPU + GUI → Qt
    mode="qt"
  else
    # Si GPU sans GUI → CLI
    mode="cli"
  fi
else
  if [ -n "$has_display" ]; then
    # Si pas de GPU mais GUI → Tkinter
    mode="tk"
  else
    # Si rien → Web (fallback)
    mode="web"
  fi
fi

echo "🧠 Environnement détecté : GPU=$has_gpu, DISPLAY=$has_display"
echo "🚀 Lancement en mode : $mode"
python3 /opt/vramancer/launcher.py --mode "$mode"
