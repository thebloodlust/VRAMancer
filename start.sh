#!/bin/bash
echo "=============================================="
echo "    🚀 Demarrage unifie de VRAMancer         "
echo "=============================================="

# Nettoyage des anciens processus
kill -9 $(lsof -t -i:8500) 2>/dev/null
kill -9 $(lsof -t -i:5030) 2>/dev/null
kill -9 $(lsof -t -i:8560) 2>/dev/null

# 1. Demarrer le serveur API principal et WebGPU
echo "[1/2] Lancement de l'API (5030) et WebGPU (8560)..."
python3 main.py serve > api.log 2>&1 &
API_PID=$!

sleep 2

# 2. Demarrer le Dashboard (et l'interface Mobile Edge)
echo "[2/2] Lancement du Dashboard (8500)..."
PYTHONPATH=. python3 dashboard/launcher.py > dash.log 2>&1 &
DASH_PID=$!

echo ""
echo "✅ VRAMancer est en ligne !"
echo "🌐 Dashboard principal : http://127.0.0.1:8500"
echo "📱 Mobile Edge Node    : http://127.0.0.1:8500/mobile_edge_node.html (ou via le dashboard)"
echo ""
echo "🛑 Appuyez sur CTRL+C pour arreter proprement tous les services."

# Intercepter le CTRL+C pour tout tuer proprement
trap "echo -e '\nArret des services...'; kill $API_PID $DASH_PID 2>/dev/null; exit" INT

# Attendre que les processus se terminent
wait