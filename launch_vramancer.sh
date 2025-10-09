#!/bin/bash

echo "====================================================="
echo "    VRAMANCER - LANCEMENT RAPIDE AMELIORE"
echo "====================================================="
echo ""

echo "🚀 Lancement avec toutes les améliorations:"
echo "   • API avec endpoints /api/nodes, /api/gpu, /api/system"
echo "   • GPU adaptatif MB/GB selon usage RTX 4060"
echo "   • Détails nodes complets"
echo "   • Mobile sans erreurs 404"
echo ""

# Arrêt des anciens processus
echo "🔧 Nettoyage des anciens processus..."
pkill -f "python.*api" 2>/dev/null
pkill -f "python.*dashboard" 2>/dev/null
pkill -f "python.*systray" 2>/dev/null
pkill -f "python.*mobile" 2>/dev/null
sleep 2

# Vérification environnement
if [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python"
    echo "✅ Environnement virtuel détecté"
else
    PYTHON_CMD="python"
    echo "⚠️ Utilisation Python système"
fi

# Lancement API améliorée
echo "🔌 Lancement API avec améliorations..."
$PYTHON_CMD api_simple.py &
API_PID=$!
echo "API lancée (PID: $API_PID)"

# Attente démarrage API
echo "⏱️ Attente démarrage API (5 secondes)..."
sleep 5

# Test API
echo "🧪 Test de l'API..."
if curl -s http://localhost:5030/health > /dev/null 2>&1; then
    echo "✅ API opérationnelle"
else
    echo "❌ API non accessible - Continuez quand même"
fi

echo ""
echo "🎯 Choix de l'interface:"
echo "[1] System Tray (Hub central)"
echo "[2] Dashboard Qt (Interface native)" 
echo "[3] Dashboard Web (http://localhost:5000)"
echo "[4] Dashboard Mobile (http://localhost:5003)"
echo "[5] Tous les dashboards"
echo ""

read -p "Votre choix (1-5): " choice

case $choice in
    1)
        echo "🚀 Lancement System Tray..."
        $PYTHON_CMD systray_vramancer.py
        ;;
    2)
        echo "🎮 Lancement Dashboard Qt..."
        $PYTHON_CMD dashboard/dashboard_qt.py
        ;;
    3)
        echo "🌐 Lancement Dashboard Web..."
        echo "URL: http://localhost:5000"
        $PYTHON_CMD dashboard/dashboard_web_advanced.py
        ;;
    4)
        echo "📱 Lancement Dashboard Mobile..."
        echo "URL: http://localhost:5003"
        $PYTHON_CMD mobile/dashboard_mobile.py
        ;;
    5)
        echo "🚀 Lancement tous les dashboards..."
        $PYTHON_CMD dashboard/dashboard_qt.py &
        $PYTHON_CMD dashboard/dashboard_web_advanced.py &
        $PYTHON_CMD mobile/dashboard_mobile.py &
        echo "✅ Tous les dashboards lancés"
        echo "Qt: Interface native"
        echo "Web: http://localhost:5000"
        echo "Mobile: http://localhost:5003"
        wait
        ;;
    *)
        echo "❌ Choix invalide"
        exit 1
        ;;
esac

echo ""
echo "✅ VRAMancer avec toutes les améliorations actif!"
echo ""