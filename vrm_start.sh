#!/bin/bash
# VRAMancer Quick Start - Mac/Linux

clear
echo "VRAMancer Quick Start"
echo ""

# Detect Python
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON="python3"
else
    PYTHON="python"
fi

echo "[1] System Tray (Hub)"
echo "[2] Qt Dashboard"
echo "[3] Web Dashboard"
echo "[4] Mobile Dashboard"
echo ""
read -p "Choice (1-4): " choice

# Clean processes
pkill -f "python.*api" 2>/dev/null
pkill -f "python.*dashboard" 2>/dev/null
sleep 1

# Start API
$PYTHON api_simple.py &
sleep 3

# Launch interface
case $choice in
    1) $PYTHON systray_vramancer.py ;;
    2) $PYTHON dashboard/dashboard_qt.py ;;
    3) 
        if command -v open &> /dev/null; then open http://localhost:5000; fi
        if command -v xdg-open &> /dev/null; then xdg-open http://localhost:5000; fi
        $PYTHON dashboard/dashboard_web_advanced.py ;;
    4) 
        if command -v open &> /dev/null; then open http://localhost:5003; fi
        if command -v xdg-open &> /dev/null; then xdg-open http://localhost:5003; fi
        $PYTHON mobile/dashboard_mobile.py ;;
    *) $PYTHON systray_vramancer.py ;;
esac

echo "VRAMancer started!"