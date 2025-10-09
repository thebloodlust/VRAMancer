#!/bin/bash
# VRAMancer Ultimate Launcher for Mac/Linux
# All RTX 4060 enhancements included

clear
echo "====================================================="
echo "      VRAMANCER - ULTIMATE LAUNCHER"
echo "====================================================="
echo ""
echo "All corrections applied:"
echo "• Mobile Dashboard: GPU error fixed"
echo "• Web Dashboard: Node details complete"
echo "• Qt Dashboard: Adaptive MB/GB display"
echo "• RTX 4060: Full support with precise monitoring"
echo ""

# Detect Python command
if [ -d ".venv" ]; then
    PYTHON_CMD=".venv/bin/python"
    echo "✅ Virtual environment detected"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "✅ Using python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
    echo "✅ Using python"
else
    echo "❌ Python not found! Please install Python"
    exit 1
fi

# Clean previous processes
echo "🧹 Cleaning previous processes..."
pkill -f "python.*api" 2>/dev/null
pkill -f "python.*dashboard" 2>/dev/null
pkill -f "python.*systray" 2>/dev/null
pkill -f "python.*mobile" 2>/dev/null
sleep 2

# Start API
echo "🚀 Starting API..."
$PYTHON_CMD api_simple.py &
API_PID=$!
echo "API launched (PID: $API_PID)"
sleep 4

# Test API
echo "🧪 Testing API..."
if curl -s http://localhost:5030/health > /dev/null 2>&1; then
    echo "✅ API operational"
else
    echo "⚠️ API starting up - continuing anyway"
fi

echo ""
echo "CHOOSE YOUR INTERFACE:"
echo "[1] System Tray (RECOMMENDED - All-in-one hub)"
echo "[2] Qt Dashboard (Native interface + RTX monitoring)"
echo "[3] Web Dashboard (Advanced supervision - localhost:5000)"
echo "[4] Mobile Dashboard (Responsive - localhost:5003)"
echo "[5] Launch ALL interfaces"
echo "[6] Console Hub (No GUI required)"
echo ""

read -p "Your choice (1-6): " choice

case $choice in
    1)
        echo "🎛️ Starting System Tray Hub..."
        echo "Right-click tray icon for full menu"
        $PYTHON_CMD systray_vramancer.py
        ;;
    2)
        echo "🖥️ Starting Qt Dashboard..."
        echo "RTX 4060 adaptive MB/GB display ready"
        $PYTHON_CMD dashboard/dashboard_qt.py
        ;;
    3)
        echo "🌐 Starting Web Dashboard..."
        echo "Opening http://localhost:5000"
        if command -v open &> /dev/null; then
            open http://localhost:5000  # macOS
        elif command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:5000  # Linux
        fi
        $PYTHON_CMD dashboard/dashboard_web_advanced.py
        ;;
    4)
        echo "📱 Starting Mobile Dashboard..."
        echo "Opening http://localhost:5003"
        if command -v open &> /dev/null; then
            open http://localhost:5003  # macOS
        elif command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:5003  # Linux
        fi
        $PYTHON_CMD mobile/dashboard_mobile.py
        ;;
    5)
        echo "🚀 Starting ALL interfaces..."
        $PYTHON_CMD dashboard/dashboard_qt.py &
        $PYTHON_CMD dashboard/dashboard_web_advanced.py &
        $PYTHON_CMD mobile/dashboard_mobile.py &
        
        if command -v open &> /dev/null; then
            open http://localhost:5000
            open http://localhost:5003
        elif command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:5000
            xdg-open http://localhost:5003
        fi
        
        echo "✅ All dashboards launched!"
        echo "Starting System Tray hub..."
        $PYTHON_CMD systray_vramancer.py
        ;;
    6)
        echo "💻 Starting Console Hub..."
        echo "Perfect for headless servers"
        $PYTHON_CMD systray_console.py
        ;;
    *)
        echo "Invalid choice - Starting System Tray by default"
        $PYTHON_CMD systray_vramancer.py
        ;;
esac

echo ""
echo "✅ VRAMancer active with all RTX 4060 enhancements!"
echo ""