#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "    🚀 Démarrage unifié de VRAMancer         "
echo "=============================================="

# Activate venv if present
if [[ -f ".venv/bin/activate" ]]; then
    source ".venv/bin/activate"
    echo "  ✓ Virtualenv activé"
fi

# Load .env if present
if [[ -f ".env" ]]; then
    set -a; source .env; set +a
    echo "  ✓ Variables .env chargées"
fi

# Kill stale processes on our ports (best-effort)
for port in 8500 5030 8560; do
    if command -v lsof &>/dev/null; then
        kill -9 $(lsof -t -i:"$port" 2>/dev/null) 2>/dev/null || true
    elif command -v fuser &>/dev/null; then
        fuser -k "$port"/tcp 2>/dev/null || true
    fi
done

# 1. Start API server
echo "[1/2] Lancement de l'API (5030) et WebGPU (8560)..."
python3 -m vramancer.main serve > api.log 2>&1 &
API_PID=$!

sleep 2

# Health check — wait for API to come up
MAX_RETRIES=10
for i in $(seq 1 $MAX_RETRIES); do
    if curl -sf http://127.0.0.1:5030/live >/dev/null 2>&1; then
        echo "  ✓ API en ligne"
        break
    fi
    if [[ $i -eq $MAX_RETRIES ]]; then
        echo "  ⚠ API ne répond pas (vérifiez api.log)"
    fi
    sleep 1
done

# 2. Start Dashboard
echo "[2/2] Lancement du Dashboard (8500)..."
PYTHONPATH=. python3 dashboard/launcher.py > dash.log 2>&1 &
DASH_PID=$!

echo ""
echo "✅ VRAMancer est en ligne !"
echo "🌐 Dashboard principal : http://127.0.0.1:8500"
echo "📡 API santé           : http://127.0.0.1:5030/health"
echo ""
echo "🛑 Appuyez sur CTRL+C pour arrêter proprement tous les services."

# Graceful shutdown on CTRL+C or SIGTERM
cleanup() {
    echo -e '\nArrêt des services...'
    kill "$API_PID" "$DASH_PID" 2>/dev/null || true
    wait "$API_PID" "$DASH_PID" 2>/dev/null || true
    echo "Services arrêtés."
    exit 0
}
trap cleanup INT TERM

# Wait for background processes
wait