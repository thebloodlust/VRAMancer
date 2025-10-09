@echo off
cls
echo =====================================================
echo    VRAMANCER - LANCEMENT RAPIDE AMELIORE
echo =====================================================
echo.

echo 🚀 Lancement automatique avec toutes les améliorations:
echo    • API avec endpoints /api/nodes, /api/gpu, /api/system
echo    • GPU adaptatif MB/GB selon usage
echo    • Détails nodes complets
echo    • Mobile sans erreurs 404
echo.

REM Arrêt des anciens processus
echo 🔧 Nettoyage des anciens processus...
taskkill /f /im python.exe >nul 2>&1
timeout /t 2 /nobreak >nul

REM Lancement API améliorée
echo 🔌 Lancement API avec améliorations...
start "VRAMancer API" cmd /k "python api_simple.py"

REM Attente démarrage API
echo ⏱️ Attente démarrage API...
timeout /t 5 /nobreak >nul

REM Test API
echo 🧪 Test de l'API...
python -c "import requests; r = requests.get('http://localhost:5030/health', timeout=3); print('✅ API OK' if r.status_code == 200 else '❌ API KO')" >nul 2>&1
if errorlevel 1 (
    echo ❌ Problème API - Continuez quand même
) else (
    echo ✅ API opérationnelle
)

echo.
echo 🎯 Lancement System Tray (Hub central)...
echo 💡 Clic droit sur l'icône pour accéder aux dashboards
echo.

python systray_vramancer.py

echo.
echo 📋 Interfaces disponibles:
echo    • System Tray: Hub central avec menu complet
echo    • Dashboard Qt: python dashboard/dashboard_qt.py
echo    • Web Advanced: python dashboard/dashboard_web_advanced.py (http://localhost:5000)
echo    • Mobile: python mobile/dashboard_mobile.py (http://localhost:5003)
echo.
pause