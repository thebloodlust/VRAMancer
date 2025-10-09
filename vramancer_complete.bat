@echo off
cls
title VRAMancer - Sequence de Lancement Correcte

echo =====================================================
echo    VRAMANCER - LANCEMENT AVEC AMELIORATIONS
echo =====================================================
echo.

echo 🔍 Verification de l'environnement Python...
if not exist ".venv\Scripts\python.exe" (
    echo ❌ Environnement virtuel non trouve
    echo 💡 Executez d'abord: python -m venv .venv
    pause
    exit /b 1
)

echo ✅ Environnement virtuel detecte
echo.

echo 🔌 Etape 1: Lancement API avec ameliorations...
echo.
start "VRAMancer API" cmd /k ".venv\Scripts\python api_simple.py"

echo ⏱️ Attente demarrage API (5 secondes)...
timeout /t 5 /nobreak >nul

echo.
echo 🌐 Test de l'API...
.venv\Scripts\python -c "import requests; r = requests.get('http://localhost:5030/health', timeout=3); print('✅ API OK' if r.status_code == 200 else '❌ API KO')" 2>nul
if errorlevel 1 (
    echo ❌ API non disponible - Verifiez le lancement
    pause
    exit /b 1
)

echo.
echo 🎯 Etape 2: Choix de l'interface
echo.
echo [1] System Tray (Hub central recommande)
echo [2] Dashboard Qt (Interface native)
echo [3] Dashboard Web Avance (Supervision cluster)
echo [4] Dashboard Mobile (Interface responsive)
echo [5] Tous les dashboards
echo.
set /p choice="Votre choix (1-5): "

if "%choice%"=="1" (
    echo 🚀 Lancement System Tray...
    .venv\Scripts\python systray_vramancer.py
) else if "%choice%"=="2" (
    echo 🎮 Lancement Dashboard Qt...
    .venv\Scripts\python dashboard/dashboard_qt.py
) else if "%choice%"=="3" (
    echo 🌐 Lancement Dashboard Web...
    echo URL: http://localhost:5000
    .venv\Scripts\python dashboard/dashboard_web_advanced.py
) else if "%choice%"=="4" (
    echo 📱 Lancement Dashboard Mobile...
    echo URL: http://localhost:5003
    .venv\Scripts\python mobile/dashboard_mobile.py
) else if "%choice%"=="5" (
    echo 🚀 Lancement tous les dashboards...
    start "Qt Dashboard" cmd /k ".venv\Scripts\python dashboard/dashboard_qt.py"
    start "Web Dashboard" cmd /k ".venv\Scripts\python dashboard/dashboard_web_advanced.py"
    start "Mobile Dashboard" cmd /k ".venv\Scripts\python mobile/dashboard_mobile.py"
    echo ✅ Tous les dashboards lances
    echo Qt: Interface native
    echo Web: http://localhost:5000
    echo Mobile: http://localhost:5003
) else (
    echo ❌ Choix invalide
    pause
    exit /b 1
)

echo.
echo ✅ VRAMancer lance avec toutes les ameliorations:
echo    • GPU adaptatif MB/GB selon usage
echo    • Details nodes complets
echo    • Endpoints corriges
echo.
pause