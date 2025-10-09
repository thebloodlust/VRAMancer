@echo off
cls
title VRAMancer - Launcher Windows
color 0A

echo =====================================================
echo    🚀 VRAMANCER LAUNCHER WINDOWS 🚀
echo =====================================================
echo.
echo ✅ Version avec toutes les améliorations RTX 4060:
echo    • GPU adaptatif MB/GB selon usage
echo    • Endpoints API complets 
echo    • Dashboard sans erreurs 404
echo    • Détails nodes cluster
echo.

echo 🔧 Préparation...
REM Arrêt anciens processus
taskkill /f /im python.exe >nul 2>&1
timeout /t 1 /nobreak >nul

REM Vérification Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python non trouvé dans PATH
    echo 💡 Installez Python ou ajoutez-le au PATH
    pause
    exit /b 1
)

echo ✅ Python détecté
echo.

echo 🔌 Étape 1: Lancement API VRAMancer...
start "VRAMancer API" cmd /k "title VRAMancer API ^& python api_simple.py"

echo ⏱️ Attente démarrage API (6 secondes)...
timeout /t 6 /nobreak >nul

echo 🧪 Test connexion API...
python -c "import requests; r = requests.get('http://localhost:5030/health', timeout=5); print('✅ API READY' if r.status_code == 200 else '❌ API ERROR')" 2>nul
if errorlevel 1 (
    echo ⚠️ API pas encore prête - Continuons
) else (
    echo ✅ API opérationnelle sur port 5030
)

echo.
echo 🎯 Étape 2: Choix interface
echo.
echo [1] System Tray (Hub central - Recommandé)
echo [2] Dashboard Qt (Interface native GPU)
echo [3] Dashboard Web (Supervision - localhost:5000)
echo [4] Dashboard Mobile (Responsive - localhost:5003)
echo [5] Tout lancer en même temps
echo [0] API seulement (déjà lancée)
echo.
set /p choice="👉 Votre choix (0-5): "

if "%choice%"=="1" (
    echo 🚀 Lancement System Tray...
    echo 💡 Icône dans barre des tâches - clic droit pour menu
    python systray_vramancer.py
) else if "%choice%"=="2" (
    echo 🎮 Lancement Dashboard Qt...
    echo 📊 VRAM RTX 4060 avec affichage adaptatif
    python dashboard/dashboard_qt.py
) else if "%choice%"=="3" (
    echo 🌐 Lancement Dashboard Web...
    echo 🔗 Ouverture: http://localhost:5000
    start http://localhost:5000
    python dashboard/dashboard_web_advanced.py
) else if "%choice%"=="4" (
    echo 📱 Lancement Dashboard Mobile...
    echo 🔗 Ouverture: http://localhost:5003
    start http://localhost:5003
    python mobile/dashboard_mobile.py
) else if "%choice%"=="5" (
    echo 🚀 Lancement TOUT EN MÊME TEMPS...
    start "Qt Dashboard" cmd /k "title VRAMancer Qt ^& python dashboard/dashboard_qt.py"
    start "Web Dashboard" cmd /k "title VRAMancer Web ^& python dashboard/dashboard_web_advanced.py"
    start "Mobile Dashboard" cmd /k "title VRAMancer Mobile ^& python mobile/dashboard_mobile.py"
    start http://localhost:5000
    start http://localhost:5003
    echo ✅ Tous les dashboards lancés!
    echo 🎮 Qt: Interface native
    echo 🌐 Web: http://localhost:5000  
    echo 📱 Mobile: http://localhost:5003
    python systray_vramancer.py
) else if "%choice%"=="0" (
    echo ✅ API seule active sur http://localhost:5030
    echo 📋 Lancez manuellement vos interfaces
) else (
    echo ❌ Choix invalide
    pause
    exit /b 1
)

echo.
echo ✅ VRAMancer lancé avec succès!
echo 📊 RTX 4060 Laptop GPU: Affichage adaptatif MB/GB
echo 🌐 Tous endpoints fonctionnels
echo.
pause