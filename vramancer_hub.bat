@echo off
title VRAMancer - Hub Central RTX 4060
color 0A

echo.
echo ===============================================
echo     🚀 VRAMANCER - HUB CENTRAL RTX 4060
echo ===============================================
echo.
echo 💡 CHOIX RECOMMANDÉ : System Tray (tout-en-un)
echo.
echo [1] 🎛️  System Tray (RECOMMANDÉ)
echo     └─ Icône barre des tâches + menu complet
echo.
echo [2] 🖥️  Dashboard Qt Direct  
echo     └─ Interface native monitoring système
echo.
echo [3] 🌐 Dashboard Web Avancé
echo     └─ Supervision cluster détaillée (port 5000)
echo.
echo [4] 📱 Dashboard Mobile
echo     └─ Interface responsive (port 5003)
echo.
echo [5] 🧪 Tests CUDA RTX 4060
echo     └─ Vérification GPU + diagnostic
echo.
echo [6] 📖 Guide complet
echo     └─ Documentation détaillée
echo.
echo [0] ❌ Quitter
echo.
echo ===============================================

set /p choice="Votre choix (1-6, 0 pour quitter) : "

if "%choice%"=="1" (
    echo.
    echo 🎛️  Lancement System Tray...
    echo 📍 Icône VRAMancer dans la barre des tâches
    echo 🖱️  Clic droit sur l'icône pour le menu complet
    echo.
    start systray_vramancer.bat
    goto end
)

if "%choice%"=="2" (
    echo.
    echo 🖥️  Lancement Dashboard Qt...
    echo 📊 Monitoring système + RTX 4060 temps réel
    echo.
    start dashboard_qt.bat
    goto end
)

if "%choice%"=="3" (
    echo.
    echo 🌐 Lancement Dashboard Web Avancé...
    echo 📊 URL: http://localhost:5000
    echo 🔄 Ouverture automatique navigateur
    echo.
    start dashboard_web_avance.bat
    goto end
)

if "%choice%"=="4" (
    echo.
    echo 📱 Lancement Dashboard Mobile...
    echo 📊 URL: http://localhost:5003
    echo 🔄 Interface responsive optimisée
    echo.
    start dashboard_mobile.bat
    goto end
)

if "%choice%"=="5" (
    echo.
    echo 🧪 Tests CUDA RTX 4060...
    echo 🎮 Vérification PyTorch + GPU
    echo.
    start test_cuda_ok.bat
    goto end
)

if "%choice%"=="6" (
    echo.
    echo 📖 Ouverture guide complet...
    echo.
    type GUIDE_LANCEMENT_SYSTRAY.txt
    echo.
    pause
    goto start
)

if "%choice%"=="0" (
    echo.
    echo ✅ Au revoir !
    goto end
)

echo.
echo ❌ Choix invalide. Essayez 1-6 ou 0.
pause
goto start

:start
cls
goto :eof

:end
echo.
echo 💡 Conseil : Utilisez le System Tray pour un accès complet !
echo.
pause