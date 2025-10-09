@echo off
title VRAMancer Hub RTX 4060
color 0A

echo.
echo ===============================================
echo     VRAMANCER - HUB CENTRAL RTX 4060
echo ===============================================
echo.
echo CHOIX RECOMMANDE : System Tray (tout-en-un)
echo.
echo [1] System Tray (RECOMMANDE)
echo     Interface complete avec icone
echo.
echo [2] Dashboard Qt Direct  
echo     Interface native monitoring systeme
echo.
echo [3] Dashboard Web Avance
echo     Supervision cluster detaillee (port 5000)
echo.
echo [4] Dashboard Mobile
echo     Interface responsive (port 5003)
echo.
echo [5] Tests CUDA RTX 4060
echo     Verification GPU + diagnostic
echo.
echo [6] Guide complet
echo     Documentation detaillee
echo.
echo [0] Quitter
echo.
echo ===============================================

set /p choice="Votre choix (1-6, 0 pour quitter) : "

if "%choice%"=="1" (
    echo.
    echo Lancement System Tray...
    echo Icone VRAMancer dans la barre des taches
    echo Clic droit sur l'icone pour le menu complet
    echo.
    start systray_vramancer.bat
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Lancement Dashboard Qt...
    echo Monitoring systeme + RTX 4060 temps reel
    echo.
    start dashboard_qt.bat
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Lancement Dashboard Web Avance...
    echo URL: http://localhost:5000
    echo Ouverture automatique navigateur
    echo.
    start dashboard_web_avance.bat
    goto end
)

if "%choice%"=="4" (
    echo.
    echo Lancement Dashboard Mobile...
    echo URL: http://localhost:5003
    echo Interface responsive optimisee
    echo.
    start dashboard_mobile.bat
    goto end
)

if "%choice%"=="5" (
    echo.
    echo Tests CUDA RTX 4060...
    echo Verification PyTorch + GPU
    echo.
    start test_cuda_ok.bat
    goto end
)

if "%choice%"=="6" (
    echo.
    echo Ouverture guide complet...
    echo.
    type GUIDE_LANCEMENT_SYSTRAY.txt
    echo.
    pause
    goto start
)

if "%choice%"=="0" (
    echo.
    echo Au revoir !
    goto end
)

echo.
echo Choix invalide. Essayez 1-6 ou 0.
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