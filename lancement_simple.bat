@echo off
cls
title VRAMancer - Post Telechargement
color 0A

echo =====================================================
echo      VRAMANCER - LANCEMENT RAPIDE
echo =====================================================
echo.
echo Toutes les corrections appliquees :
echo - Mobile Dashboard : Erreur GPU corrigee
echo - Hub : Caracteres compatibles Windows  
echo - GPU adaptatif RTX 4060 : MB/GB selon usage
echo.

REM Nettoyage processus precedents
taskkill /f /im python.exe >nul 2>&1

echo Lancement API...
start "VRAMancer API" cmd /k "python api_simple.py"
timeout /t 4 /nobreak >nul

echo.
echo CHOIX INTERFACE:
echo [1] System Tray (Recommande)
echo [2] Dashboard Qt 
echo [3] Dashboard Web
echo [4] Dashboard Mobile
echo.
set /p choice="Votre choix (1-4): "

if "%choice%"=="1" (
    echo Lancement System Tray...
    python systray_vramancer.py
) else if "%choice%"=="2" (
    echo Lancement Dashboard Qt...
    python dashboard/dashboard_qt.py
) else if "%choice%"=="3" (
    echo Lancement Dashboard Web...
    start http://localhost:5000
    python dashboard/dashboard_web_advanced.py
) else if "%choice%"=="4" (
    echo Lancement Dashboard Mobile...
    start http://localhost:5003
    python mobile/dashboard_mobile.py
) else (
    echo Choix invalide
)

echo.
echo VRAMancer actif avec toutes les corrections !
pause