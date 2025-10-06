@echo off
title VRAMancer Dashboard Launcher
color 0A

echo ===============================================
echo        VRAMancer Dashboard Launcher
echo ===============================================
echo.

REM Vérifier si Python est disponible
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installe ou pas dans le PATH
    echo Installez Python depuis https://python.org
    pause
    exit /b 1
)

echo Python detecte:
python --version

echo.
echo Verification et lancement automatique...
echo.

REM Lancer le script Python intelligent
python launch_vramancer.py

if errorlevel 1 (
    echo.
    echo ERREUR lors du lancement
    echo.
    echo Solutions de secours:
    echo 1. Installer les dependances: pip install -r requirements-windows.txt
    echo 2. Lancer le diagnostic: python fix_windows_dashboard.py
    echo 3. Dashboard minimal: python dashboard_minimal_windows.py
    echo.
    pause
)

echo.
echo Appuyez sur une touche pour fermer...
pause >nul