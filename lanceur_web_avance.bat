@echo off
title VRAMancer - Lanceur Dashboard Web Avance

cls
echo ===============================================================================
echo              VRAMANCER - DASHBOARD WEB AVANCE CORRIGE
echo ===============================================================================
echo.
echo Supervision cluster en temps reel
echo URL: http://localhost:5000
echo.

echo Test prerequis...
python --version >nul 2>&1 || (
    echo ERREUR: Python non trouve
    pause
    exit /b 1
)

python -c "import flask" >nul 2>&1 || (
    echo ERREUR: Flask non installe
    echo Installez avec: pip install flask
    pause
    exit /b 1
)

echo.
echo Lancement Dashboard Web Avance...
echo Ne fermez pas cette fenetre !
echo.

cd /d "%~dp0"
python dashboard\dashboard_web_advanced.py

echo.
echo Dashboard Web Avance arrete
pause