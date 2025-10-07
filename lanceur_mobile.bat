@echo off
title VRAMancer - Lanceur Mobile Dashboard

cls
echo ===============================================================================
echo                VRAMANCER - MOBILE DASHBOARD CORRIGE
echo ===============================================================================
echo.
echo Interface mobile responsive
echo URL: http://localhost:5003
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
echo Lancement Mobile Dashboard...
echo Ne fermez pas cette fenetre !
echo Une fois lance, ouvrez: http://localhost:5003
echo.

cd /d "%~dp0"
python mobile\dashboard_mobile.py

echo.
echo Mobile Dashboard arrete
pause