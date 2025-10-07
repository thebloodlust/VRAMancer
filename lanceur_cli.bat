@echo off
title VRAMancer - CLI Dashboard CORRIGE

cls
echo ===============================================================================
echo                   VRAMANCER - CLI DASHBOARD CORRIGE
echo ===============================================================================
echo.
echo Dashboard en mode ligne de commande
echo Interface textuelle avec rafraichissement
echo.

echo Test prerequis...
python --version >nul 2>&1 || (
    echo ERREUR: Python non trouve
    pause
    exit /b 1
)

echo Test requests disponible...
python -c "import requests" >nul 2>&1 || (
    echo Installation requests...
    pip install requests
)

echo.
echo Lancement CLI Dashboard VRAMancer...
echo Version non-bloquante pour demonstration
echo.

cd /d "%~dp0"

REM Essai version simplifiee d'abord
if exist "cli_simple.py" (
    python cli_simple.py
) else (
    python dashboard\dashboard_cli.py
)

echo.
echo CLI Dashboard arrete
pause