@echo off
title VRAMancer - Tkinter Dashboard CORRIGE

cls
echo ===============================================================================
echo                   VRAMANCER - TKINTER DASHBOARD CORRIGE
echo ===============================================================================
echo.
echo Dashboard avec interface graphique Tkinter
echo Interface complete avec onglets et monitoring
echo.

echo Test prerequis...
python --version >nul 2>&1 || (
    echo ERREUR: Python non trouve
    pause
    exit /b 1
)

echo Test Tkinter disponible...
python -c "import tkinter" >nul 2>&1 || (
    echo ERREUR: Tkinter non disponible
    echo Tkinter est normalement inclus avec Python
    pause
    exit /b 1
)

echo Test requests disponible...
python -c "import requests" >nul 2>&1 || (
    echo Installation requests...
    pip install requests
)

echo.
echo Lancement Tkinter Dashboard VRAMancer...
echo Interface graphique avec monitoring automatique
echo.

cd /d "%~dp0"

REM Essai version simplifiee d'abord
if exist "tkinter_simple.py" (
    python tkinter_simple.py
) else (
    python dashboard\dashboard_tk.py
)

echo.
echo Tkinter Dashboard arrete
pause