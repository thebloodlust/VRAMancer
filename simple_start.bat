@echo off
echo ====================================
echo   VRAMANCER - LANCEMENT SIMPLE
echo ====================================
echo.
echo 1. API + System Tray
echo 2. API + Dashboard Qt  
echo 3. API + Dashboard Web
echo.
set /p choice="Choix (1-3): "

if "%choice%"=="1" (
    start cmd /k "python api_simple.py"
    timeout /t 3 /nobreak >nul
    python systray_vramancer.py
) else if "%choice%"=="2" (
    start cmd /k "python api_simple.py"  
    timeout /t 3 /nobreak >nul
    python dashboard/dashboard_qt.py
) else if "%choice%"=="3" (
    start cmd /k "python api_simple.py"
    timeout /t 3 /nobreak >nul  
    start http://localhost:5000
    python dashboard/dashboard_web_advanced.py
) else (
    echo Choix invalide
    pause
    exit
)

pause