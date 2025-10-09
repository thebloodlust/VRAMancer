@echo off
title VRAMancer Quick

echo VRAMancer Quick Start
echo.
echo [1] System Tray (Hub)
echo [2] Qt Dashboard
echo [3] Web Dashboard
echo [4] Mobile Dashboard
echo.
set /p choice="Choice (1-4): "

taskkill /f /im python.exe >nul 2>&1
start "API" cmd /k "python api_simple.py"
timeout /t 3 /nobreak >nul

if "%choice%"=="1" python systray_vramancer.py
if "%choice%"=="2" python dashboard/dashboard_qt.py
if "%choice%"=="3" (
    start http://localhost:5000
    python dashboard/dashboard_web_advanced.py
)
if "%choice%"=="4" (
    start http://localhost:5003
    python mobile/dashboard_mobile.py
)

echo VRAMancer started!
pause