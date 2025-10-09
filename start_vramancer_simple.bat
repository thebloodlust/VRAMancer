@echo off
cls
title VRAMancer Launch

echo =====================================================  
echo       VRAMANCER - SIMPLE LAUNCHER
echo =====================================================
echo.
echo Corrections applied:
echo - Mobile Dashboard: GPU error fixed
echo - Windows Hub: Character encoding fixed
echo - RTX 4060: Adaptive MB/GB display
echo.

REM Clean previous processes
taskkill /f /im python.exe >nul 2>&1
timeout /t 1 /nobreak >nul

echo Starting API...
start "VRAMancer API" cmd /k "python api_simple.py"
timeout /t 4 /nobreak >nul

echo.
echo INTERFACE CHOICE:
echo [1] System Tray (Recommended)
echo [2] Dashboard Qt
echo [3] Dashboard Web  
echo [4] Dashboard Mobile
echo.
set /p choice="Your choice (1-4): "

if "%choice%"=="1" (
    echo Starting System Tray...
    python systray_vramancer.py
) else if "%choice%"=="2" (
    echo Starting Dashboard Qt...
    python dashboard/dashboard_qt.py
) else if "%choice%"=="3" (
    echo Starting Dashboard Web...
    start http://localhost:5000
    python dashboard/dashboard_web_advanced.py
) else if "%choice%"=="4" (
    echo Starting Dashboard Mobile...
    start http://localhost:5003
    python mobile/dashboard_mobile.py
) else (
    echo Invalid choice
)

echo.
echo VRAMancer active with all corrections!
pause