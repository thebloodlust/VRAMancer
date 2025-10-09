@echo off
title VRAMancer - RTX 4060 Ready

echo =====================================================
echo      VRAMANCER - ULTIMATE LAUNCHER 
echo =====================================================
echo.
echo All corrections applied:
echo - Mobile Dashboard: GPU error fixed
echo - Web Dashboard: Node details complete  
echo - Qt Dashboard: Adaptive MB/GB display
echo - RTX 4060: Full support with precise monitoring
echo.

REM Clean any previous processes
taskkill /f /im python.exe >nul 2>&1
timeout /t 1 /nobreak >nul

echo Starting API...
start "VRAMancer API" cmd /k "python api_simple.py"
timeout /t 4 /nobreak >nul

echo.
echo CHOOSE YOUR INTERFACE:
echo [1] System Tray (RECOMMENDED - All-in-one hub)
echo [2] Qt Dashboard (Native interface + RTX monitoring)
echo [3] Web Dashboard (Advanced supervision - localhost:5000)
echo [4] Mobile Dashboard (Responsive - localhost:5003)
echo [5] Launch ALL interfaces
echo.
set /p choice="Your choice (1-5): "

if "%choice%"=="1" (
    echo Starting System Tray Hub...
    echo Right-click tray icon for full menu
    python systray_vramancer.py
) else if "%choice%"=="2" (
    echo Starting Qt Dashboard...
    echo RTX 4060 adaptive MB/GB display ready
    python dashboard/dashboard_qt.py
) else if "%choice%"=="3" (
    echo Starting Web Dashboard...
    echo Opening http://localhost:5000
    start http://localhost:5000
    python dashboard/dashboard_web_advanced.py
) else if "%choice%"=="4" (
    echo Starting Mobile Dashboard...
    echo Opening http://localhost:5003
    start http://localhost:5003
    python mobile/dashboard_mobile.py
) else if "%choice%"=="5" (
    echo Starting ALL interfaces...
    start "Qt" cmd /k "python dashboard/dashboard_qt.py"
    start "Web" cmd /k "python dashboard/dashboard_web_advanced.py"
    start "Mobile" cmd /k "python mobile/dashboard_mobile.py"
    start http://localhost:5000
    start http://localhost:5003
    echo All dashboards launched!
    python systray_vramancer.py
) else (
    echo Invalid choice - Starting System Tray by default
    python systray_vramancer.py
)

echo.
echo VRAMancer active with all RTX 4060 enhancements!
pause