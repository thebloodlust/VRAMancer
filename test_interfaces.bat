@echo off
title VRAMancer - Test Interfaces (API Requise)

echo ===============================================
echo   VRAMANCER - TEST INTERFACES
===============================================
echo.
echo PREREQUIS: L'API doit tourner sur port 5030
echo Si pas d'API, lancez d'abord: api_permanente.bat
echo.

:test_api_first
echo Test presence API...
python -c "import requests; requests.get('http://localhost:5030/health', timeout=2)" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ❌ API NON ACCESSIBLE sur port 5030
    echo.
    echo Solutions:
    echo 1. Lancez api_permanente.bat dans une autre fenetre
    echo 2. Ou lancez tout_en_un.bat pour demarrage automatique
    echo.
    pause
    exit /b 1
)

echo ✅ API accessible sur port 5030

:menu
echo.
echo ===============================================
echo   INTERFACES DISPONIBLES
echo ===============================================
echo 1. Test API rapide
echo 2. Debug Web (interface diagnostic)
echo 3. Qt Dashboard (interface native)
echo 4. Re-tester API
echo 5. Quitter
echo.

set /p choice=Choix (1-5): 

if "%choice%"=="1" goto quick_api
if "%choice%"=="2" goto web_debug
if "%choice%"=="3" goto qt_interface
if "%choice%"=="4" goto test_api_first
if "%choice%"=="5" exit /b 0

echo Choix invalide
goto menu

:quick_api
echo.
echo Test API rapide...
python -c "import requests; r=requests.get('http://localhost:5030/health'); print('Status:', r.status_code); print('Data:', r.json())"
if %errorlevel% neq 0 (
    echo API a plante ! Relancez api_permanente.bat
)
echo.
pause
goto menu

:web_debug
echo.
echo Lancement Debug Web...
echo Interface sur http://localhost:8080
echo Fermez avec Ctrl+C
echo.
python debug_web.py
goto menu

:qt_interface
echo.
echo Lancement Qt Dashboard...
echo Interface native Qt
echo.
python dashboard\dashboard_qt.py
echo.
echo Qt Dashboard ferme
pause
goto menu