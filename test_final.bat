@echo off
title VRAMancer - Test Final

echo ===============================================
echo   VRAMANCER - TEST FINAL (API DEJA ACTIVE)
echo ===============================================
echo.
echo L'API semble fonctionner sur port 5030
echo Ce script teste les interfaces directement
echo.

:menu
echo ===============================================
echo   MENU RAPIDE
echo ===============================================
echo 1. Test API simple
echo 2. Qt Dashboard (corrige)
echo 3. Debug Web
echo 4. Quitter
echo.

set /p choice=Choix (1-4): 

if "%choice%"=="1" goto test_api
if "%choice%"=="2" goto qt_dash
if "%choice%"=="3" goto debug_web
if "%choice%"=="4" exit /b 0

echo Choix invalide
goto menu

:test_api
echo.
echo Test API simple...
python -c "import requests; r=requests.get('http://localhost:5030/health'); print('Status:', r.status_code); print('Data:', r.json())"
echo.
pause
goto menu

:qt_dash
echo.
echo Qt Dashboard (version corrigee)...
echo Si ca plante encore, l'erreur sera differente
python dashboard\dashboard_qt.py
echo.
echo Qt termine
pause
goto menu

:debug_web
echo.
echo Debug Web...
python debug_web.py
goto menu