@echo off
title VRAMancer - Solution Complete

echo ===============================================
echo   VRAMANCER - SOLUTION COMPLETE
echo ===============================================
echo.

REM Test basique
python --version
if %ERRORLEVEL% neq 0 (
    echo ERREUR: Python manquant
    pause
    exit /b 1
)

REM Installation des dependencies
echo Installation dependencies...
python -m pip install flask requests PyQt5 --quiet

echo.
echo ===============================================
echo   DEMARRAGE API (OBLIGATOIRE)
echo ===============================================

REM Kill processes existants sur port 5030
echo Nettoyage processes existants...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5030"') do taskkill /f /pid %%p >nul 2>&1

REM Test si API deja active
echo Test API existante...
python -c "import requests; requests.get('http://localhost:5030/health', timeout=1); print('API deja active')" 2>nul
if %ERRORLEVEL% equ 0 (
    echo API deja disponible
    goto menu
)

REM Lancement API SYNCHRONE pour debug
echo Lancement API en mode debug...
echo Si l'API ne demarre pas, vous verrez l'erreur:
echo.
start "VRAMancer API" cmd /c "python start_api.py & pause"

REM Attente plus longue
echo Attente demarrage API (15 secondes)...
timeout /t 15 /nobreak >nul

REM Verification API
echo Test final API...
python -c "import requests; requests.get('http://localhost:5030/health', timeout=2); print('API OK')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo.
    echo ===============================================
    echo   PROBLEME API - DIAGNOSTIC
    echo ===============================================
    echo L'API ne repond pas sur localhost:5030
    echo.
    echo Solutions possibles:
    echo 1. Verifiez que start_api.py existe
    echo 2. Regardez la fenetre API pour les erreurs
    echo 3. Port 5030 peut-etre utilise par un autre programme
    echo.
    echo Test manuel: python start_api.py
    echo.
    pause
    goto menu
) else (
    echo API demarree avec succes !
)

:menu
echo.
echo ===============================================
echo   MENU PRINCIPAL
echo ===============================================
echo 1. Test API direct
echo 2. Debug Web (port 8080)
echo 3. Qt Dashboard (interface native)
echo 4. Redemarrer API
echo 5. Quitter
echo.

:input
set /p choice=Choix (1-5): 

if "%choice%"=="1" goto test_api
if "%choice%"=="2" goto web_debug  
if "%choice%"=="3" goto qt_dash
if "%choice%"=="4" goto restart_api
if "%choice%"=="5" exit /b 0

echo Choix invalide
goto input

:test_api
echo.
echo ===============================================
echo   TEST API DIRECT
echo ===============================================
python -c "try: import requests; r=requests.get('http://localhost:5030/health', timeout=5); print('Status:', r.status_code); print('Response:', r.json()); except Exception as e: print('Erreur:', e)"
echo.
pause
goto menu

:web_debug
echo.
echo ===============================================
echo   LANCEMENT DEBUG WEB
echo ===============================================
echo Interface sur: http://localhost:8080
echo Ctrl+C pour arreter
echo.
python debug_web.py
goto menu

:qt_dash
echo.
echo ===============================================
echo   LANCEMENT QT DASHBOARD
echo ===============================================
echo Test import Qt...
python -c "from PyQt5.QtWidgets import QApplication; print('Qt OK')" 2>nul
if %ERRORLEVEL% neq 0 (
    echo Probleme Qt, installation...
    python -m pip install PyQt5
)

echo Lancement Qt Dashboard...
python dashboard\dashboard_qt.py
goto menu

:restart_api
echo.
echo Redemarrage API...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5030"') do taskkill /f /pid %%p >nul 2>&1
timeout /t 2 /nobreak >nul
start "VRAMancer API" cmd /c "python start_api.py & pause"
echo API redemarree, attente...
timeout /t 10 /nobreak >nul
goto menu