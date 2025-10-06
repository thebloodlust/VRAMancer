@echo off
chcp 65001 >nul
title VRAMancer - Lanceur Automatique
color 0A

echo ===============================================
echo   VRAMANCER - LANCEUR AUTOMATIQUE
echo ===============================================
echo.
echo Repertoire: %CD%
echo.

REM Verification Python
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERREUR: Python non trouve
    echo Installez Python depuis python.org
    pause
    exit /b 1
)
echo Python: OK

REM Verification fichiers
if not exist "debug_web.py" (
    echo ERREUR: debug_web.py non trouve
    echo Etes-vous dans le bon repertoire VRAMancer?
    pause
    exit /b 1
)
if not exist "start_api.py" (
    echo ERREUR: start_api.py non trouve
    pause
    exit /b 1
)
echo Fichiers: OK

REM Installation dependencies si necessaire
echo Verification dependencies...
python -c "import flask, requests" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installation Flask et Requests...
    python -m pip install flask requests >nul 2>&1
)
echo Dependencies: OK

echo.
echo ===============================================
echo   DEMARRAGE AUTOMATIQUE API
echo ===============================================
echo.

REM Test si API deja active
python -c "import requests; requests.get('http://localhost:5030/health', timeout=1)" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo API deja active sur port 5030
) else (
    echo Demarrage de l'API en arriere-plan...
    start "VRAMancer API" /min cmd /c "python start_api.py"
    echo Attente demarrage API (8 secondes)...
    timeout /t 8 /nobreak >nul
    
    REM Verification API demarree
    python -c "import requests; requests.get('http://localhost:5030/health', timeout=2)" >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo API demarree avec succes
    ) else (
        echo ATTENTION: API peut-etre pas encore prete
    )
)

echo.
echo ===============================================
echo   MENU PRINCIPAL
echo ===============================================
echo.
echo 1. Debug Web (interface diagnostic web)
echo 2. Qt Dashboard (interface native)
echo 3. Test API uniquement
echo 4. Redemarrer API
echo 5. Quitter
echo.

:menu
set /p "choice=Votre choix (1-5): "

if "%choice%"=="1" goto debug_web
if "%choice%"=="2" goto qt_dashboard
if "%choice%"=="3" goto test_api
if "%choice%"=="4" goto restart_api
if "%choice%"=="5" goto end

echo Choix invalide
goto menu

:debug_web
echo.
echo Lancement Debug Web...
echo Interface web: http://localhost:8080
echo.
python debug_web.py
echo.
echo Debug Web termine.
goto menu

:qt_dashboard
echo.
echo Lancement Qt Dashboard...
echo.
REM Verification PyQt
python -c "import PyQt5" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installation PyQt5...
    python -m pip install PyQt5 >nul 2>&1
)
python dashboard\dashboard_qt.py
echo.
echo Qt Dashboard termine.
goto menu

:test_api
echo.
echo Test API...
python -c "try: import requests; r=requests.get('http://localhost:5030/health', timeout=3); print('API Status:', r.status_code); print('Response:', r.json()); except Exception as e: print('Erreur:', str(e))"
echo.
pause
goto menu

:restart_api
echo.
echo Redemarrage API...
taskkill /f /im python.exe /fi "windowtitle eq VRAMancer API*" >nul 2>&1
timeout /t 2 /nobreak >nul
start "VRAMancer API" /min cmd /c "python start_api.py"
echo API redemarree
timeout /t 5 /nobreak >nul
goto menu

:end
echo Au revoir!
exit /b 0