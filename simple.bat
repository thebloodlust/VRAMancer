@echo off
title VRAMancer Simple

echo ===============================================
echo   VRAMANCER - LANCEUR SIMPLE (Sans emojis)
echo ===============================================
echo.

REM Test basique Python
python --version
if %ERRORLEVEL% neq 0 (
    echo ERREUR: Python non trouve
    pause
    exit /b 1
)

REM Test fichiers
if not exist "debug_web.py" (
    echo ERREUR: debug_web.py manquant
    pause
    exit /b 1
)

if not exist "start_api.py" (
    echo ERREUR: start_api.py manquant  
    pause
    exit /b 1
)

echo Python: OK
echo Fichiers: OK

REM Installation dependencies basique
echo Installation des dependencies...
python -m pip install flask requests PyQt5 >nul 2>&1

REM Nettoyage port 5030
echo Nettoyage port 5030...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5030" 2^>nul') do taskkill /f /pid %%p >nul 2>&1

REM Demarrage API avec fenetre visible pour debug
echo Demarrage API (fenetre visible)...
start "VRAMancer API - NE PAS FERMER" python start_api.py
echo Attente demarrage API (12 secondes)...
timeout /t 12 /nobreak >nul

REM Test API
python -c "import requests; requests.get('http://localhost:5030/health', timeout=2); print('API OK')" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ATTENTION: API peut ne pas etre prete
    echo Verifiez la fenetre API pour les erreurs
)

:menu
echo.
echo ===============================================
echo MENU:
echo 1 = Debug Web
echo 2 = Qt Dashboard  
echo 3 = Test API
echo 4 = Quitter
echo ===============================================
set /p choice=Choix: 

if "%choice%"=="1" (
    echo.
    echo --- LANCEMENT DEBUG WEB ---
    python debug_web.py
    goto menu
)

if "%choice%"=="2" (
    echo.
    echo --- LANCEMENT QT DASHBOARD ---
    python dashboard\dashboard_qt.py
    goto menu
)

if "%choice%"=="3" (
    echo.
    echo --- TEST API ---
    python -c "import requests; print(requests.get('http://localhost:5030/health').json())"
    pause
    goto menu
)

if "%choice%"=="4" exit /b 0

echo Choix invalide
goto menu