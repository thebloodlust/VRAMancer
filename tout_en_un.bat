@echo off
title VRAMancer - Tout-en-Un

echo ===============================================
echo   VRAMANCER - LANCEUR TOUT-EN-UN
echo ===============================================
echo.

REM Kill processus existants sur port 5030
echo Nettoyage port 5030...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5030" 2^>nul') do (
    echo Arret processus PID %%p
    taskkill /f /pid %%p >nul 2>&1
)

REM Forcer l'installation des dépendances
echo Installation dependencies...
python -m pip install flask requests PyQt5 --quiet --upgrade

echo.
echo ===============================================
echo   DEMARRAGE API (FORCE)
echo ===============================================

REM Lancement API avec gestion d'erreur
echo Lancement API VRAMancer...
start "VRAMancer API - LAISSER OUVERT" cmd /k "echo API VRAMancer - NE PAS FERMER CETTE FENETRE & echo. & python start_api.py"

echo Attendre que l'API se lance (10 secondes)...
timeout /t 10 /nobreak >nul

REM Test API plusieurs fois
echo Test API tentative 1/3...
python -c "import requests; requests.get('http://localhost:5030/health', timeout=2)" >nul 2>&1
if %errorlevel% equ 0 goto api_ready

timeout /t 3 /nobreak >nul
echo Test API tentative 2/3...
python -c "import requests; requests.get('http://localhost:5030/health', timeout=2)" >nul 2>&1
if %errorlevel% equ 0 goto api_ready

timeout /t 3 /nobreak >nul
echo Test API tentative 3/3...
python -c "import requests; requests.get('http://localhost:5030/health', timeout=2)" >nul 2>&1
if %errorlevel% equ 0 goto api_ready

goto api_ok

:api_ready
echo API repond correctement !
goto menu

:api_ok
echo.
echo ATTENTION: API peut ne pas etre prete
echo Regardez la fenetre API pour les erreurs
echo Appuyez sur une touche pour continuer quand meme...
pause

:menu
echo.
echo ===============================================
echo   MENU PRINCIPAL (API Active)
echo ===============================================
echo 1. Test API (avec gestion erreur)
echo 2. Qt Dashboard
echo 3. Debug Web
echo 4. Relancer API
echo 5. Quitter
echo.

set /p choice=Choix (1-5): 

if "%choice%"=="1" goto test_api
if "%choice%"=="2" goto qt_dash
if "%choice%"=="3" goto debug_web
if "%choice%"=="4" goto restart_api
if "%choice%"=="5" goto quit

echo Choix invalide
goto menu

:test_api
echo.
echo ===============================================
echo   TEST API AVEC GESTION ERREUR
echo ===============================================
python -c "try: import requests; r=requests.get('http://localhost:5030/health', timeout=5); print('✓ API Status:', r.status_code); print('✓ Response:', r.json()); except Exception as e: print('✗ Erreur API:', str(e)); print('Solution: Relancez l\'API avec option 4')"
echo.
pause
goto menu

:qt_dash
echo.
echo ===============================================
echo   QT DASHBOARD (AVEC TIMEOUT)
echo ===============================================
echo Test rapide API avant Qt...
python -c "import requests; requests.get('http://localhost:5030/health', timeout=1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo API non accessible - Qt va planter
    echo Relancez l'API d'abord (option 4)
    pause
    goto menu
)

echo API OK, lancement Qt Dashboard...
python dashboard\dashboard_qt.py
echo.
echo Qt Dashboard termine.
pause
goto menu

:debug_web
echo.
echo ===============================================
echo   DEBUG WEB
echo ===============================================
echo Test rapide API avant debug web...
python -c "import requests; requests.get('http://localhost:5030/health', timeout=1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo API non accessible - Debug web va rester bloque
    echo Relancez l'API d'abord (option 4)
    pause
    goto menu
)

echo API OK, lancement Debug Web...
python debug_web.py
goto menu

:restart_api
echo.
echo ===============================================
echo   RELANCER API
===============================================
echo Arret API existante...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5030" 2^>nul') do taskkill /f /pid %%p >nul 2>&1
timeout /t 2 /nobreak >nul

echo Relancement API...
start "VRAMancer API - LAISSER OUVERT" cmd /k "echo API VRAMancer - NE PAS FERMER CETTE FENETRE & echo. & python start_api.py"
echo Attente demarrage...
timeout /t 8 /nobreak >nul
echo API relancee
goto menu

:quit
echo.
echo Arret de l'API...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5030" 2^>nul') do taskkill /f /pid %%p >nul 2>&1
echo Au revoir !
exit /b 0