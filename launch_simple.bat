@echo off
echo ================================================
echo           VRAMancer - Lanceur Simple
echo ================================================

REM Variables d'environnement
set VRM_API_BASE=http://localhost:5030
set VRM_API_PORT=5030

echo Configuration:
echo VRM_API_BASE=%VRM_API_BASE%
echo VRM_API_PORT=%VRM_API_PORT%
echo.

REM Étape 1: Démarrer l'API
echo Etape 1: Demarrage de l'API...
start "VRAMancer API" /min cmd /c "python start_api.py & pause"

REM Attendre que l'API démarre
echo Attente demarrage API (10 secondes)...
timeout /t 10 /nobreak >nul

REM Étape 2: Test de l'API
echo Etape 2: Test de l'API...
curl -s http://localhost:5030/health >nul 2>&1
if errorlevel 1 (
    echo ❌ API non accessible
    echo Tentative de test avec Python...
    python -c "import requests; print('API:', requests.get('http://localhost:5030/health').json())" 2>nul || echo API inactive
) else (
    echo ✅ API accessible
)

echo.
echo ================================================
echo       Choisissez votre interface:
echo ================================================
echo 1. Dashboard Qt (interface native)
echo 2. Dashboard Web (navigateur) 
echo 3. Systray (barre des taches)
echo 4. Interface Tkinter (simple)
echo 5. Test API seulement
echo 6. Quitter
echo.

:MENU
set /p choice="Votre choix (1-6): "

if "%choice%"=="1" (
    echo Lancement Dashboard Qt...
    python dashboard\dashboard_qt.py
    goto END
)

if "%choice%"=="2" (
    echo Lancement Dashboard Web...
    if exist "dashboard\dashboard_web.py" (
        start "VRAMancer Web" python dashboard\dashboard_web.py
        timeout /t 2 >nul
        start http://localhost:8080
    ) else (
        echo Dashboard web non trouve
    )
    goto END
)

if "%choice%"=="3" (
    echo Lancement Systray...
    if exist "systray_vramancer.py" (
        python systray_vramancer.py
    ) else (
        echo Systray non trouve, lancement GUI principal...
        python gui.py
    )
    goto END
)

if "%choice%"=="4" (
    echo Lancement Interface Tkinter...
    if exist "dashboard\dashboard_tk.py" (
        python dashboard\dashboard_tk.py
    ) else (
        python gui.py
    )
    goto END
)

if "%choice%"=="5" (
    echo Test API...
    curl -i http://localhost:5030/health
    echo.
    curl -i http://localhost:5030/api/status
    pause
    goto MENU
)

if "%choice%"=="6" (
    goto EXIT
)

echo Choix invalide, reessayez...
goto MENU

:END
echo.
echo Interface lancee.
pause

:EXIT
echo Arret des processus VRAMancer...
taskkill /f /fi "WindowTitle eq VRAMancer*" >nul 2>&1
echo Au revoir !
exit /b 0