@echo off
echo ===============================================
echo      VRAMancer - Lanceur Ultra Simple
echo ===============================================

REM Variables d'environnement
set VRM_API_BASE=http://localhost:5030
set VRM_API_PORT=5030

echo Configuration:
echo VRM_API_BASE=%VRM_API_BASE%
echo VRM_API_PORT=%VRM_API_PORT%
echo.

REM Démarrage API en arrière-plan
echo Demarrage de l'API...
start "VRAMancer API" /min python start_api.py

REM Attente démarrage
echo Attente demarrage API (5 secondes)...
timeout /t 5 /nobreak >nul

echo.
echo ===============================================
echo   Interfaces disponibles (qui fonctionnent):
echo ===============================================
echo 1. Interface Tkinter (recommandee)
echo 2. Dashboard Web (si vous voulez le web)
echo 3. Test API seulement
echo 4. Quitter
echo.

:MENU
set /p choice="Votre choix (1-4): "

if "%choice%"=="1" (
    echo.
    echo Lancement Interface Tkinter...
    python dashboard_tk_simple.py
    goto END
)

if "%choice%"=="2" (
    echo.
    echo Lancement Dashboard Web...
    echo Le navigateur va s'ouvrir automatiquement...
    start "VRAMancer Web" python dashboard_web_simple.py
    goto END
)

if "%choice%"=="3" (
    echo.
    echo Test de l'API...
    python -c "import requests; r=requests.get('http://localhost:5030/health'); print('API Health:', r.json()); r2=requests.get('http://localhost:5030/api/status'); print('API Status:', r2.json())"
    pause
    goto MENU
)

if "%choice%"=="4" (
    goto EXIT
)

echo Choix invalide, reessayez...
goto MENU

:END
echo.
echo Interface terminee.
pause

:EXIT
echo Arret des processus VRAMancer...
taskkill /f /fi "WindowTitle eq VRAMancer*" >nul 2>&1
echo Au revoir !
exit /b 0