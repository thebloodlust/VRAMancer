@echo off
title VRAMancer - Lanceur Unifie Complet

echo ===============================================
echo   VRAMANCER - TOUTES LES INTERFACES
echo ===============================================
echo.

REM Test API
python -c "import requests; requests.get('http://localhost:5030/health', timeout=1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo API non accessible sur port 5030
    echo Lancez d'abord: api_permanente.bat
    echo.
    pause
    exit /b 1
)
echo API OK sur port 5030

:menu
echo.
echo ===============================================
echo   INTERFACES DISPONIBLES
echo ===============================================
echo.
echo === INTERFACES PRINCIPALES ===
echo 1. Qt Dashboard (interface native) 
echo 2. Debug Web (diagnostic complet)
echo 3. Debug Web Simple (pour tests)
echo 4. Dashboard Web Avance (supervision cluster)
echo 5. Dashboard Tk (interface Tkinter)
echo.
echo === INTERFACES SPECIALISEES ===
echo 6. Dashboard CLI (ligne de commande)
echo 7. System Tray (monitoring permanent)
echo 8. Launcher Principal (auto-detection)
echo 9. Mobile Dashboard (interface adaptee)
echo.
echo === OUTILS ===
echo 10. Test API rapide
echo 11. Relancer API
echo 12. Quitter
echo.

set /p choice=Choix (1-12): 

if "%choice%"=="1" goto qt_dash
if "%choice%"=="2" goto debug_web
if "%choice%"=="3" goto debug_simple
if "%choice%"=="4" goto web_advanced
if "%choice%"=="5" goto tk_dash
if "%choice%"=="6" goto cli_dash
if "%choice%"=="7" goto systray_dash
if "%choice%"=="8" goto launcher
if "%choice%"=="9" goto mobile_dash
if "%choice%"=="10" goto test_api
if "%choice%"=="11" goto restart_api
if "%choice%"=="12" exit /b 0

echo Choix invalide
goto menu

:qt_dash
echo.
echo === QT DASHBOARD (Interface Native) ===
echo Interface Qt avec monitoring en temps reel
python dashboard\dashboard_qt.py
goto menu

:debug_web
echo.
echo === DEBUG WEB (Diagnostic Complet) ===
echo Interface web sur http://localhost:8080
python debug_web.py
goto menu

:debug_simple
echo.
echo === DEBUG WEB SIMPLE (Tests) ===
echo Interface web simplifiee sur http://localhost:8080
python debug_web_simple.py
goto menu

:web_advanced
echo.
echo === DASHBOARD WEB AVANCE (Supervision Cluster) ===
echo Supervision avancee cluster sur http://localhost:5000
if exist "dashboard\dashboard_web_advanced.py" (
    python dashboard\dashboard_web_advanced.py
) else (
    echo Fichier non trouve: dashboard\dashboard_web_advanced.py
    pause
)
goto menu

:tk_dash
echo.
echo === DASHBOARD TKINTER (Interface Stable) ===
echo Interface Tkinter native Python
if exist "dashboard\dashboard_tk.py" (
    python dashboard\dashboard_tk.py
) else (
    echo Fichier non trouve: dashboard\dashboard_tk.py
    pause
)
goto menu

:cli_dash
echo.
echo === DASHBOARD CLI (Ligne de Commande) ===
echo Interface en mode texte
if exist "dashboard\dashboard_cli.py" (
    python dashboard\dashboard_cli.py
) else (
    echo Fichier non trouve: dashboard\dashboard_cli.py
    pause
)
goto menu

:systray_dash
echo.
echo === SYSTEM TRAY (Monitoring Permanent) ===
echo Interface systray avec monitoring permanent
if exist "systray_vramancer.py" (
    python systray_vramancer.py
) else (
    echo Fichier non trouve: systray_vramancer.py
    pause
)
goto menu

:launcher
echo.
echo === LAUNCHER PRINCIPAL (Auto-detection) ===
echo Detection automatique de la meilleure interface
if exist "dashboard\launcher.py" (
    python dashboard\launcher.py --mode auto
) else (
    echo Fichier non trouve: dashboard\launcher.py
    pause
)
goto menu

:mobile_dash
echo.
echo === MOBILE DASHBOARD (Interface Adaptee) ===
echo Interface optimisee pour appareils mobiles
if exist "mobile\dashboard_mobile.py" (
    python mobile\dashboard_mobile.py
) else (
    echo Fichier non trouve: mobile\dashboard_mobile.py
    pause
)
goto menu

:test_api
echo.
echo === TEST API RAPIDE ===
python -c "import requests; r=requests.get('http://localhost:5030/health'); print('Status:', r.status_code); print('Data:', r.json())"
pause
goto menu

:restart_api
echo.
echo === RELANCER API ===
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5030" 2^>nul') do taskkill /f /pid %%p >nul 2>&1
echo API arretee, relancement...
start "VRAMancer API" cmd /k "python start_api.py"
timeout /t 5 /nobreak >nul
goto menu