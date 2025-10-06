@echo off
echo ===============================================
echo    VRAMancer - Launcher Windows Universel
echo ===============================================
echo.

REM Détection automatique du répertoire
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Repertoire de travail: %CD%
echo.

REM Vérification de Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python non detecte
    echo Installation automatique depuis Microsoft Store...
    start ms-windows-store://pdp/?ProductId=9NRWMJP3717K
    echo.
    echo Ou telecharger depuis: https://python.org
    pause
    exit /b 1
)

echo Python OK: 
python --version
echo.

REM Menu de choix
:MENU
echo ===============================================
echo Choisissez votre interface VRAMancer:
echo ===============================================
echo 1. Systray (icone dans la barre des taches)
echo 2. Dashboard Web (navigateur)
echo 3. Interface graphique Qt
echo 4. Interface graphique Tkinter  
echo 5. Fix et diagnostic Windows
echo 6. Test cluster heterogene
echo 7. Configuration cluster
echo 8. Quitter
echo.
set /p choice="Votre choix (1-8): "

if "%choice%"=="1" goto SYSTRAY
if "%choice%"=="2" goto WEB
if "%choice%"=="3" goto QT
if "%choice%"=="4" goto TKINTER
if "%choice%"=="5" goto FIX
if "%choice%"=="6" goto TEST
if "%choice%"=="7" goto CONFIG
if "%choice%"=="8" goto EXIT

echo Choix invalide, reessayez...
goto MENU

:SYSTRAY
echo.
echo Lancement du systray...
if exist "systray_vramancer.py" (
    python systray_vramancer.py
) else if exist "release_bundle\systray_vramancer.py" (
    python release_bundle\systray_vramancer.py
) else (
    echo Systray non trouve, lancement du dashboard web...
    goto WEB
)
goto END

:WEB
echo.
echo Lancement du dashboard web...
if exist "dashboard\dashboard_web.py" (
    python dashboard\dashboard_web.py
) else if exist "gui.py" (
    python gui.py
) else (
    echo Dashboard web non trouve
    goto FIX
)
goto END

:QT
echo.
echo Lancement interface Qt...
if exist "dashboard\dashboard_qt.py" (
    python dashboard\dashboard_qt.py
) else (
    echo Interface Qt non trouvee, tentative Tkinter...
    goto TKINTER
)
goto END

:TKINTER
echo.
echo Lancement interface Tkinter...
if exist "dashboard\dashboard_tk.py" (
    python dashboard\dashboard_tk.py
) else if exist "gui.py" (
    python gui.py
) else (
    echo Interface Tkinter non trouvee
    goto WEB
)
goto END

:FIX
echo.
echo Lancement diagnostic Windows...
if exist "fix_windows_simple.py" (
    python fix_windows_simple.py
) else if exist "fix_windows_dashboard.py" (
    python fix_windows_dashboard.py
) else (
    echo Aucun outil de diagnostic trouve
    echo Creation d'un diagnostic minimal...
    python -c "import sys; print('Python:', sys.version); import platform; print('OS:', platform.system(), platform.release()); print('Repertoire:', '%CD%'); import os; print('Fichiers Python:'); [print(f) for f in os.listdir('.') if f.endswith('.py')]; input('Appuyez sur Entree...')"
)
goto MENU

:TEST
echo.
echo Lancement test cluster...
if exist "test_heterogeneous_cluster.py" (
    python test_heterogeneous_cluster.py
) else (
    echo Test cluster non trouve
    python -c "print('Module de test non disponible dans cette version'); input('Appuyez sur Entree...')"
)
goto MENU

:CONFIG
echo.
echo Configuration cluster...
if exist "setup_heterogeneous_cluster.py" (
    python setup_heterogeneous_cluster.py
) else (
    echo Configuration cluster non trouvee
    python -c "print('Module de configuration non disponible'); input('Appuyez sur Entree...')"
)
goto MENU

:EXIT
echo Au revoir !
exit /b 0

:END
if errorlevel 1 (
    echo.
    echo Une erreur s'est produite
    echo Retour au menu principal...
    echo.
    pause
    goto MENU
)

echo.
echo Termine. Retour au menu...
pause
goto MENU