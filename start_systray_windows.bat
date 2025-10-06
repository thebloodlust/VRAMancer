@echo off
echo ===============================================
echo    VRAMancer - Lancement Systray Windows
echo ===============================================
echo.

REM Détection automatique du répertoire
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Repertoire actuel: %CD%
echo.

REM Vérification de Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python n'est pas installe ou pas dans le PATH
    echo Veuillez installer Python 3.8+ depuis python.org
    pause
    exit /b 1
)

echo Python detecte: 
python --version

REM Recherche du fichier systray
if exist "systray_vramancer.py" (
    echo Lancement du systray VRAMancer...
    echo.
    python systray_vramancer.py
) else if exist "release_bundle\systray_vramancer.py" (
    echo Lancement du systray depuis release_bundle...
    echo.
    python release_bundle\systray_vramancer.py
) else if exist "dashboard\dashboard_qt.py" (
    echo Systray non trouve, lancement du dashboard Qt...
    echo.
    python dashboard\dashboard_qt.py
) else if exist "gui.py" (
    echo Lancement de l'interface graphique principale...
    echo.
    python gui.py
) else (
    echo ERREUR: Aucun fichier d'interface graphique trouve
    echo Fichiers recherches:
    echo - systray_vramancer.py
    echo - release_bundle\systray_vramancer.py  
    echo - dashboard\dashboard_qt.py
    echo - gui.py
    echo.
    echo Contenu du repertoire:
    dir /b *.py
    echo.
    pause
    exit /b 1
)

if errorlevel 1 (
    echo.
    echo ERREUR lors du lancement
    echo Tentative avec le dashboard web...
    if exist "dashboard\dashboard_web.py" (
        python dashboard\dashboard_web.py
    ) else (
        echo Aucune interface disponible
    )
)

echo.
echo Appuyez sur une touche pour fermer...
pause >nul