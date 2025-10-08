@echo off
title VRAMancer - System Tray CORRIGE

cls
echo ===============================================================================
echo                   VRAMANCER - SYSTEM TRAY CORRIGE
echo ===============================================================================
echo.
echo Monitoring permanent du systeme
echo Interface systray avec icone Windows
echo.

echo Test prerequis...
python --version >nul 2>&1 || (
    echo ERREUR: Python non trouve
    pause
    exit /b 1
)

echo Test compatibilite System Tray...
python test_systray_icon.py

echo.
echo Test Qt disponible...
python -c "from PyQt5.QtWidgets import QApplication" >nul 2>&1 || (
    python -c "from PyQt6.QtWidgets import QApplication" >nul 2>&1 || (
        python -c "from PySide2.QtWidgets import QApplication" >nul 2>&1 || (
            python -c "from PySide6.QtWidgets import QApplication" >nul 2>&1 || (
                echo ERREUR: Aucune librairie Qt trouvee
                echo.
                echo Solution:
                echo pip install PyQt5
                echo.
                echo NOTE: System Tray necessite un environnement Windows/Linux desktop
                echo Dans cet environnement container, utilisez les interfaces web
                pause
                exit /b 1
            )
        )
    )
)

echo.
echo Lancement System Tray VRAMancer...
echo Icone VRAMancer sera utilisee automatiquement
echo Recherche de l'icone dans la barre des taches
echo.

cd /d "%~dp0"

REM Verification de l'icone vramancer.png
if exist "vramancer.png" (
    echo ✓ Icone VRAMancer detectee: vramancer.png
) else (
    echo ⚠️  Icone vramancer.png non trouvee - icone par defaut utilisee
)

REM Priorite: version complete, puis simplifiee
if exist "systray_vramancer.py" (
    echo Lancement System Tray complet avec icone...
    python systray_vramancer.py
) else if exist "systray_simple.py" (
    echo Lancement System Tray simplifie...
    python systray_simple.py
) else if exist "dashboard\systray_vramancer.py" (
    echo Lancement depuis dossier dashboard...
    python dashboard\systray_vramancer.py
) else (
    echo ERREUR: Aucun fichier System Tray trouve
    echo Fichiers recherches:
    echo - systray_vramancer.py
    echo - systray_simple.py
    echo - dashboard\systray_vramancer.py
    pause
    exit /b 1
)

echo.
echo System Tray arrete
pause