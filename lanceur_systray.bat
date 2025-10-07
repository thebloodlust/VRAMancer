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

echo Test Qt disponible...
python -c "from PyQt5.QtWidgets import QApplication" >nul 2>&1 || (
    python -c "from PyQt6.QtWidgets import QApplication" >nul 2>&1 || (
        python -c "from PySide2.QtWidgets import QApplication" >nul 2>&1 || (
            python -c "from PySide6.QtWidgets import QApplication" >nul 2>&1 || (
                echo ERREUR: Aucune librairie Qt trouvee
                echo Installez: pip install PyQt5 ou PyQt6
                pause
                exit /b 1
            )
        )
    )
)

echo.
echo Lancement System Tray VRAMancer...
echo Recherche de l'icone dans la barre des taches
echo.

cd /d "%~dp0"

REM Essai version simplifiee d'abord
if exist "systray_simple.py" (
    python systray_simple.py
) else if exist "systray_vramancer.py" (
    python systray_vramancer.py
) else if exist "dashboard\systray_vramancer.py" (
    python dashboard\systray_vramancer.py
) else (
    echo ERREUR: Aucun fichier System Tray trouve
    pause
    exit /b 1
)

echo.
echo System Tray arrete
pause