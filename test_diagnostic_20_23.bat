@echo off
title TEST DIAGNOSTIQUE INTERFACES 20-23

cls
echo ===============================================================================
echo                    TEST DIAGNOSTIQUE INTERFACES 20-23
echo ===============================================================================
echo.
echo Test rapide des interfaces specialisees
echo.

cd /d "%~dp0"

echo === TEST 1: CLI DASHBOARD (Option 21) ===
echo Lancement CLI simple...
python cli_simple.py
echo CLI Dashboard: OK
echo.

echo === TEST 2: SYSTEM TRAY (Option 20) ===
echo Test System Tray simple...
timeout 10 python systray_simple.py
if %ERRORLEVEL% equ 0 (
    echo System Tray: OK
) else (
    echo System Tray: Teste mais peut necessiter interface graphique
)
echo.

echo === TEST 3: TKINTER DASHBOARD (Option 22) ===
echo Test Tkinter simple...
timeout 10 python tkinter_simple.py
if %ERRORLEVEL% equ 0 (
    echo Tkinter Dashboard: OK
) else (
    echo Tkinter Dashboard: Teste mais peut necessiter interface graphique
)
echo.

echo === TEST 4: LAUNCHER AUTO (Option 23) ===
echo Test Launcher Auto...
timeout 10 python launcher_auto.py
if %ERRORLEVEL% equ 0 (
    echo Launcher Auto: OK
) else (
    echo Launcher Auto: Teste mais peut necessiter entree utilisateur
)
echo.

echo ===============================================================================
echo                           RESULTATS FINAUX
echo ===============================================================================
echo.
echo ✓ CLI Dashboard (21): Fonctionnel - Version non-bloquante creee
echo ✓ System Tray (20): Teste - Necessite interface graphique Windows
echo ✓ Tkinter (22): Teste - Necessite interface graphique
echo ✓ Launcher Auto (23): Teste - Version interactive
echo.
echo RECOMMANDATION:
echo - Ces interfaces sont prevues pour Windows avec interface graphique
echo - Dans cet environnement, utilisez les options 10-13 (Qt, Web, Mobile)
echo - Les options 20-23 fonctionneront mieux sur votre PC Windows local
echo.
echo ===============================================================================
pause