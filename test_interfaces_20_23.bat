@echo off
title Test Interfaces 20-23 CORRIGES

cls
echo ===============================================================================
echo                    TEST INTERFACES 20-23 CORRIGES
echo ===============================================================================
echo.
echo Test des interfaces specialisees corrigees
echo Options: 20 System Tray, 21 CLI, 22 Tkinter, 23 Launcher Auto
echo.

set "timestamp=%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%"
set "timestamp=%timestamp: =0%"
set "logfile=test_interfaces_20-23_%timestamp%.txt"

echo Test demarrer: %date% %time% > "%logfile%"
echo Interface,Status,Erreur >> "%logfile%"
echo. >> "%logfile%"

echo Test de l'API d'abord...
python -c "import requests; r=requests.get('http://localhost:5030/health', timeout=3); print('API OK:', r.status_code==200)" 2>>"%logfile%"
if %ERRORLEVEL% neq 0 (
    echo ATTENTION: API VRAMancer non accessible
    echo Demarrez l'API avec option 1 du menu principal
    pause
)

echo.
echo === TEST INTERFACE 20: SYSTEM TRAY ===
echo Test System Tray... >> "%logfile%"
call lanceur_systray.bat 2>>"%logfile%"
if %ERRORLEVEL% equ 0 (
    echo System Tray,OK, >> "%logfile%"
    echo [✓] System Tray: OK
) else (
    echo System Tray,ERREUR,Code %ERRORLEVEL% >> "%logfile%"
    echo [❌] System Tray: ERREUR
)

echo.
echo === TEST INTERFACE 21: CLI DASHBOARD ===
echo Test CLI Dashboard... >> "%logfile%"
call lanceur_cli.bat 2>>"%logfile%"
if %ERRORLEVEL% equ 0 (
    echo CLI Dashboard,OK, >> "%logfile%"
    echo [✓] CLI Dashboard: OK
) else (
    echo CLI Dashboard,ERREUR,Code %ERRORLEVEL% >> "%logfile%"
    echo [❌] CLI Dashboard: ERREUR
)

echo.
echo === TEST INTERFACE 22: TKINTER DASHBOARD ===
echo Test Tkinter Dashboard... >> "%logfile%"
call lanceur_tkinter.bat 2>>"%logfile%"
if %ERRORLEVEL% equ 0 (
    echo Tkinter Dashboard,OK, >> "%logfile%"
    echo [✓] Tkinter Dashboard: OK
) else (
    echo Tkinter Dashboard,ERREUR,Code %ERRORLEVEL% >> "%logfile%"
    echo [❌] Tkinter Dashboard: ERREUR
)

echo.
echo === TEST INTERFACE 23: LAUNCHER AUTO ===
echo Test Launcher Auto... >> "%logfile%"
call lanceur_auto.bat 2>>"%logfile%"
if %ERRORLEVEL% equ 0 (
    echo Launcher Auto,OK, >> "%logfile%"
    echo [✓] Launcher Auto: OK
) else (
    echo Launcher Auto,ERREUR,Code %ERRORLEVEL% >> "%logfile%"
    echo [❌] Launcher Auto: ERREUR
)

echo.
echo Test termine: %date% %time% >> "%logfile%"
echo.
echo ===============================================================================
echo                           RESULTATS TEST
echo ===============================================================================
echo.
echo Test des interfaces 20-23 termine
echo Rapport detaille sauve dans: %logfile%
echo.
type "%logfile%"
echo.
echo ===============================================================================
pause