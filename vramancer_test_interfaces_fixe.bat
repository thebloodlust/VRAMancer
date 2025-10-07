@echo off
title VRAMancer - Test Interfaces CORRIGE

cls
echo ===============================================================================
echo           VRAMANCER - TEST INTERFACES CORRIGE STANDALONE
echo ===============================================================================
echo.
echo Ce script genere TOUJOURS un fichier de resultats
echo Fonctionne meme sans API active
echo.

set "LOGFILE=vramancer_test_results.txt"
echo === VRAMANCER TEST INTERFACES === > "%LOGFILE%"
echo Date: %date% %time% >> "%LOGFILE%"
echo. >> "%LOGFILE%"

echo Test en cours...
echo Resultats sauves dans: %LOGFILE%
echo.

REM Test 1 - API
echo ===============================================================================
echo   TEST 1/10 - API VRAMANCER
echo ===============================================================================
echo.
echo [TEST 1] API Health Check >> "%LOGFILE%"
python -c "import requests; r=requests.get('http://localhost:5030/health',timeout=3); print('API: OK' if r.status_code==200 else 'API: ERROR')" >> "%LOGFILE%" 2>&1 || echo API: DISCONNECTED >> "%LOGFILE%"
echo Test 1 termine

REM Test 2 - Debug Web Ultra
echo.
echo ===============================================================================
echo   TEST 2/10 - DEBUG WEB ULTRA
echo ===============================================================================
echo.
echo [TEST 2] Debug Web Ultra >> "%LOGFILE%"
if exist "debug_web_ultra.py" (
    echo DEBUG_WEB_ULTRA: FILE_OK >> "%LOGFILE%"
) else (
    echo DEBUG_WEB_ULTRA: MISSING >> "%LOGFILE%"
)
echo Test 2 termine

REM Test 3 - Qt Dashboard
echo.
echo ===============================================================================
echo   TEST 3/10 - QT DASHBOARD
echo ===============================================================================
echo.
echo [TEST 3] Qt Dashboard >> "%LOGFILE%"
python -c "from PyQt5 import QtWidgets; print('QT_DASHBOARD: PYQT5_OK')" >> "%LOGFILE%" 2>&1 || python -c "from PyQt6 import QtWidgets; print('QT_DASHBOARD: PYQT6_OK')" >> "%LOGFILE%" 2>&1 || echo QT_DASHBOARD: NO_QT >> "%LOGFILE%"
echo Test 3 termine

REM Test 4 - Dashboard Web Avance
echo.
echo ===============================================================================
echo   TEST 4/10 - DASHBOARD WEB AVANCE
echo ===============================================================================
echo.
echo [TEST 4] Dashboard Web Avance >> "%LOGFILE%"
if exist "dashboard\dashboard_web_advanced.py" (
    echo WEB_ADVANCED: FILE_OK >> "%LOGFILE%"
) else (
    echo WEB_ADVANCED: MISSING >> "%LOGFILE%"
)
echo Test 4 termine

REM Test 5 - Mobile Dashboard
echo.
echo ===============================================================================
echo   TEST 5/10 - MOBILE DASHBOARD
echo ===============================================================================
echo.
echo [TEST 5] Mobile Dashboard >> "%LOGFILE%"
if exist "mobile\dashboard_mobile.py" (
    echo MOBILE_DASHBOARD: FILE_OK >> "%LOGFILE%"
) else (
    echo MOBILE_DASHBOARD: MISSING >> "%LOGFILE%"
)
echo Test 5 termine

REM Test 6 - System Tray
echo.
echo ===============================================================================
echo   TEST 6/10 - SYSTEM TRAY
echo ===============================================================================
echo.
echo [TEST 6] System Tray >> "%LOGFILE%"
if exist "dashboard\systray_vramancer.py" (
    echo SYSTEM_TRAY: FILE_OK >> "%LOGFILE%"
) else (
    echo SYSTEM_TRAY: MISSING >> "%LOGFILE%"
)
echo Test 6 termine

REM Test 7 - CLI Dashboard
echo.
echo ===============================================================================
echo   TEST 7/10 - CLI DASHBOARD
echo ===============================================================================
echo.
echo [TEST 7] CLI Dashboard >> "%LOGFILE%"
if exist "dashboard\dashboard_cli.py" (
    echo CLI_DASHBOARD: FILE_OK >> "%LOGFILE%"
) else (
    echo CLI_DASHBOARD: MISSING >> "%LOGFILE%"
)
echo Test 7 termine

REM Test 8 - Tkinter Dashboard
echo.
echo ===============================================================================
echo   TEST 8/10 - TKINTER DASHBOARD
echo ===============================================================================
echo.
echo [TEST 8] Tkinter Dashboard >> "%LOGFILE%"
python -c "import tkinter; print('TKINTER_DASHBOARD: OK')" >> "%LOGFILE%" 2>&1 || echo TKINTER_DASHBOARD: NO_TKINTER >> "%LOGFILE%"
echo Test 8 termine

REM Test 9 - Launcher Auto
echo.
echo ===============================================================================
echo   TEST 9/10 - LAUNCHER AUTO
echo ===============================================================================
echo.
echo [TEST 9] Launcher Auto >> "%LOGFILE%"
if exist "dashboard\launcher.py" (
    echo LAUNCHER_AUTO: FILE_OK >> "%LOGFILE%"
) else (
    echo LAUNCHER_AUTO: MISSING >> "%LOGFILE%"
)
echo Test 9 termine

REM Test 10 - Debug Web Simple
echo.
echo ===============================================================================
echo   TEST 10/10 - DEBUG WEB SIMPLE
echo ===============================================================================
echo.
echo [TEST 10] Debug Web Simple >> "%LOGFILE%"
if exist "debug_web_simple.py" (
    echo DEBUG_WEB_SIMPLE: FILE_OK >> "%LOGFILE%"
) else (
    echo DEBUG_WEB_SIMPLE: MISSING >> "%LOGFILE%"
)
echo Test 10 termine

REM Generation rapport
echo.
echo ===============================================================================
echo   GENERATION RAPPORT FINAL
echo ===============================================================================
echo.
echo. >> "%LOGFILE%"
echo === RESUME FINAL === >> "%LOGFILE%"

echo Generation du rapport final...

REM Lecture et analyse simple
echo. >> "%LOGFILE%"
echo INTERFACES TESTEES: >> "%LOGFILE%"
findstr "API:\|DEBUG_WEB_ULTRA:\|QT_DASHBOARD:\|WEB_ADVANCED:\|MOBILE_DASHBOARD:\|SYSTEM_TRAY:\|CLI_DASHBOARD:\|TKINTER_DASHBOARD:\|LAUNCHER_AUTO:\|DEBUG_WEB_SIMPLE:" "%LOGFILE%" >> "%LOGFILE%"

echo. >> "%LOGFILE%"
echo RECOMMANDATIONS: >> "%LOGFILE%"
echo 1. Lancer api_permanente.bat en premier >> "%LOGFILE%"
echo 2. Utiliser Qt Dashboard si PyQt detecte >> "%LOGFILE%"
echo 3. Debug Web Ultra pour interface web >> "%LOGFILE%"
echo 4. Mobile Dashboard sur http://localhost:5003 >> "%LOGFILE%"

echo ===============================================================================
echo                              AFFICHAGE RAPPORT
echo ===============================================================================
echo.
type "%LOGFILE%"

echo.
echo ===============================================================================
echo Tests termines !
echo Rapport sauve dans: %LOGFILE%
echo.
echo PROCHAINES ETAPES:
echo   1. Lancer vramancer_menu_simple.bat
echo   2. Option 1: API Permanente
echo   3. Option 11: Qt Dashboard (si PyQt OK)
echo   4. Option 10: Debug Web Ultra
echo.
echo Appuyez sur une touche pour fermer...
pause >nul