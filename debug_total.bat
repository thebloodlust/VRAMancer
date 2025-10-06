@echo off
title VRAMancer - Debug Total

echo ===============================================
echo   VRAMANCER - DEBUG TOTAL
echo ===============================================
echo.
echo Repertoire: %CD%
echo Python: 
python --version
echo.

echo ===============================================
echo   PHASE 1: VERIFICATION FICHIERS
echo ===============================================
if exist "start_api.py" (echo ✓ start_api.py) else (echo ✗ start_api.py MANQUANT & pause & exit)
if exist "debug_web.py" (echo ✓ debug_web.py) else (echo ✗ debug_web.py MANQUANT)
if exist "dashboard\dashboard_qt.py" (echo ✓ dashboard_qt.py) else (echo ✗ dashboard_qt.py MANQUANT)

echo.
echo ===============================================
echo   PHASE 2: NETTOYAGE PORT 5030
echo ===============================================
echo Recherche processus sur port 5030...
netstat -ano | findstr ":5030"
echo Arret processus existants...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5030" 2^>nul') do (
    echo Arret PID %%p
    taskkill /f /pid %%p >nul 2>&1
)

echo.
echo ===============================================
echo   PHASE 3: TEST API SIMPLE
echo ===============================================
echo Test basique start_api.py...
echo IMPORTANT: Regardez cette fenetre pour voir les erreurs API
timeout /t 3 /nobreak >nul

echo Lancement API...
python start_api.py

echo.
echo API s'est arretee. Regardez les erreurs ci-dessus.
pause