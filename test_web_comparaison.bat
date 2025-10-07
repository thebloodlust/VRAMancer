@echo off
title VRAMancer - Test Debug Web Comparaison

echo ===============================================
echo   VRAMANCER - TEST DEBUG WEB COMPARAISON
echo ===============================================
echo.
echo Test API d'abord...
python -c "import requests; r=requests.get('http://localhost:5030/health'); print('API Status:', r.status_code)" 2>nul
if %errorlevel% neq 0 (
    echo API non accessible. Lancez api_permanente.bat d'abord.
    pause
    exit /b 1
)
echo API OK
echo.

:menu
echo ===============================================
echo   VERSIONS DEBUG WEB
echo ===============================================
echo 1. Debug Web Original (problematique)
echo 2. Debug Web Simplifie (pour diagnostic)
echo 3. Comparer les deux (2 onglets)
echo 4. Quitter
echo.

set /p choice=Choix (1-4): 

if "%choice%"=="1" goto original
if "%choice%"=="2" goto simple
if "%choice%"=="3" goto compare
if "%choice%"=="4" exit /b 0

echo Choix invalide
goto menu

:original
echo.
echo Lancement Debug Web Original...
echo Interface: http://localhost:8080
echo (Celui qui reste coince sur verification)
echo.
python debug_web.py
goto menu

:simple
echo.
echo Lancement Debug Web Simplifie...
echo Interface: http://localhost:8080
echo (Version diagnostic - boutons fonctionnels)
echo.
python debug_web_simple.py
goto menu

:compare
echo.
echo Lancement des deux versions...
echo Original sur port 8080, Simplifie sur port 8081
echo.
start "Debug Original" python debug_web.py
timeout /t 3 /nobreak >nul
start "Debug Simple" cmd /c "python -c \"
import debug_web_simple
debug_web_simple.app.run(host='0.0.0.0', port=8081, debug=False)
\""
echo.
echo Deux interfaces lancees:
echo - Original: http://localhost:8080
echo - Simple: http://localhost:8081
echo.
pause
goto menu