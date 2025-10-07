@echo off
title VRAMancer - Test API Simple

cls
echo ===============================================================================
echo                      VRAMANCER - TEST API SIMPLE
echo ===============================================================================
echo.
echo Ce script teste uniquement la connectivite API de base
echo.

echo Test 1: Python disponible?
python --version >nul 2>&1 
if %errorlevel% equ 0 (
    echo [OK] Python detecte
    python --version
) else (
    echo [ERREUR] Python non trouve dans PATH
    echo         Installez Python ou ajoutez-le au PATH
    pause
    exit /b 1
)

echo.
echo Test 2: Module requests disponible?
python -c "import requests" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Module requests disponible
) else (
    echo [ERREUR] Module requests manquant
    echo          Installez avec: pip install requests
    pause
    exit /b 1
)

echo.
echo Test 3: API VRAMancer sur port 5030...
python -c "import requests; r=requests.get('http://localhost:5030/health',timeout=3); exit(0 if r.status_code==200 else 1)" >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] API VRAMancer active sur port 5030
    python -c "import requests; r=requests.get('http://localhost:5030/health'); print('Reponse:', r.json())"
) else (
    echo [ERREUR] API VRAMancer non disponible sur port 5030
    echo          Lancez d'abord: api_permanente.bat
)

echo.
echo Test 4: Ports en cours d'utilisation...
echo Ports VRAMancer detectes:
netstat -an | findstr ":5030 :8080 :5000 :5003" 2>nul || echo Aucun port VRAMancer actif

echo.
echo Test 5: Fichiers principaux...
if exist "api_permanente.bat" (
    echo [OK] api_permanente.bat present
) else (
    echo [ERREUR] api_permanente.bat manquant
)

if exist "debug_web_ultra.py" (
    echo [OK] debug_web_ultra.py present
) else (
    echo [ERREUR] debug_web_ultra.py manquant
)

if exist "dashboard\dashboard_qt.py" (
    echo [OK] dashboard_qt.py present
) else (
    echo [ERREUR] dashboard_qt.py manquant
)

echo.
echo ===============================================================================
echo                              RESUME TEST
echo ===============================================================================
echo.
echo Si tous les tests sont [OK], vous pouvez utiliser VRAMancer
echo Si des [ERREUR] apparaissent:
echo   1. Pour Python: Installez Python et ajoutez au PATH
echo   2. Pour requests: pip install requests
echo   3. Pour API: Lancez api_permanente.bat
echo   4. Pour fichiers: Verifiez git pull origine main
echo.
echo Appuyez sur une touche pour fermer...
pause >nul