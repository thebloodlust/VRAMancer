@echo off
echo ===============================================
echo     TEST PYTHON MINIMAL
echo ===============================================

echo Repertoire actuel: %CD%
echo.

echo Test 1: Python disponible?
python --version 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERREUR: Python non trouve
    echo Installez Python depuis python.org
    pause
    exit /b 1
)

echo.
echo Test 2: Python peut executer du code?
python -c "print('Hello World')" 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERREUR: Python ne peut pas executer de code
    pause
    exit /b 1
)

echo.
echo Test 3: Fichiers VRAMancer presents?
dir debug_web.py 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERREUR: debug_web.py non trouve
    echo Etes-vous dans le bon repertoire?
    pause
    exit /b 1
)

echo.
echo Test 4: Import Flask possible?
python -c "import flask; print('Flask disponible')" 2>&1
if %ERRORLEVEL% neq 0 (
    echo Flask non installe, installation...
    python -m pip install flask 2>&1
)

echo.
echo Test 5: Import requests possible?
python -c "import requests; print('Requests disponible')" 2>&1
if %ERRORLEVEL% neq 0 (
    echo Requests non installe, installation...
    python -m pip install requests 2>&1
)

echo.
echo ===============================================
echo   TOUS LES TESTS PASSES
echo ===============================================
echo Python fonctionne correctement
echo Fichiers presents
echo Dependencies installees
echo.

echo Maintenant, choisissez:
echo 1. Lancer debug_web.py
echo 2. Lancer start_api.py  
echo 3. Quitter
echo.

:choice_loop
set /p user_choice="Votre choix (1-3): "

if "%user_choice%"=="1" (
    echo.
    echo Lancement debug_web.py...
    python debug_web.py
    echo.
    echo Debug web termine.
    goto choice_loop
)

if "%user_choice%"=="2" (
    echo.
    echo Lancement start_api.py...
    python start_api.py
    echo.
    echo API terminee.
    goto choice_loop
)

if "%user_choice%"=="3" (
    echo Au revoir!
    exit /b 0
)

echo Choix invalide, essayez encore.
goto choice_loop