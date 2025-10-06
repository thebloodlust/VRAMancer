@echo off
title VRAMancer - Diagnostic Robuste
color 0A

:start
cls
echo.
echo  ███████████████████████████████████████████████████
echo  █                                                 █
echo  █         VRAMANCER - DIAGNOSTIC ROBUSTE         █  
echo  █              (Ne se ferme JAMAIS)              █
echo  █                                                 █
echo  ███████████████████████████████████████████████████
echo.
echo  Repertoire: %CD%
echo  Date/Heure: %DATE% %TIME%
echo.

echo ===============================================
echo   DIAGNOSTIC AUTOMATIQUE
echo ===============================================

REM Test 1: Python
set PYTHON_OK=0
python --version >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo ✅ Python installe
    set PYTHON_OK=1
) else (
    echo ❌ Python NON installe ou pas dans PATH
)

REM Test 2: Fichiers
set FILES_OK=0
if exist "debug_web.py" (
    if exist "start_api.py" (
        echo ✅ Fichiers VRAMancer presents
        set FILES_OK=1
    ) else (
        echo ❌ start_api.py manquant
    )
) else (
    echo ❌ debug_web.py manquant
)

REM Test 3: Dependencies (seulement si Python OK)
set DEPS_OK=0
if %PYTHON_OK% equ 1 (
    python -c "import flask, requests; print('Dependencies OK')" >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        echo ✅ Dependencies Python installees
        set DEPS_OK=1
    ) else (
        echo ❌ Flask ou Requests manquant
    )
)

echo.
echo ===============================================
echo   MENU PRINCIPAL
echo ===============================================
echo.
if %PYTHON_OK% equ 1 if %FILES_OK% equ 1 if %DEPS_OK% equ 1 (
    echo 🟢 SYSTEME PRET - Tous les composants fonctionnels
) else (
    echo 🔴 SYSTEME NON PRET - Problemes detectes
)
echo.
echo   1. 🐍 Installer/Verifier Python et dependencies
echo   2. 🔧 Lancer Debug Web (diagnostic + interface)
echo   3. 🖥️  Lancer Qt Dashboard (interface native)
echo   4. 📡 Tester API seulement
echo   5. 📁 Verifier fichiers et repertoire
echo   6. 🔄 Actualiser diagnostic
echo   7. ❌ Quitter
echo.

:input
set /p "choice=👉 Votre choix (1-7): "

REM Validation input
if "%choice%"=="1" goto install_deps
if "%choice%"=="2" goto launch_debug_web
if "%choice%"=="3" goto launch_qt
if "%choice%"=="4" goto test_api_only
if "%choice%"=="5" goto check_files
if "%choice%"=="6" goto start
if "%choice%"=="7" goto confirm_exit

echo ❌ Choix invalide. Utilisez 1-7.
goto input

:install_deps
cls
echo ===============================================
echo   INSTALLATION DEPENDENCIES
echo ===============================================
echo.
echo Installation de Python dependencies...
python -m pip install --upgrade pip
python -m pip install flask requests PyQt5
echo.
echo Installation terminee.
pause
goto start

:launch_debug_web
cls
echo ===============================================
echo   LANCEMENT DEBUG WEB
echo ===============================================
echo.
if %PYTHON_OK% neq 1 (
    echo ❌ Python requis. Utilisez option 1 d'abord.
    pause
    goto start
)
if %FILES_OK% neq 1 (
    echo ❌ Fichiers manquants. Verifiez le repertoire.
    pause
    goto start
)
echo 🚀 Lancement debug_web.py...
echo 📱 Une page web devrait s'ouvrir sur http://localhost:8080
echo ⏹️  Appuyez sur Ctrl+C pour arreter
echo.
python debug_web.py
echo.
echo Debug Web termine.
pause
goto start

:launch_qt
cls
echo ===============================================
echo   LANCEMENT QT DASHBOARD
echo ===============================================
echo.
if %PYTHON_OK% neq 1 (
    echo ❌ Python requis. Utilisez option 1 d'abord.
    pause
    goto start
)
if not exist "dashboard\dashboard_qt.py" (
    echo ❌ dashboard\dashboard_qt.py manquant.
    pause
    goto start
)
echo 🚀 Lancement dashboard_qt.py...
echo 🖥️  Interface Qt native devrait s'ouvrir
echo.
python dashboard\dashboard_qt.py
echo.
echo Qt Dashboard termine.
pause
goto start

:test_api_only
cls
echo ===============================================
echo   TEST API SEULEMENT
echo ===============================================
echo.
if %PYTHON_OK% neq 1 (
    echo ❌ Python requis.
    pause
    goto start
)
echo 🧪 Test de l'API VRAMancer...
echo.
python -c "try: import requests; r=requests.get('http://localhost:5030/health', timeout=3); print('✅ API Active:', r.json()); except Exception as e: print('❌ API Erreur:', e)"
echo.
pause
goto start

:check_files
cls
echo ===============================================
echo   VERIFICATION FICHIERS
echo ===============================================
echo.
echo Repertoire courant: %CD%
echo.
echo Fichiers VRAMancer:
if exist "debug_web.py" (echo ✅ debug_web.py) else (echo ❌ debug_web.py MANQUANT)
if exist "start_api.py" (echo ✅ start_api.py) else (echo ❌ start_api.py MANQUANT)
if exist "dashboard\dashboard_qt.py" (echo ✅ dashboard\dashboard_qt.py) else (echo ❌ dashboard\dashboard_qt.py MANQUANT)
echo.
echo Contenu du repertoire:
dir /b *.py 2>nul
echo.
pause
goto start

:confirm_exit
echo.
set /p "confirm=❓ Etes-vous sur de vouloir quitter? (o/N): "
if /i "%confirm%"=="o" goto real_exit
if /i "%confirm%"=="oui" goto real_exit
goto start

:real_exit
echo.
echo 👋 Au revoir!
timeout /t 2 >nul
exit /b 0