@echo off
echo ===============================================
echo     DIAGNOSTIC ULTRA SIMPLE - Ne se ferme JAMAIS
echo ===============================================

echo Repertoire: %CD%
echo.

echo 1. Test Python de base...
python --version
echo Code retour Python version: %ERRORLEVEL%
echo.

echo 2. Test import requests...
python -c "import requests; print('Import requests: OK')"
echo Code retour import requests: %ERRORLEVEL%
echo.

echo 3. Test connexion sans timeout...
echo Tentative de connexion a localhost:5030...
python -c "print('Test basique')"
echo Code retour test basique: %ERRORLEVEL%
echo.

echo 4. Test avec try/catch...
python -c "try: import requests; print('Requests OK'); except Exception as e: print('Erreur:', e)"
echo Code retour try/catch: %ERRORLEVEL%
echo.

echo 5. Test connexion avec gestion erreur...
python -c "try: import requests; r=requests.get('http://localhost:5030/health', timeout=1); print('API OK:', r.status_code); except Exception as e: print('API erreur:', e)"
echo Code retour test API: %ERRORLEVEL%
echo.

echo ===============================================
echo   MENU QUI NE SE FERME JAMAIS
echo ===============================================

:menu
echo.
echo A. Lancer debug_web.py directement
echo B. Lancer start_api.py directement  
echo C. Tester existence fichiers
echo D. Installer Flask
echo E. Quitter
echo.
choice /c ABCDE /m "Votre choix"

if errorlevel 5 goto end
if errorlevel 4 goto install_flask
if errorlevel 3 goto test_files
if errorlevel 2 goto start_api
if errorlevel 1 goto debug_web

:debug_web
echo.
echo Lancement debug_web.py DIRECT...
if exist debug_web.py (
    echo Fichier debug_web.py trouve
    python debug_web.py
    echo Code retour debug_web.py: %ERRORLEVEL%
) else (
    echo ERREUR: debug_web.py non trouve dans %CD%
)
echo.
echo Appuyez sur une touche...
pause >nul
goto menu

:start_api
echo.
echo Lancement start_api.py DIRECT...
if exist start_api.py (
    echo Fichier start_api.py trouve
    python start_api.py
    echo Code retour start_api.py: %ERRORLEVEL%
) else (
    echo ERREUR: start_api.py non trouve dans %CD%
)
echo.
echo Appuyez sur une touche...
pause >nul
goto menu

:test_files
echo.
echo Test existence fichiers...
if exist debug_web.py echo ✅ debug_web.py EXISTE
if not exist debug_web.py echo ❌ debug_web.py MANQUANT
if exist start_api.py echo ✅ start_api.py EXISTE  
if not exist start_api.py echo ❌ start_api.py MANQUANT
if exist dashboard\dashboard_qt.py echo ✅ dashboard\dashboard_qt.py EXISTE
if not exist dashboard\dashboard_qt.py echo ❌ dashboard\dashboard_qt.py MANQUANT
echo.
echo Appuyez sur une touche...
pause >nul
goto menu

:install_flask
echo.
echo Installation Flask...
python -m pip install flask requests
echo Code retour installation: %ERRORLEVEL%
echo.
echo Appuyez sur une touche...
pause >nul
goto menu

:end
echo.
echo Script termine normalement.
pause
exit /b 0