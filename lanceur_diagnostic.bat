@echo off
echo ===============================================
echo     LANCEUR SIMPLE - Diagnostic Total
echo ===============================================

:menu
echo.
echo Repertoire: %CD%
echo.
echo 1. Test Debug Web (avec diagnostic complet)
echo 2. Test Qt Dashboard (avec diagnostic complet)
echo 3. Test API seulement
echo 4. Installer dependances Python
echo 5. Quitter
echo.
set /p choice="Votre choix (1-5): "

if "%choice%"=="1" goto debug_web
if "%choice%"=="2" goto qt_dashboard
if "%choice%"=="3" goto test_api
if "%choice%"=="4" goto install_deps
if "%choice%"=="5" goto end
goto menu

:debug_web
echo.
echo Lancement test Debug Web complet...
call test_debug_web_full.bat
echo.
echo Appuyez sur une touche pour revenir au menu...
pause >nul
goto menu

:qt_dashboard
echo.
echo Lancement test Qt Dashboard complet...
call test_qt_dashboard_full.bat
echo.
echo Appuyez sur une touche pour revenir au menu...
pause >nul
goto menu

:test_api
echo.
echo Test de l'API...
python --version
echo.
python -c "import requests; r=requests.get('http://localhost:5030/health', timeout=5); print('API Status:', r.status_code, r.json())" 2>&1
echo.
echo Appuyez sur une touche pour revenir au menu...
pause >nul
goto menu

:install_deps
echo.
echo Installation des dependances...
python -m pip install --upgrade pip
python -m pip install flask requests PyQt5
echo.
echo Installation terminee.
echo Appuyez sur une touche pour revenir au menu...
pause >nul
goto menu

:end
echo Au revoir !
exit /b 0