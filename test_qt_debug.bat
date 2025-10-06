@echo off
echo ===============================================
echo   TEST QT DASHBOARD - DIAGNOSTIC DETAILLE
echo ===============================================

echo Test Python...
python --version
echo.

echo Test PyQt...
python -c "print('Test import PyQt5...');"
python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 OK')"
if %ERRORLEVEL% neq 0 (
    echo Installation PyQt5...
    python -m pip install PyQt5
    echo Test apres installation...
    python -c "from PyQt5.QtWidgets import QApplication; print('PyQt5 installe OK')"
)
echo.

echo Test fichier dashboard...
if exist "dashboard\dashboard_qt.py" (
    echo dashboard_qt.py trouve
) else (
    echo ERREUR: dashboard_qt.py non trouve
    pause
    exit /b 1
)
echo.

echo Demarrage API si necessaire...
python -c "import requests; requests.get('http://localhost:5030/health', timeout=1)" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo API non active, demarrage...
    start "API" /min python start_api.py
    timeout /t 5 /nobreak >nul
)
echo.

echo Test direct du dashboard Qt avec debug...
echo Si une fenetre Qt s'ouvre, le dashboard fonctionne
echo Si rien ne s'affiche, il y a un probleme Qt
echo.

python -c "print('=== DEBUG QT DASHBOARD ==='); from PyQt5.QtWidgets import QApplication; print('Import QApplication: OK'); import sys; app = QApplication(sys.argv); print('QApplication OK'); print('Test termine - Qt fonctionne')"

echo.
echo Test Qt Dashboard REEL...
python dashboard\dashboard_qt.py

echo.
echo Test Qt termine.
pause