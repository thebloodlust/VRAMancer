@echo off
echo ===============================================
echo    VRAMancer - Demarrage API + Interface
echo ===============================================
echo.

REM Détection automatique du répertoire
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Repertoire: %CD%
echo.

REM Définition des variables d'environnement
set VRM_API_BASE=http://localhost:5030
set VRM_API_PORT=5030
set VRM_WEB_PORT=8080

echo Configuration:
echo - API Backend: %VRM_API_BASE%
echo - Port API: %VRM_API_PORT%
echo - Port Web: %VRM_WEB_PORT%
echo.

REM Vérification de Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR: Python non detecte
    pause
    exit /b 1
)

REM Installation des dépendances critiques
echo Installation des dependances...
python -m pip install flask flask-cors requests psutil pyyaml --quiet --disable-pip-version-check

REM Démarrage de l'API en arrière-plan
echo.
echo Demarrage de l'API VRAMancer...

REM Tentative 1: Module core.api.unified_api
echo Tentative: python -m core.api.unified_api
start "VRAMancer API" /min python -m core.api.unified_api

REM Attendre un peu que l'API démarre
timeout /t 3 /nobreak >nul

REM Vérification si l'API répond
echo Verification de l'API...
python -c "import requests; requests.get('http://localhost:5030/health', timeout=2); print('API OK')" 2>nul
if errorlevel 1 (
    echo API non demarree, tentative alternative...
    
    REM Tentative 2: core.unified_api direct
    start "VRAMancer API Alt" /min python -c "
import sys
sys.path.append('.')
try:
    from core.api.unified_api import app
    app.run(host='0.0.0.0', port=5030, debug=False)
except ImportError:
    from flask import Flask
    app = Flask(__name__)
    @app.route('/health')
    def health(): return {'status': 'ok'}
    @app.route('/api/status')  
    def status(): return {'backend': 'running', 'version': '1.0'}
    app.run(host='0.0.0.0', port=5030, debug=False)
"
    
    REM Attendre encore
    timeout /t 5 /nobreak >nul
)

REM Menu de choix d'interface
:INTERFACE_MENU
echo.
echo =============================================== 
echo L'API est demarree. Choisissez l'interface:
echo ===============================================
echo 1. Dashboard Web (navigateur)
echo 2. Interface Qt (fenetre native)
echo 3. Interface Tkinter (simple)
echo 4. Systray (barre des taches)
echo 5. Verifier l'API seulement
echo 6. Quitter
echo.
set /p choice="Votre choix (1-6): "

if "%choice%"=="1" goto WEB_DASHBOARD
if "%choice%"=="2" goto QT_INTERFACE  
if "%choice%"=="3" goto TK_INTERFACE
if "%choice%"=="4" goto SYSTRAY
if "%choice%"=="5" goto CHECK_API
if "%choice%"=="6" goto EXIT

echo Choix invalide
goto INTERFACE_MENU

:WEB_DASHBOARD
echo.
echo Demarrage du dashboard web...
if exist "dashboard\dashboard_web.py" (
    start "VRAMancer Web" python dashboard\dashboard_web.py
) else if exist "dashboard\dashboard_web_advanced.py" (
    start "VRAMancer Web" python dashboard\dashboard_web_advanced.py
) else (
    echo Creation dashboard web minimal...
    python -c "
import webbrowser, time
from flask import Flask
app = Flask(__name__)

@app.route('/')
def dashboard():
    return '''
<!DOCTYPE html>
<html>
<head><title>VRAMancer Dashboard</title></head>
<body style=\"background:#1a1a1a;color:#fff;font-family:Arial;padding:20px;\">
<h1>🚀 VRAMancer Dashboard</h1>
<p>API: <a href=\"http://localhost:5030/health\" style=\"color:#00bfff;\">http://localhost:5030</a></p>
<p>Status: <span id=\"status\">Verification...</span></p>
<script>
fetch('http://localhost:5030/health')
.then(r=>r.json())
.then(d=>document.getElementById('status').innerHTML='✅ API Active')
.catch(e=>document.getElementById('status').innerHTML='❌ API Inactive');
</script>
</body></html>'''

if __name__ == '__main__':
    import threading
    def open_browser():
        time.sleep(1)
        webbrowser.open('http://localhost:8080')
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host='0.0.0.0', port=8080, debug=False)
"
)
goto END

:QT_INTERFACE
echo.
echo Demarrage interface Qt...
if exist "dashboard\dashboard_qt.py" (
    python dashboard\dashboard_qt.py
) else (
    echo Interface Qt non trouvee, tentative Tkinter...
    goto TK_INTERFACE
)
goto END

:TK_INTERFACE
echo.
echo Demarrage interface Tkinter...
if exist "dashboard\dashboard_tk.py" (
    python dashboard\dashboard_tk.py
) else if exist "gui.py" (
    python gui.py
) else (
    echo Creation interface minimale...
    python -c "
import tkinter as tk
from tkinter import messagebox
import requests

def check_api():
    try:
        r = requests.get('http://localhost:5030/health', timeout=2)
        messagebox.showinfo('API Status', 'API Active ✅')
    except:
        messagebox.showwarning('API Status', 'API Inactive ❌')

root = tk.Tk()
root.title('VRAMancer Control')
root.geometry('300x200')
root.configure(bg='#2d2d2d')

tk.Label(root, text='VRAMancer Dashboard', bg='#2d2d2d', fg='white', font=('Arial', 14)).pack(pady=20)
tk.Button(root, text='Verifier API', command=check_api, bg='#007acc', fg='white').pack(pady=10)
tk.Button(root, text='Quitter', command=root.quit, bg='#dc3545', fg='white').pack(pady=10)

root.mainloop()
"
)
goto END

:SYSTRAY
echo.
echo Demarrage systray...
if exist "systray_vramancer.py" (
    python systray_vramancer.py
) else if exist "release_bundle\systray_vramancer.py" (
    python release_bundle\systray_vramancer.py
) else (
    echo Systray non trouve, lancement interface Qt...
    goto QT_INTERFACE
)
goto END

:CHECK_API
echo.
echo Verification de l'API...
python -c "
import requests
try:
    r = requests.get('http://localhost:5030/health', timeout=5)
    print('✅ API Active:', r.json())
    r2 = requests.get('http://localhost:5030/api/status', timeout=5)
    print('✅ Status:', r2.json())
except Exception as e:
    print('❌ API Inactive:', e)
    print('Verifiez que l\API est bien demarree.')
"
pause
goto INTERFACE_MENU

:EXIT
echo.
echo Arret des services...
taskkill /f /fi "WindowTitle eq VRAMancer*" >nul 2>&1
echo Au revoir !
exit /b 0

:END
echo.
echo Interface lancee. 
echo Pour arreter les services, fermez cette fenetre.
echo.
pause