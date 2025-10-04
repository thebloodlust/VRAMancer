@echo off
REM =============================================================
REM  VRAMancer - Lancement complet Windows (API + Dashboards)
REM  Usage : double-clic ou depuis PowerShell:
REM     .\installers\start_windows_all.bat
REM =============================================================
setlocal ENABLEDELAYEDEXPANSION

REM Localisation racine projet (ce script est dans installers\)
set SCRIPT_DIR=%~dp0
pushd %SCRIPT_DIR%..
set ROOT=%CD%

echo [1/6] Racine projet : %ROOT%
IF NOT EXIST %ROOT%\core\api\unified_api.py (
  echo [ERREUR] Structure projet introuvable (core/api/unified_api.py manquant)
  pause
  exit /b 1
)

REM -------------------------------------------------------------
REM  VENV
REM -------------------------------------------------------------
if NOT EXIST .venv (
  echo [2/6] Création environnement virtuel (.venv)
  python -m venv .venv || (echo [ERREUR] Echec creation venv & pause & exit /b 1)
) else (
  echo [2/6] Environnement virtuel déjà présent
)
call .venv\Scripts\activate.bat

REM Détection requirements windows adaptés si présence tokenizers issue
if EXIST requirements-windows.txt (
  set REQ_FILE=requirements-windows.txt
) else (
  set REQ_FILE=requirements.txt
)

echo [3/6] Installation dépendances: %REQ_FILE%
python -m pip install --upgrade pip >nul 2>nul
pip install -r %REQ_FILE% || (echo [WARN] Installation partielle (mode dégradé possible))

REM PyQt5 si pas installé (web + qt optionnels)
python -c "import PyQt5" 2>nul || (echo [INFO] Installation PyQt5... & pip install pyqt5 >nul 2>nul)

REM -------------------------------------------------------------
REM  Configuration runtime
REM -------------------------------------------------------------
if "%VRM_API_PORT%"=="" set VRM_API_PORT=5030
set PYTHONPATH=%ROOT%;%PYTHONPATH%
set VRM_API_BASE=http://127.0.0.1:%VRM_API_PORT%

echo [4/6] Démarrage API unifiée (port %VRM_API_PORT%)
start "VRMancer API" .venv\Scripts\python.exe -m core.api.unified_api

REM Attente disponibilité API (max ~12s)
set /a ATTEMPTS=0
:wait_api
set /a ATTEMPTS+=1
>nul 2>nul powershell -Command "try{ $r=Invoke-WebRequest -UseBasicParsing %VRM_API_BASE%/api/health -TimeoutSec 2; if($r.StatusCode -eq 200){ exit 0 } else { exit 1 }} catch { exit 1 }"
if %ERRORLEVEL%==0 (
  echo [OK] API disponible: %VRM_API_BASE%
) else (
  if %ATTEMPTS% GEQ 6 (
    echo [WARN] API non joignable après %ATTEMPTS% tentatives (dashboard tentera auto-détection)
    goto launch_dash
  )
  timeout /t 2 >nul
  goto wait_api
)

:launch_dash
echo [5/6] Lancement Dashboard Web (port 5000)
if EXIST installers\dashboard\dashboard_web.py (
  start "VRMancer Web" .venv\Scripts\python.exe installers\dashboard\dashboard_web.py
) else if EXIST dashboard\dashboard_web.py (
  start "VRMancer Web" .venv\Scripts\python.exe dashboard\dashboard_web.py
) else (
  echo [INFO] Dashboard web introuvable (fichier manquant)
)

echo [6/6] Lancement Dashboard Qt (si PyQt5 fonctionnel)
python -c "import PyQt5" 2>nul && (start "VRMancer Qt" .venv\Scripts\python.exe dashboard\dashboard_qt.py) || echo [INFO] Qt non disponible (PyQt5 absent ou échec import)

echo.
echo =============================================================
echo  Fini. Accès:
echo   API:      %VRM_API_BASE%/api/health
if EXIST dashboard\dashboard_web.py echo   Web UI:   http://127.0.0.1:5000/
echo   Qt UI:    (fenêtre séparée si PyQt5)
echo =============================================================
echo Pour forcer un port différent: set VRM_API_PORT=5040 puis relancer.
echo Pour debug réseau Qt: set VRM_API_DEBUG=1

echo (CTRL+C pour quitter ce shell ou fermer fenêtres séparées)
popd
endlocal
