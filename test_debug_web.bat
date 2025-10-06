@echo off
echo ===============================================
echo     Test Debug Web - Version Corrigee
echo ===============================================

REM Variables d'environnement
set VRM_API_BASE=http://localhost:5030
set VRM_API_PORT=5030

echo Configuration:
echo VRM_API_BASE=%VRM_API_BASE%
echo VRM_API_PORT=%VRM_API_PORT%
echo.

REM Démarrage API en arrière-plan si pas déjà active
echo Test de l'API...
python -c "import requests; requests.get('http://localhost:5030/health', timeout=2)" 2>nul
if errorlevel 1 (
    echo API non active, demarrage...
    start "VRAMancer API" /min python start_api.py
    echo Attente demarrage API (5 secondes)...
    timeout /t 5 /nobreak >nul
) else (
    echo API deja active
)

echo.
echo ===============================================
echo   Test du Debug Web corrige
echo ===============================================

echo Corrections apportees:
echo 1. Fix SyntaxWarning escape sequence
echo 2. Fonction testConnectivity exposee globalement
echo 3. Bouton de test avec logs de debug
echo.

echo Lancement du Debug Web...
echo Une page web va s'ouvrir sur http://localhost:8080
echo.

python debug_web.py

if errorlevel 1 (
    echo.
    echo =============================================== 
    echo   Debug Web a plante - Info de debug:
    echo ===============================================
    echo 1. Verifiez que Flask est installe
    echo 2. L'API doit etre active sur localhost:5030
    echo 3. Consultez les logs ci-dessus pour plus de details
    echo.
    pause
) else (
    echo.
    echo ===============================================
    echo   Debug Web fonctionne !
    echo ===============================================
    echo Le debug web s'est lance avec succes
    pause
)

echo Au revoir !
exit /b 0