@echo off
echo ===============================================
echo     DIAGNOSTIC Qt Dashboard - Version Complete
echo ===============================================

REM Test du répertoire courant
echo Repertoire courant: %CD%
echo.

REM Test de Python
echo Test de Python...
python --version
if errorlevel 1 (
    echo ❌ ERREUR: Python non trouve dans PATH
    echo Installez Python ou ajoutez-le au PATH
    pause
    exit /b 1
) else (
    echo ✅ Python disponible
)
echo.

REM Test des fichiers nécessaires
echo Verification des fichiers...
if not exist "dashboard\dashboard_qt.py" (
    echo ❌ ERREUR: dashboard\dashboard_qt.py non trouve dans %CD%
    echo Assurez-vous d'etre dans le bon repertoire
    pause
    exit /b 1
) else (
    echo ✅ dashboard\dashboard_qt.py trouve
)

if not exist "start_api.py" (
    echo ❌ ERREUR: start_api.py non trouve dans %CD%
    echo Assurez-vous d'etre dans le bon repertoire
    pause
    exit /b 1
) else (
    echo ✅ start_api.py trouve
)
echo.

REM Variables d'environnement
set VRM_API_BASE=http://localhost:5030
set VRM_API_PORT=5030

echo Configuration:
echo VRM_API_BASE=%VRM_API_BASE%
echo VRM_API_PORT=%VRM_API_PORT%
echo.

REM Test des dépendances Python
echo Test des dependances Python...
python -c "import requests; print('Requests: OK')" 2>nul
if errorlevel 1 (
    echo ❌ Requests non installe, installation...
    python -m pip install requests
) else (
    echo ✅ Requests disponible
)

echo Test PyQt...
python -c "import PyQt5; print('PyQt5: OK')" 2>nul
if errorlevel 1 (
    python -c "import PyQt6; print('PyQt6: OK')" 2>nul
    if errorlevel 1 (
        python -c "import PySide2; print('PySide2: OK')" 2>nul
        if errorlevel 1 (
            python -c "import PySide6; print('PySide6: OK')" 2>nul
            if errorlevel 1 (
                echo ❌ Aucune librairie Qt trouvee
                echo Installation de PyQt5...
                python -m pip install PyQt5
            ) else (
                echo ✅ PySide6 disponible
            )
        ) else (
            echo ✅ PySide2 disponible
        )
    ) else (
        echo ✅ PyQt6 disponible
    )
) else (
    echo ✅ PyQt5 disponible
)
echo.

REM Démarrage API en arrière-plan si pas déjà active
echo Test de l'API...
python -c "import requests; requests.get('http://localhost:5030/health', timeout=2)" 2>nul
if errorlevel 1 (
    echo API non active, demarrage...
    echo Lancement start_api.py...
    start "VRAMancer API" python start_api.py
    echo Attente demarrage API (10 secondes)...
    timeout /t 10 /nobreak >nul
) else (
    echo ✅ API deja active
)
echo.

echo ===============================================
echo   LANCEMENT Qt Dashboard
echo ===============================================
echo Lancement de dashboard_qt.py...
echo Une interface Qt native devrait s'ouvrir
echo.

REM Lancement avec gestion d'erreur détaillée
python dashboard\dashboard_qt.py 2>&1
set ERROR_CODE=%ERRORLEVEL%

echo.
echo ===============================================
echo   RESULTAT
echo ===============================================
if %ERROR_CODE% neq 0 (
    echo ❌ Erreur lors du lancement (code: %ERROR_CODE%)
    echo Verifiez les messages d'erreur ci-dessus
) else (
    echo ✅ Qt Dashboard termine normalement
)

echo.
echo Appuyez sur une touche pour continuer...
pause >nul
exit /b %ERROR_CODE%