@echo off
title DEBUG API VRAMancer

echo ===============================================
echo   DEBUG API VRAMANCER
===============================================
echo.

echo 1. Test fichier start_api.py...
if exist "start_api.py" (
    echo ✓ start_api.py existe
) else (
    echo ✗ start_api.py MANQUANT
    echo Le fichier start_api.py n'existe pas dans ce repertoire
    pause
    exit /b 1
)

echo.
echo 2. Test port 5030...
netstat -an | findstr ":5030"
if %ERRORLEVEL% equ 0 (
    echo Port 5030 deja utilise par un autre processus
    echo Tentative de liberation...
    for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5030"') do (
        echo Arret processus PID %%p
        taskkill /f /pid %%p
    )
) else (
    echo Port 5030 libre
)

echo.
echo 3. Test lancement API direct...
echo Ceci va lancer l'API en mode visible pour voir les erreurs
echo Fermez cette fenetre et testez localhost:5030 dans votre navigateur
echo.
pause

python start_api.py

echo.
echo API terminee.
pause