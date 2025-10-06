@echo off
title VRAMancer - API Permanente

echo ===============================================
echo   VRAMANCER - API PERMANENTE
echo ===============================================
echo Ce script lance l'API et la garde en vie
echo.

:restart
echo Nettoyage port 5030...
for /f "tokens=5" %%p in ('netstat -aon ^| findstr ":5030" 2^>nul') do taskkill /f /pid %%p >nul 2>&1

echo.
echo Lancement API VRAMancer...
echo IMPORTANT: Ne fermez pas cette fenetre
echo.

REM Lancement API avec relance automatique
python start_api.py

echo.
echo ===============================================
echo   API ARRETEE - RELANCE DANS 5 SECONDES
echo ===============================================
echo Si vous voulez arreter definitivement, fermez cette fenetre
timeout /t 5 /nobreak >nul
goto restart