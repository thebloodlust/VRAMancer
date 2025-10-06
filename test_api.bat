@echo off
echo ===============================================
echo    VRAMancer - Test API et Resolution
echo ===============================================

set VRM_API_BASE=http://localhost:5030
set VRM_API_PORT=5030

echo Variables definies:
echo VRM_API_BASE=%VRM_API_BASE%
echo VRM_API_PORT=%VRM_API_PORT%
echo.

REM Test si le port est occupé
echo Test du port 5030...
netstat -an | findstr :5030
if errorlevel 1 (
    echo Port 5030 libre
) else (
    echo Port 5030 occupe - arret des processus...
    taskkill /f /fi "PORT eq 5030" 2>nul
)

echo.
echo Demarrage API...
python start_api.py

pause