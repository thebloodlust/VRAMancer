@echo off
chcp 65001 > nul
echo =======================================================
echo    VRAMancer - Installation Double Clic (Windows)
echo =======================================================
echo.
echo Initialisation du moteur d'installation infaillible...
echo Merci de patienter (environ 1-2 minutes).
echo.

set INSTALL_DIR=%LOCALAPPDATA%\VRAMancer
set PYTHON_DIR=%INSTALL_DIR%\python_sys
set PYTHON_EXE=%PYTHON_DIR%\python.exe

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

if exist "%PYTHON_EXE%" goto RUN_INSTALLER

echo [1/3] Telechargement de l'environnement Python portable...
curl.exe -sL -o "%INSTALL_DIR%\python.zip" "https://www.python.org/ftp/python/3.11.8/python-3.11.8-embed-amd64.zip"

echo [2/3] Extraction du systeme Python...
if not exist "%PYTHON_DIR%" mkdir "%PYTHON_DIR%"
tar.exe -xf "%INSTALL_DIR%\python.zip" -C "%PYTHON_DIR%"
del "%INSTALL_DIR%\python.zip"

if exist "%PYTHON_DIR%\python311._pth" del "%PYTHON_DIR%\python311._pth"

:RUN_INSTALLER
echo [3/3] Lancement automatique de l'installeur VRAMancer...
echo.
curl.exe -sL -o "%INSTALL_DIR%rm_installer.py" "https://raw.githubusercontent.com/thebloodlust/VRAMancer/main/scripts/vramancer_web_installer.py"

"%PYTHON_EXE%" "%INSTALL_DIR%rm_installer.py"

echo.
pause
