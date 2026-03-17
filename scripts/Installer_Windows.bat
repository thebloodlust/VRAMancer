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

echo [1/3] Telechargement d'un Python sain...
curl.exe -sL -o "%INSTALL_DIR%\python_installer.exe" "https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe"

echo [2/3] Installation silencieuse en arriere plan...
start /wait "" "%INSTALL_DIR%\python_installer.exe" /quiet InstallAllUsers=0 PrependPath=0 Include_doc=0 Include_tcltk=0 Include_test=0 TargetDir="%PYTHON_DIR%"
del "%INSTALL_DIR%\python_installer.exe"

:RUN_INSTALLER
echo [3/3] Lancement automatique de l'installeur VRAMancer...
echo.
curl.exe -sL -o "%INSTALL_DIR%\vrm_installer.py" "https://raw.githubusercontent.com/thebloodlust/VRAMancer/main/scripts/vramancer_web_installer.py"

"%PYTHON_EXE%" "%INSTALL_DIR%\vrm_installer.py"

echo.
pause