@echo off
chcp 65001 > nul
set DIR=%LOCALAPPDATA%\VRAMancer
set PYEXE=%DIR%\python_sys\python.exe

echo [1/3] Telechargement de get-pip...
curl.exe -sL -o "%DIR%\get-pip.py" "https://bootstrap.pypa.io/get-pip.py"

echo [2/3] Installation de pip...
"%PYEXE%" "%DIR%\get-pip.py"

echo [3/3] Installation des dependances VRAMancer...
cd "%DIR%\app\VRAMancer-main"
"%PYEXE%" -m pip install -r requirements.txt
"%PYEXE%" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Termine ! Lancement de VRAMancer...
"%PYEXE%" -m vramancer.main serve --host 0.0.0.0 --port 5031
pause
