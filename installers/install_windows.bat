
@echo off
REM Installeur Windows VRAMancer
cd /d %~dp0
REM Vérifie que Python est installé
where python >nul 2>nul
if errorlevel 1 (
	echo Python n'est pas installé. Veuillez installer Python 3.x.
	pause
	exit /b
)
REM Mise à jour pip et installation des dépendances
python -m pip install --upgrade pip
if exist ..\requirements.txt (
	pip install -r ..\requirements.txt
) else (
	pip install -r ..\..\requirements.txt
)
REM Installation explicite de PyQt5
pip install pyqt5
REM Lancement du systray et de l'interface graphique
cd /d %~dp0
if exist ..\systray_vramancer.py (
	start python ..\systray_vramancer.py
)
if exist ..\installer_gui.py (
	python ..\installer_gui.py
) else (
	python ..\..\installer_gui.py
)
REM Lancement auto du nœud et benchmark
echo.
echo 🔎 Benchmark du nœud pour optimisation du cluster...
if exist ..\scripts\node_benchmark.py (
	python ..\scripts\node_benchmark.py
) else (
	python ..\..\scripts\node_benchmark.py
)
if exist ..\core\network\cluster_discovery.py (
	python ..\core\network\cluster_discovery.py
) else (
	python ..\..\core\network\cluster_discovery.py
)
pause
