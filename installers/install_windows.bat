
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
REM Ajout dossier racine au PYTHONPATH pour éviter ModuleNotFoundError utils
set PYTHONPATH=%CD%\..;%PYTHONPATH%

echo.
echo Démarrage optionnel de l'API unifiée (port 5030) ?
choice /M "Lancer maintenant l'API backend" /C O/N
if errorlevel 2 goto skip_api
start python -m core.api.unified_api
:skip_api
REM Lancement du systray et de l'interface graphique
cd /d %~dp0
REM Lancement systray et GUI installateur (fichiers à la racine du projet)
if exist ..\..\systray_vramancer.py (
	start python ..\..\systray_vramancer.py
) else if exist ..\systray_vramancer.py (
	start python ..\systray_vramancer.py
)
if exist ..\..\installer_gui.py (
	python ..\..\installer_gui.py
) else if exist ..\installer_gui.py (
	python ..\installer_gui.py
) else (
	echo (INFO) Pas de programme d'installation GUI trouvé (installer_gui.py)
)
REM Lancement auto du nœud et benchmark
echo.
echo 🔎 Benchmark du nœud pour optimisation du cluster...
for %%F in (..\..\scripts\node_benchmark.py ..\scripts\node_benchmark.py) do (
	if exist %%F (
		python %%F & goto nb_done
	)
)
:nb_done
for %%F in (..\..\core\network\cluster_discovery.py ..\core\network\cluster_discovery.py) do (
	if exist %%F (
		python %%F & goto cd_done
	)
)
:cd_done
pause
