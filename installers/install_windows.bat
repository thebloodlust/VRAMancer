@echo off
REM Installeur Windows VRAMancer
python -m pip install --upgrade pip
pip install -r ..\requirements.txt
python ..\installer_gui.py
REM Lancement auto du nœud
REM Benchmark automatique du nœud
echo.
echo 🔎 Benchmark du nœud pour optimisation du cluster...
python ..\scripts\node_benchmark.py
python ..\core\network\cluster_discovery.py
pause
