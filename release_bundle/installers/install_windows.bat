@echo off
REM Installeur Windows VRAMancer
python -m pip install --upgrade pip
pip install -r ..\requirements.txt
python ..\installer_gui.py
python ..\scripts\node_benchmark.py
python ..\core\network\cluster_discovery.py
pause
