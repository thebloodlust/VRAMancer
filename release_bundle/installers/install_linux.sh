#!/bin/bash
# Installeur Linux VRAMancer
echo "Installation VRAMancer..."
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
bash Install.sh
python3 installer_gui.py
python3 core/network/cluster_discovery.py
python3 scripts/node_benchmark.py
