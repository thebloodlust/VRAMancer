#!/bin/bash
# Installeur Linux VRAMancer
echo "Installation VRAMancer..."
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
bash Install.sh
python3 installer_gui.py
# Lancement auto du nœud
python3 core/network/cluster_discovery.py
# Benchmark automatique du nœud
echo "\n🔎 Benchmark du nœud pour optimisation du cluster..."
python3 scripts/node_benchmark.py
