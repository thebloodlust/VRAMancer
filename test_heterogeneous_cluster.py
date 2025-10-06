#!/usr/bin/env python3
"""
Script de test et configuration pour le cluster hétérogène de l'utilisateur
Configuration spécifique: EPYC + RTX3090 + MI50 + laptop i5 4060Ti + MacBook M4
"""

import os
import sys
import json
import platform
import subprocess
from pathlib import Path

class HeterogeneousClusterTester:
    def __init__(self):
        self.cluster_config = {
            "master_node": {
                "type": "server",
                "hardware": "EPYC 7xx3 + RTX 3090 + AMD MI50",
                "ram": "256GB",
                "os": "Ubuntu (Proxmox)",
                "role": "master",
                "backends": ["cuda", "rocm", "cpu"],
                "ip": "192.168.1.100",  # À adapter
                "priority": 1
            },
            "laptop_node": {
                "type": "laptop", 
                "hardware": "i5-12xxx + RTX 4060Ti",
                "ram": "16-32GB",
                "os": "Windows 11",
                "role": "worker",
                "backends": ["cuda", "cpu"],
                "ip": "192.168.1.101",  # À adapter
                "priority": 2
            },
            "macbook_node": {
                "type": "edge",
                "hardware": "Apple M4",
                "ram": "32GB+",
                "os": "macOS",
                "role": "edge",
                "backends": ["mps", "cpu"],
                "ip": "192.168.1.102",  # À adapter  
                "priority": 3
            }
        }
        
        self.current_node = self.detect_current_node()
    
    def detect_current_node(self):
        """Détecte automatiquement le nœud actuel"""
        system = platform.system()
        processor = platform.processor()
        
        if system == "Windows":
            return "laptop_node"
        elif system == "Darwin":  # macOS
            return "macbook_node" 
        elif system == "Linux":
            return "master_node"
        else:
            return "unknown"
    
    def test_gpu_backends(self):
        """Teste les backends GPU disponibles sur ce nœud"""
        print(f"🔍 Test des backends GPU sur {self.current_node}...")
        
        backends_found = []
        
        # Test CUDA
        try:
            import torch
            if torch.cuda.is_available():
                backends_found.append("cuda")
                print(f"✅ CUDA disponible: {torch.cuda.get_device_name()}")
                print(f"   - Devices: {torch.cuda.device_count()}")
                print(f"   - Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
        except Exception as e:
            print(f"❌ CUDA non disponible: {e}")
        
        # Test ROCm (AMD)
        try:
            if hasattr(torch, 'version') and hasattr(torch.version, 'hip'):
                backends_found.append("rocm")
                print("✅ ROCm disponible")
        except:
            pass
        
        # Test Apple MPS
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                backends_found.append("mps")
                print("✅ Apple MPS disponible")
        except:
            pass
        
        # CPU toujours disponible
        backends_found.append("cpu")
        print("✅ CPU backend disponible")
        
        return backends_found
    
    def generate_node_config(self):
        """Génère la configuration pour ce nœud"""
        node_info = self.cluster_config.get(self.current_node, {})
        backends = self.test_gpu_backends()
        
        config = {
            "node_id": self.current_node,
            "node_type": node_info.get("type", "unknown"),
            "role": node_info.get("role", "worker"),
            "backends": backends,
            "capabilities": {
                "compute_score": self.calculate_compute_score(backends),
                "memory_score": self.calculate_memory_score(),
                "network_score": 80  # Par défaut
            },
            "cluster": {
                "master_ip": self.cluster_config["master_node"]["ip"],
                "discovery_port": 8899,
                "api_port": 5030
            }
        }
        
        return config
    
    def calculate_compute_score(self, backends):
        """Calcule le score de calcul selon les backends"""
        scores = {
            "cuda": 90,  # RTX 3090/4060Ti
            "rocm": 85,  # MI50  
            "mps": 75,   # Apple M4
            "cpu": 30    # Fallback CPU
        }
        
        return max([scores.get(b, 0) for b in backends])
    
    def calculate_memory_score(self):
        """Calcule le score mémoire"""
        try:
            import psutil
            total_ram = psutil.virtual_memory().total // (1024**3)
            
            if total_ram >= 200:  # EPYC 256GB
                return 100
            elif total_ram >= 30:  # MacBook/Laptop 32GB
                return 70
            elif total_ram >= 15:  # Laptop 16GB
                return 50
            else:
                return 30
        except:
            return 50
    
    def setup_cluster_node(self):
        """Configure ce nœud pour le cluster"""
        print(f"⚙️ Configuration du nœud {self.current_node}...")
        
        config = self.generate_node_config()
        
        # Sauvegarder la config
        os.makedirs("config", exist_ok=True)
        with open(f"config/node_{self.current_node}.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration sauvée: config/node_{self.current_node}.json")
        
        # Créer le script de lancement pour ce nœud
        self.create_node_launcher(config)
        
        return config
    
    def create_node_launcher(self, config):
        """Crée un script de lancement spécifique au nœud"""
        role = config["role"]
        
        if role == "master":
            script_content = f'''#!/bin/bash
# Lanceur nœud maître EPYC
echo "🚀 Démarrage nœud maître EPYC..."

# Variables d'environnement
export CUDA_VISIBLE_DEVICES=0,1  # RTX3090 + MI50
export VRM_NODE_ROLE=master
export VRM_API_PORT=5030

# Lancer l'orchestrateur hétérogène
python core/orchestrator/heterogeneous_manager.py --role master &

# Lancer l'API unifiée
python -m core.api.unified_api &

# Lancer le dashboard web
python dashboard/dashboard_web.py &

# Lancer le monitoring mobile
python mobile/dashboard_heterogeneous.py &

echo "✅ Nœud maître démarré"
echo "   - API: http://localhost:5030"
echo "   - Dashboard: http://localhost:5000"  
echo "   - Mobile: http://localhost:8080"

wait
'''
        
        elif role == "worker":
            master_ip = config["cluster"]["master_ip"]
            script_content = f'''@echo off
title VRAMancer Worker Node - Laptop i5 4060Ti
echo 🚀 Demarrage noeud worker laptop...

REM Variables d'environnement Windows
set CUDA_VISIBLE_DEVICES=0
set VRM_NODE_ROLE=worker
set VRM_MASTER_IP={master_ip}

REM Lancer le worker
python core/orchestrator/heterogeneous_manager.py --role worker --master-ip {master_ip}

echo ✅ Noeud worker demarre et connecte au maitre
pause
'''
        
        elif role == "edge":
            master_ip = config["cluster"]["master_ip"]
            script_content = f'''#!/bin/bash
# Lanceur nœud edge MacBook M4
echo "🚀 Démarrage nœud edge MacBook M4..."

# Variables d'environnement macOS
export VRM_NODE_ROLE=edge
export VRM_MASTER_IP={master_ip}
export VRM_BACKEND=mps

# Lancer le nœud edge
python core/orchestrator/heterogeneous_manager.py --role edge --master-ip {master_ip} --backend mps

echo "✅ Nœud edge MacBook connecté au cluster"
'''
        
        # Sauvegarder le script
        if role == "master":
            filename = "start_master_node.sh"
        elif role == "worker":
            filename = "start_worker_node.bat"
        else:  # edge
            filename = "start_edge_node.sh"
        
        with open(filename, "w") as f:
            f.write(script_content)
        
        # Rendre exécutable sur Unix
        if filename.endswith(".sh"):
            os.chmod(filename, 0o755)
        
        print(f"✅ Script de lancement créé: {filename}")
    
    def test_cluster_connectivity(self):
        """Teste la connectivité du cluster"""
        print("🌐 Test de connectivité cluster...")
        
        master_ip = self.cluster_config["master_node"]["ip"]
        
        # Test ping
        try:
            if platform.system() == "Windows":
                result = subprocess.run(["ping", "-n", "1", master_ip], 
                                      capture_output=True, text=True)
            else:
                result = subprocess.run(["ping", "-c", "1", master_ip], 
                                      capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Connectivité réseau OK vers {master_ip}")
            else:
                print(f"❌ Pas de connectivité vers {master_ip}")
                print("💡 Vérifiez l'IP du nœud maître dans la config")
        except Exception as e:
            print(f"⚠️ Test ping impossible: {e}")
        
        # Test ports API
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex((master_ip, 5030))
            sock.close()
            
            if result == 0:
                print(f"✅ API maître accessible sur {master_ip}:5030")
            else:
                print(f"❌ API maître inaccessible sur {master_ip}:5030")
        except Exception as e:
            print(f"⚠️ Test API impossible: {e}")
    
    def show_cluster_status(self):
        """Affiche l'état du cluster"""
        print("\n" + "="*60)
        print("🏁 Configuration Cluster Hétérogène VRAMancer")
        print("="*60)
        
        for node_name, node_info in self.cluster_config.items():
            status = "🟢 ACTUEL" if node_name == self.current_node else "⚪ AUTRE"
            print(f"\n{status} {node_info['type'].upper()} ({node_name}):")
            print(f"   Hardware: {node_info['hardware']}")
            print(f"   RAM: {node_info['ram']}")
            print(f"   OS: {node_info['os']}")
            print(f"   Role: {node_info['role']}")
            print(f"   Backends: {', '.join(node_info['backends'])}")
            print(f"   IP: {node_info['ip']}")
        
        print(f"\n🎯 Architecture:")
        print(f"   - Nœud Maître: EPYC (orchestration + calcul lourd)")
        print(f"   - Nœud Worker: Laptop (calcul intermédiaire)")  
        print(f"   - Nœud Edge: MacBook (tâches légères + mobile)")
        
        print(f"\n🔄 Auto-sensing des performances:")
        print(f"   - Détection automatique CUDA/ROCm/MPS")
        print(f"   - Équilibrage selon compute/memory scores")
        print(f"   - Failover automatique maître-esclave")
    
    def run_full_test(self):
        """Lance tous les tests"""
        print("🧪 Test complet du cluster hétérogène VRAMancer")
        print("="*60)
        
        # 1. Détection du nœud
        print(f"\n1️⃣ Nœud détecté: {self.current_node}")
        
        # 2. Test des backends
        print(f"\n2️⃣ Test des backends GPU:")
        backends = self.test_gpu_backends()
        
        # 3. Configuration du nœud
        print(f"\n3️⃣ Configuration du nœud:")
        config = self.setup_cluster_node()
        
        # 4. Test connectivité
        print(f"\n4️⃣ Test connectivité:")
        self.test_cluster_connectivity()
        
        # 5. Statut cluster
        print(f"\n5️⃣ Statut du cluster:")
        self.show_cluster_status()
        
        # 6. Prochaines étapes
        print(f"\n6️⃣ Prochaines étapes:")
        
        if self.current_node == "master_node":
            print("   - Lancez: ./start_master_node.sh")
            print("   - Dashboard: http://localhost:5000")
            print("   - Mobile: http://localhost:8080")
        
        elif self.current_node == "laptop_node":
            print("   - Lancez: start_worker_node.bat")
            print("   - Vérifiez la connexion au maître")
        
        elif self.current_node == "macbook_node":
            print("   - Lancez: ./start_edge_node.sh")
            print("   - Backend MPS sera utilisé")
        
        print(f"\n💡 Astuce: Configuration sauvée dans config/node_{self.current_node}.json")

def main():
    """Point d'entrée principal"""
    tester = HeterogeneousClusterTester()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--backends-only":
            tester.test_gpu_backends()
        elif sys.argv[1] == "--config-only":
            tester.setup_cluster_node()
        elif sys.argv[1] == "--status":
            tester.show_cluster_status()
        else:
            print("Options: --backends-only, --config-only, --status")
    else:
        tester.run_full_test()

if __name__ == "__main__":
    main()