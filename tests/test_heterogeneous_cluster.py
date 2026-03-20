"""
Test de configuration hétérogène pour VRAMancer.
Simule votre setup: RTX 3090 + AMD MI50 + i5 4060Ti + MacBook M4.
"""
import time
import json
from core.orchestrator.heterogeneous_manager import HeterogeneousManager, NodeCapabilities

def simulate_your_cluster():
    """Simule votre configuration de test hétérogène."""
    hetero_mgr = HeterogeneousManager()
    
    # 1. Serveur EPYC + RTX 3090 + AMD MI50 (Proxmox Ubuntu)
    server_caps = NodeCapabilities(
        hostname="epyc-server",
        architecture="x86_64", 
        platform="Linux",
        cpu_model="AMD EPYC 7xxx Series",
        cpu_cores=32,
        ram_gb=256.0,
        gpus=[
            {
                'id': 'cuda:0',
                'backend': 'cuda', 
                'name': 'NVIDIA GeForce RTX 3090',
                'memory_mb': 24576,  # 24GB
                'architecture': 'Ampere'
            },
            {
                'id': 'rocm:0',
                'backend': 'rocm',
                'name': 'AMD Instinct MI50',
                'memory_mb': 32768,  # 32GB HBM2e
                'architecture': 'GCN 5'
            }
        ],
        primary_backend="cuda",  # RTX 3090 principale
        network_interfaces=["enp1s0", "ens18"],  # Proxmox interfaces
        bandwidth_mbps=10000.0,  # 10Gbps
        latency_ms=0.5,
        is_edge=False,
        power_profile="high_perf",
        compute_score=9.5,  # Très puissant
        memory_score=9.8,   # 256GB + 56GB VRAM
        network_score=9.0   # 10Gbps
    )
    
    # 2. PC Portable i5 + RTX 4060 Ti
    laptop_caps = NodeCapabilities(
        hostname="laptop-i5",
        architecture="x86_64",
        platform="Windows",  # ou Linux
        cpu_model="Intel Core i5-12400",
        cpu_cores=6,
        ram_gb=16.0,
        gpus=[
            {
                'id': 'cuda:0',
                'backend': 'cuda',
                'name': 'NVIDIA GeForce RTX 4060 Ti',
                'memory_mb': 16384,  # 16GB
                'architecture': 'Ada Lovelace'
            }
        ],
        primary_backend="cuda",
        network_interfaces=["WiFi_6", "Ethernet"],
        bandwidth_mbps=1000.0,  # Gigabit
        latency_ms=2.0,
        is_edge=False,
        power_profile="standard",
        compute_score=5.5,  # Bon milieu de gamme
        memory_score=4.2,   # 16GB + 16GB VRAM
        network_score=6.0
    )
    
    # 3. MacBook M4 (Apple Silicon)
    macbook_caps = NodeCapabilities(
        hostname="macbook-m4",
        architecture="arm64",
        platform="Darwin",
        cpu_model="Apple M4",
        cpu_cores=10,  # 4P + 6E cores
        ram_gb=24.0,   # Configuration probable
        gpus=[
            {
                'id': 'mps:0',
                'backend': 'mps',
                'name': 'Apple M4 GPU',
                'memory_mb': 0,  # Mémoire unifiée
                'architecture': 'Apple Silicon'
            }
        ],
        primary_backend="mps",
        network_interfaces=["en0", "en1"],  # WiFi + Ethernet
        bandwidth_mbps=1000.0,
        latency_ms=1.5,
        is_edge=False,
        power_profile="low_power",  # Efficience énergétique
        compute_score=6.0,  # Très efficace
        memory_score=6.0,   # Mémoire unifiée
        network_score=7.0
    )
    
    # Enregistrement des nœuds
    hetero_mgr.register_node(server_caps)
    hetero_mgr.register_node(laptop_caps)
    hetero_mgr.register_node(macbook_caps)
    
    return hetero_mgr

def test_heterogeneous_workload():
    """Test de répartition de charge hétérogène."""
    print(" Simulation cluster hétérogène VRAMancer")
    print("=" * 60)
    
    hetero_mgr = simulate_your_cluster()
    
    # Résumé du cluster
    summary = hetero_mgr.get_cluster_summary()
    print(f" Résumé Cluster:")
    print(f"   • Nœuds: {summary['nodes']}")
    print(f"   • GPUs total: {summary['total_gpus']}")
    print(f"   • RAM totale: {summary['total_ram_gb']:.0f} GB")
    print(f"   • Backends: {summary['backends']}")
    print(f"   • Architectures: {summary['architectures']}")
    print()
    
    # Détails des nœuds
    print("️  Détails des Nœuds:")
    for hostname, caps in hetero_mgr.nodes.items():
        print(f"   {hostname}:")
        print(f"      • {caps.architecture} / {caps.platform}")
        print(f"      • CPU: {caps.cpu_cores} cores ({caps.cpu_model})")
        print(f"      • RAM: {caps.ram_gb:.0f} GB")
        print(f"      • GPUs: {len(caps.gpus)} ({caps.primary_backend})")
        for gpu in caps.gpus:
            vram_gb = gpu['memory_mb'] / 1024 if gpu['memory_mb'] else 0
            print(f"         - {gpu['name']} ({vram_gb:.0f}GB)")
        print(f"      • Score compute: {caps.compute_score:.1f}")
        print(f"      • Profile: {caps.power_profile}")
        print()
    
    # Test de tâches diverses
    tasks = [
        {
            'id': 'llm_inference_large',
            'memory_mb': 40000,  # 40GB requis
            'compute_score': 7.0,
            'backend': 'any'
        },
        {
            'id': 'training_medium', 
            'memory_mb': 16000,  # 16GB
            'compute_score': 5.0,
            'backend': 'cuda'
        },
        {
            'id': 'inference_edge',
            'memory_mb': 4000,   # 4GB
            'compute_score': 2.0,
            'backend': 'any'
        },
        {
            'id': 'apple_ml_task',
            'memory_mb': 8000,   # 8GB
            'compute_score': 3.0,
            'backend': 'mps'
        }
    ]
    
    print("Répartition des Tâches:")
    allocation = hetero_mgr.balance_load(tasks)
    
    for hostname, assigned_tasks in allocation.items():
        if assigned_tasks:
            print(f"   {hostname}:")
            for task in assigned_tasks:
                mem_gb = task['memory_mb'] / 1024
                print(f"      • {task['id']} ({mem_gb:.1f}GB, score {task['compute_score']})")
        else:
            print(f"   {hostname}: (aucune tâche)")
    
    print()
    print(" Test terminé! Votre configuration est optimale pour:")
    print("   • Serveur EPYC: Tâches lourdes (LLM géants, training)")
    print("   • Laptop i5: Inférence rapide, développement")  
    print("   • MacBook M4: Tâches efficaces, mobile, prototypage")

if __name__ == "__main__":
    test_heterogeneous_workload()