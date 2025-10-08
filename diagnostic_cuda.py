#!/usr/bin/env python3
"""Diagnostic CUDA/GPU complet pour RTX 4060 Ti."""

import sys
import os
import subprocess

def run_command(cmd, description):
    """Execute une commande et affiche le résultat."""
    print(f"\n🔍 {description}")
    print("=" * 50)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(f"STDERR: {result.stderr.strip()}")
        if result.returncode != 0:
            print(f"Code retour: {result.returncode}")
    except Exception as e:
        print(f"Erreur: {e}")

def main():
    print("=" * 60)
    print("    DIAGNOSTIC CUDA/GPU - RTX 4060 Ti")
    print("=" * 60)
    
    # 1. Infos système de base
    print(f"\n📊 SYSTÈME:")
    print(f"Python: {sys.version}")
    print(f"Plateforme: {sys.platform}")
    print(f"Architecture: {os.name}")
    
    # 2. Variables d'environnement CUDA
    print(f"\n🔧 VARIABLES ENVIRONNEMENT CUDA:")
    cuda_vars = ['CUDA_PATH', 'CUDA_HOME', 'CUDA_ROOT', 'PATH']
    for var in cuda_vars:
        value = os.environ.get(var, 'Non défini')
        print(f"{var}: {value}")
    
    # 3. Commandes système pour NVIDIA
    commands = [
        ('nvidia-smi', 'NVIDIA System Management Interface'),
        ('nvcc --version', 'NVIDIA CUDA Compiler'),
        ('wmic path win32_VideoController get name', 'Cartes graphiques Windows'),
        ('dxdiag /t dxdiag.txt && type dxdiag.txt', 'DirectX Diagnostic (partiel)'),
    ]
    
    for cmd, desc in commands:
        run_command(cmd, desc)
    
    # 4. Test PyTorch CUDA
    print(f"\n🔥 TEST PYTORCH CUDA:")
    print("=" * 50)
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA disponible: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"Nombre GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
                print(f"  Mémoire: {props.total_memory / 1024**3:.1f} GB")
                print(f"  Compute capability: {props.major}.{props.minor}")
        else:
            print("❌ CUDA non disponible dans PyTorch")
            print("Causes possibles:")
            print("- PyTorch CPU-only installé")
            print("- Drivers NVIDIA manquants")
            print("- CUDA Toolkit non installé")
            print("- Version CUDA incompatible")
            
    except ImportError:
        print("❌ PyTorch non installé")
    except Exception as e:
        print(f"❌ Erreur PyTorch: {e}")
    
    # 5. Test alternative avec core.utils
    print(f"\n⚙️ TEST VRAMANCER CORE.UTILS:")
    print("=" * 50)
    try:
        sys.path.insert(0, os.getcwd())
        from core.utils import detect_backend, enumerate_devices
        
        backend = detect_backend()
        devices = enumerate_devices()
        
        print(f"Backend détecté: {backend}")
        print(f"Nombre devices: {len(devices)}")
        
        for device in devices:
            print(f"- {device['id']}: {device['name']} ({device['backend']})")
            if device.get('total_memory'):
                print(f"  Mémoire: {device['total_memory'] / 1024**3:.1f} GB")
                
    except Exception as e:
        print(f"❌ Erreur core.utils: {e}")
    
    # 6. Recommandations
    print(f"\n💡 RECOMMANDATIONS:")
    print("=" * 50)
    print("Pour RTX 4060 Ti + i5 12e gen:")
    print()
    print("1. DRIVERS NVIDIA:")
    print("   - Installez les derniers drivers GeForce")
    print("   - Redémarrez après installation")
    print()
    print("2. CUDA TOOLKIT:")
    print("   - Téléchargez CUDA 12.x depuis nvidia.com/cuda")
    print("   - Ajoutez au PATH : C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.x\\bin")
    print()
    print("3. PYTORCH CUDA:")
    print("   - Désinstallez : pip uninstall torch")
    print("   - Réinstallez CUDA : pip install torch --index-url https://download.pytorch.org/whl/cu121")
    print()
    print("4. VÉRIFICATION:")
    print("   - Redémarrez l'ordinateur")
    print("   - Relancez ce diagnostic")
    print("   - nvidia-smi doit afficher la RTX 4060 Ti")

if __name__ == "__main__":
    main()