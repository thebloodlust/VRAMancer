#!/usr/bin/env python3
"""
VRAMancer Standalone Executable Builder
---------------------------------------
Ce script utilise PyInstaller pour compiler VRAMancer en un seul fichier exécutable
(.exe sur Windows, binaire sur Linux/macOS) contenant Python et toutes les dépendances.

Pré-requis:
    pip install pyinstaller

Usage:
    python build_standalone.py
"""

import os
import sys
import subprocess
import platform

def main():
    print("=== VRAMancer Standalone Builder ===")
    
    # Vérifier si PyInstaller est installé
    try:
        import PyInstaller
    except ImportError:
        print("Installation de PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    # Déterminer le nom de l'exécutable selon l'OS
    system = platform.system().lower()
    exe_name = "vramancer"
    if system == "windows":
        exe_name += ".exe"
    elif system == "darwin":
        exe_name += "-macos"
    else:
        exe_name += "-linux"

    print(f"\nCompilation de l'exécutable pour {system.upper()}...")
    
    # Commande PyInstaller
    # --name : nom de l'exécutable
    # --hidden-import : forcer l'inclusion de modules dynamiques (PyTorch, Flask, XPU, NPU etc.)
    # --collect-all : inclure toutes les sous-dépendances des bibliothèques clés
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", exe_name,
    ]

    # Pour les gros projets AI (PyTorch > 2 Go), le mode "onedir" (un dossier) est 
    # souvent recommandé car "--onefile" doit tout désarchiver dans un /tmp à chaque lancement (très lent).
    # Toutefois, vous avez demandé un fichier unique, on tente le onefile par défaut.
    use_onefile = os.environ.get("VRM_BUILD_DIR", "0") != "1"
    if use_onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    cmd.extend([
        "--clean",
        "--hidden-import", "torch",
        "--hidden-import", "flask",
        "--hidden-import", "transformers",
        "--hidden-import", "intel_extension_for_pytorch",  # Pour XPU Intel
        "--hidden-import", "vllm",                         # Backend vLLM
        "--hidden-import", "ollama",                       # Backend Ollama
        "--hidden-import", "pynvml",                       # Pour AMD/NVIDIA (monitoring)
        "--hidden-import", "safetensors",                  # Souvent manquant en HF
        "--hidden-import", "core.production_api",
        "--hidden-import", "core.inference_pipeline",
        "--hidden-import", "core.backends_vllm",
        "--hidden-import", "core.backends_ollama",
        "--hidden-import", "core.backends_webgpu",
        "--collect-all", "core",
        "--collect-all", "torch",
        "--collect-all", "transformers",
        "--collect-data", "flask",
        "vramancer/main.py"
    ])

    print(f"Exécution: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
        print("\n✅ SUCCÈS !")
        print(f"Votre exécutable autonome se trouve dans le dossier 'dist/'.")
        print(f"Fichier généré : dist/{exe_name}")
        print("\nVous pouvez copier ce fichier sur n'importe quelle machine du même OS,")
        print("il fonctionnera sans avoir besoin d'installer Python ou PyTorch !")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERREUR lors de la compilation : {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
