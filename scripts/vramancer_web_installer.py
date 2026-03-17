#!/usr/bin/env python3
"""
VRAMancer Smart Web Installer (Bootstrapper)
--------------------------------------------
C'est le "Mini-Installeur". Une fois compilé avec PyInstaller, il pèse ~10 Mo.
Quand l'utilisateur lance ce fichier (VRAMancer_Setup.exe / VRAMancer_Setup_Mac),
il agit comme un installeur pro :
1. Il télécharge un environnement Python léger et autonome (Micromamba).
2. Il télécharge la dernière version de VRAMancer depuis GitHub.
3. Il configure les drivers CUDA ou MPS adaptés.
4. Il crée un raccourci sur le Bureau.
"""

import os
import sys
import platform
import subprocess
import urllib.request
import tarfile
import zipfile
import shutil
import time

# --- Configuration ---
APP_NAME = "VRAMancer"
REPO_ZIP_URL = "https://github.com/thebloodlust/VRAMancer/archive/refs/heads/main.zip"

def get_install_dir():
    if platform.system() == "Windows":
        return os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), APP_NAME)
    else:
        return os.path.expanduser(f"~/.{APP_NAME.lower()}")

def get_mamba_url():
    sys_os = platform.system().lower()
    arch = platform.machine().lower()
    
    if sys_os == "windows":
        return "https://micro.mamba.pm/api/micromamba/win-64/latest"
    elif sys_os == "darwin":
        if "arm" in arch or "aarch64" in arch:
            return "https://micro.mamba.pm/api/micromamba/osx-arm64/latest"
        else:
            return "https://micro.mamba.pm/api/micromamba/osx-64/latest"
    else: # Linux
        if "aarch64" in arch or "arm" in arch:
            return "https://micro.mamba.pm/api/micromamba/linux-aarch64/latest"
        else:
            return "https://micro.mamba.pm/api/micromamba/linux-64/latest"

def download_with_progress(url, dest_path):
    print(f"Téléchargement de {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
    except Exception as e:
        print(f"Erreur de téléchargement : {e}")
        sys.exit(1)

def extract_archive(archive_path, extract_to):
    print(f"Extraction de {archive_path}...")
    if archive_path.endswith(".zip"):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith(".tar.bz2") or archive_path.endswith(".tar.gz"):
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)

def create_desktop_shortcut(install_dir, env_dir):
    sys_os = platform.system().lower()
    desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.exists(desktop_dir):
        desktop_dir = os.path.join(os.path.expanduser("~"), "Bureau") # French Mac/Win fallback
    
    if sys_os == "windows":
        # Create a simple .bat shortcut on Windows
        shortcut_path = os.path.join(desktop_dir, f"{APP_NAME}.bat")
        mamba_exe = os.path.join(install_dir, "mamba", "Library", "bin", "micromamba.exe")
        vrm_dir = os.path.join(install_dir, "app", "VRAMancer-main")
        
        with open(shortcut_path, "w") as f:
            f.write(f"@echo off\n")
            f.write(f"cd /d \"{vrm_dir}\"\n")
            f.write(f"\"{mamba_exe}\" run -p \"{env_dir}\" python -m vramancer.main\n")
            f.write("pause\n")
        print(f"✅ Raccourci créé sur le Bureau : {shortcut_path}")
        
    elif sys_os == "darwin" or sys_os == "linux":
        # Create a .command or .desktop file
        shortcut_path = os.path.join(desktop_dir, f"{APP_NAME}.command")
        mamba_exe = os.path.join(install_dir, "mamba", "bin", "micromamba")
        vrm_dir = os.path.join(install_dir, "app", "VRAMancer-main")
        
        with open(shortcut_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"cd \"{vrm_dir}\"\n")
            f.write(f"\"{mamba_exe}\" run -p \"{env_dir}\" python -m vramancer.main\n")
        
        os.chmod(shortcut_path, 0o755)
        print(f"✅ Raccourci exécutable créé sur le Bureau : {shortcut_path}")

def main():
    print(f"==================================================")
    print(f"🚀 VRAMancer Smart Installer - Configuration Rapide")
    print(f"==================================================")
    
    install_dir = get_install_dir()
    print(f"\nDossier d'installation : {install_dir}")
    os.makedirs(install_dir, exist_ok=True)
    
    # 1. Download & Extract Micromamba (Standalone Python)
    mamba_dir = os.path.join(install_dir, "mamba")
    mamba_archive = os.path.join(install_dir, "mamba_archive.tar.bz2")
    
    if not os.path.exists(mamba_dir):
        print("\n--- Étape 1/4 : Installation du moteur Python intégré ---")
        mamba_url = get_mamba_url()
        download_with_progress(mamba_url, mamba_archive)
        os.makedirs(mamba_dir, exist_ok=True)
        extract_archive(mamba_archive, mamba_dir)
        os.remove(mamba_archive)
    else:
        print("\n--- Étape 1/4 : Moteur Python intégré déjà présent.")

    # Locate mamba executable
    if platform.system().lower() == "windows":
        mamba_exe = os.path.join(mamba_dir, "Library", "bin", "micromamba.exe")
    else:
        mamba_exe = os.path.join(mamba_dir, "bin", "micromamba")

    if not os.path.exists(mamba_exe):
        print(f"Erreur fatale : impossible de trouver '{mamba_exe}'.")
        sys.exit(1)
        
    os.chmod(mamba_exe, 0o755) # Ensure executable on unix

    # 2. Create Python Environment & Setup PyTorch
    env_dir = os.path.join(install_dir, "env")
    print("\n--- Étape 2/4 : Création de l'environnement virtuel IA ---")
    if not os.path.exists(env_dir):
        print("Installation de Python 3.10 et des packages de base (Conda-Forge)...")
        subprocess.check_call([mamba_exe, "create", "-p", env_dir, "python=3.10", "-c", "conda-forge", "-y"])
    else:
        print("Environnement déjà créé.")

    # 3. Download VRAMancer Source
    app_dir = os.path.join(install_dir, "app")
    app_zip = os.path.join(install_dir, "vrm_source.zip")
    
    print("\n--- Étape 3/4 : Téléchargement du cœur de VRAMancer ---")
    if os.path.exists(app_dir):
        shutil.rmtree(app_dir)
    os.makedirs(app_dir, exist_ok=True)
    
    download_with_progress(REPO_ZIP_URL, app_zip)
    extract_archive(app_zip, app_dir)
    os.remove(app_zip)

    vrm_main_dir = os.path.join(app_dir, "VRAMancer-main")

    # 4. Install Dependencies
    print("\n--- Étape 4/4 : Installation des dépendances (PyTorch, Transformers...) ---")
    print("Cela peut prendre quelques minutes selon votre connexion internet.")
    
    requirements_file = os.path.join(vrm_main_dir, "requirements-lite.txt") # Use lite for fast install by default
    subprocess.check_call([mamba_exe, "run", "-p", env_dir, "pip", "install", "-r", requirements_file])

    # Install specific torch version depending on OS (for GPU support)
    sys_os = platform.system().lower()
    if sys_os == "darwin":
        # Mac MPS
        subprocess.check_call([mamba_exe, "run", "-p", env_dir, "pip", "install", "torch", "torchvision", "torchaudio"])
    elif sys_os == "windows":
        # Windows CUDA (Assuming CUDA 12.1 for modern RTX cards)
        print("Configuration de NVIDIA CUDA 12.1...")
        subprocess.check_call([mamba_exe, "run", "-p", env_dir, "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"])
    else:
        # Linux Default (CUDA 12.1)
        subprocess.check_call([mamba_exe, "run", "-p", env_dir, "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"])

    # 5. Create Shortcuts
    print("\n--- Finalisation ---")
    create_desktop_shortcut(install_dir, env_dir)
    
    print(f"\n✅ INSTALLATION TERMINÉE AVEC SUCCÈS !")
    print("Vous pouvez maintenant lancer VRAMancer depuis le raccourci sur votre Bureau.")
    print("Le Swarm et l'IA sont prêts à être réveillés.")
    
    time.sleep(5)

if __name__ == "__main__":
    main()
