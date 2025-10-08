@echo off
echo ==========================================
echo    GUIDE CUDA WINDOWS - RTX 4060 Ti
echo ==========================================
echo.

echo [ÉTAPE 1] Vérification pilotes NVIDIA existants
echo ----------------------------------------------
echo Ouvrez une invite de commande et tapez :
echo   nvidia-smi
echo.
echo ✅ Si ça fonctionne : notez la version du driver
echo ❌ Si erreur : réinstallez les pilotes GeForce
echo.

echo [ÉTAPE 2] Téléchargement CUDA Toolkit
echo ----------------------------------------------
echo 1. Allez sur : https://developer.nvidia.com/cuda-downloads
echo 2. Sélectionnez :
echo    - Operating System : Windows
echo    - Architecture : x86_64
echo    - Version : 10/11 (votre Windows)
echo    - Installer Type : exe (network) - RECOMMANDÉ
echo.
echo 🎯 Version recommandée : CUDA 12.1 ou 12.4
echo    (Compatible avec PyTorch et RTX 4060 Ti)
echo.

echo [ÉTAPE 3] Installation CUDA Toolkit
echo ----------------------------------------------
echo 1. Lancez le .exe téléchargé EN TANT QU'ADMINISTRATEUR
echo 2. Choisissez "Express Installation" 
echo 3. Laissez les composants par défaut :
echo    ✅ CUDA Toolkit
echo    ✅ CUDA Samples
echo    ✅ CUDA Documentation
echo    ✅ NVIDIA GeForce Experience (si pas déjà installé)
echo.
echo ⏱️  Installation : 5-15 minutes selon connexion
echo.

echo [ÉTAPE 4] Vérification installation
echo ----------------------------------------------
echo Après redémarrage, ouvrez cmd et tapez :
echo   nvcc --version
echo   nvidia-smi
echo.
echo ✅ nvcc doit afficher la version CUDA
echo ✅ nvidia-smi doit montrer votre RTX 4060 Ti
echo.

echo [ÉTAPE 5] Variables d'environnement (automatique)
echo ----------------------------------------------
echo L'installeur ajoute automatiquement :
echo   CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X
echo   PATH += ....\CUDA\vXX.X\bin
echo   PATH += ....\CUDA\vXX.X\libnvvp
echo.
echo 🔍 Pour vérifier : Panneau de configuration ^> Système ^> Variables d'environnement
echo.

echo [ÉTAPE 6] Installation PyTorch CUDA
echo ----------------------------------------------
echo Dans votre environnement Python VRAMancer :
echo   pip uninstall torch torchvision torchaudio
echo   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.
echo 🎯 Remplacez cu121 par cu124 si vous installez CUDA 12.4
echo.

echo [ÉTAPE 7] Test final
echo ----------------------------------------------
echo Lancez Python et testez :
echo   import torch
echo   print(torch.cuda.is_available())        # Doit être True
echo   print(torch.cuda.get_device_name(0))    # Doit afficher RTX 4060 Ti
echo   print(torch.cuda.device_count())        # Doit afficher 1
echo.

echo ==========================================
echo           LIENS UTILES
echo ==========================================
echo 🔗 CUDA Toolkit : https://developer.nvidia.com/cuda-downloads
echo 🔗 PyTorch CUDA : https://pytorch.org/get-started/locally/
echo 🔗 Drivers GeForce : https://www.nvidia.com/fr-fr/drivers/
echo.
echo 💡 CONSEIL : Redémarrez APRÈS l'installation CUDA
echo              avant d'installer PyTorch CUDA
echo.
pause