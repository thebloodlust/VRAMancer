@echo off
title VRAMancer - AI Swarm Launcher
color 0B

echo ======================================================================
echo 🚀  VRAMancer - The Heterogeneous AI Swarm
echo ======================================================================
echo ^> Initialisation de l'environnement de production AI Windows...

REM 1. Verification de Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Erreur : Python n'est pas installe. Veuillez l'installer depuis python.org.
    pause
    exit /b 1
)

REM 2. Verification de Rust
cargo --version >nul 2>&1
if errorlevel 1 (
    echo ⚙️ Rust (Cargo) n'est pas detecte.
    echo Veuillez installer Rust depuis: https://rustup.rs/ puis relancer ce script.
    pause
    exit /b 1
)
echo ✅ Rust (Cargo) detecte.

REM 3. Environnement virtuel
if not exist "venv\" (
    echo 🐍 Creation de l'environnement virtuel Python...
    python -m venv venv
)
call venv\Scripts\activate.bat

REM 4. Dependencies & Compilation
echo 📦 Installation des dependances (PyTorch, vLLM, Maturin)...
pip install -r requirements-windows.txt -q
pip install maturin -q

echo ⚙️ Compilation du Cœur Haute-Performance (Rust P2P Bypass)...
cd rust_core
maturin develop --release
cd ..

echo ✅ Build Termine avec Succes !
echo.

REM 5. Menu Dashboard
echo [35m[ 📚 HUB DES MODELES (HuggingFace) ][0m
echo Telechargez manuellement ou laissez VRAMancer autolocaliser :
echo  - Modeles Meta Llama 3 (8B / 70B) : https://huggingface.co/meta-llama
echo  - Poids Quantifies et Legers (GGUF, AWQ, GPTQ) :
echo    ^> TheBloke : https://huggingface.co/TheBloke
echo    ^> MaziyarPanahi : https://huggingface.co/MaziyarPanahi
echo.
echo [33m[ ⚙️ CONFIGURATIONS EXPERT ][0m
echo - Sur Windows, l'IOMMU/ReBAR se gere directement via le BIOS de votre carte mere.
echo - Autorisez Python à passer le Pare-feu Windows Defender pour le Swarm IP Distant.
echo ======================================================================

echo 🔥 Demarrage du Nœud VRAMancer Windows...
python core\production_api.py

pause
