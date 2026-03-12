#!/bin/bash
# VRAMancer - Easy One-Click Setup & Launcher (Linux / macOS)
# Based on experimental/rust-core

echo -e "\033[36m======================================================================\033[0m"
echo -e "\033[36m🚀  VRAMancer - The Heterogeneous AI Swarm                             \033[0m"
echo -e "\033[36m======================================================================\033[0m"
echo ">> Initialisation de l'environnement de production AI..."

# 1. Vérification de Python 3.10+
if ! command -v python3 &> /dev/null; then
    echo "❌ Erreur : Python3 n'est pas installé. Veuillez l'installer."
    exit 1
fi

# 2. Vérification et Installation de RUST
if ! command -v cargo &> /dev/null; then
    echo "⚙️  Rust (Cargo) n'est pas détecté. Installation de Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    echo "✅ Rust (Cargo) détecté."
fi

# 3. Environnement Virtuel Python
if [ ! -d "venv" ]; then
    echo "🐍 Création de l'environnement virtuel Python..."
    python3 -m venv venv
fi
source venv/bin/activate

# 4. Installation des dépendances et du Module Rust (Maturin)
echo "📦 Installation des dépendances (PyTorch, vLLM, Maturin)..."
pip install -r requirements.txt --quiet
pip install maturin --quiet

echo "⚙️  Compilation du Cœur Haute-Performance (Rust ReBAR Bypass)..."
cd rust_core && maturin develop --release && cd ..

echo -e "\033[32m✅ Build Teriné avec Succès !\033[0m\n"

# 5. Dashboard Terminal & Hub LLM
echo -e "\033[35m[ 📚 HUB DES MODÈLES (HuggingFace) ]\033[0m"
echo "Téléchargez manuellement ou laissez VRAMancer autolocaliser :"
echo " - Modèles Meta Llama 3 (8B / 70B) : https://huggingface.co/meta-llama"
echo " - Poids Quantifiés et Légers (GGUF, AWQ, GPTQ) :"
echo "   > TheBloke : https://huggingface.co/TheBloke"
echo "   > MaziyarPanahi : https://huggingface.co/MaziyarPanahi"
echo ""
echo -e "\033[33m[ ⚙️  CONFIGURATIONS EXPERT ]\033[0m"
echo "- Pour activer le Bypass Nvidia Complet sur Proxmox/Linux Host :"
echo "  Ajoutez 'pcie_acs_override=downstream,multifunction' à votre GRUB."
echo "- Pour passer en mode Swarm IP Distant, ouvrez vos ports (ex: 9108, 9109)."
echo -e "\033[36m======================================================================\033[0m"

# Lancement
echo "🔥 Démarrage du Nœud VRAMancer..."
python core/production_api.py
