#!/bin/bash
# Installation VRAMancer pour cluster hétérogène

set -e

echo "🚀 Installation VRAMancer - Cluster Hétérogène"

# Détection du nœud
if [[ "$HOSTNAME" == *"epyc"* ]] || [[ "$1" == "master" ]]; then
    NODE_TYPE="master"
    echo "📊 Configuration: Nœud Master (EPYC + RTX 3090 + MI50)"
elif [[ "$HOSTNAME" == *"laptop"* ]] || [[ "$1" == "laptop" ]]; then
    NODE_TYPE="laptop"
    echo "💻 Configuration: Laptop Worker (i5 + RTX 4060 Ti)"
elif [[ "$HOSTNAME" == *"macbook"* ]] || [[ "$1" == "mac" ]]; then
    NODE_TYPE="mac"
    echo "🍎 Configuration: MacBook Worker (M4)"
else
    echo "❓ Type de nœud non détecté. Usage: $0 [master|laptop|mac]"
    exit 1
fi

# Installation des dépendances système
case "$OSTYPE" in
    linux*)
        echo "🐧 Installation Linux..."
        sudo apt update
        sudo apt install -y python3 python3-pip python3-venv git
        
        if [[ "$NODE_TYPE" == "master" ]]; then
            # Drivers NVIDIA + AMD ROCm
            sudo apt install -y nvidia-driver-470 rocm-dev
        elif [[ "$NODE_TYPE" == "laptop" ]]; then
            # Drivers NVIDIA uniquement
            sudo apt install -y nvidia-driver-470
        fi
        ;;
    darwin*)
        echo "🍎 Installation macOS..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        brew install python3 git
        ;;
    msys*|cygwin*|mingw*)
        echo "🪟 Installation Windows..."
        # Chocolatey ou manual install
        ;;
esac

# Clone et installation VRAMancer
cd /opt
sudo git clone https://github.com/thebloodlust/VRAMancer.git vramancer
cd vramancer
sudo chown -R $USER:$USER .

# Environment virtuel
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configuration par nœud
case "$NODE_TYPE" in
    master)
        cp config/epyc-server.env .env
        echo "VRM_CLUSTER_MASTER=true" >> .env
        ;;
    laptop)
        cp config/laptop-i5.env .env
        echo "VRM_CLUSTER_MASTER_IP=192.168.1.100" >> .env
        ;;
    mac)
        cp config/macbook-m4.env .env
        echo "VRM_CLUSTER_MASTER_IP=192.168.1.100" >> .env
        ;;
esac

# Service systemd (Linux)
if [[ "$OSTYPE" == "linux"* ]]; then
    sudo cp config/vramancer-$NODE_TYPE.service /etc/systemd/system/
    sudo systemctl enable vramancer-$NODE_TYPE
    sudo systemctl start vramancer-$NODE_TYPE
fi

echo "✅ Installation terminée!"
echo "🌐 Dashboard: http://$(hostname -I | awk '{print $1}'):5000"
echo "📊 API: http://$(hostname -I | awk '{print $1}'):503X/api/health"
