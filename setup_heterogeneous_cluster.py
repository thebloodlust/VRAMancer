"""
Guide de déploiement pour configuration hétérogène.
Setup: Serveur EPYC + RTX 3090 + AMD MI50 + Laptop i5 4060Ti + MacBook M4
"""

# Configuration réseau recommandée pour votre setup
NETWORK_CONFIG = {
    "epyc-server": {
        "ip": "192.168.1.100",
        "interfaces": {
            "enp1s0": "10Gbps",  # Interface principale
            "ens18": "1Gbps"     # Proxmox bridge
        },
        "ports": {
            "api": 5030,
            "fastpath": 5040,
            "dashboard": 5000
        }
    },
    "laptop-i5": {
        "ip": "192.168.1.101", 
        "interfaces": {
            "WiFi_6": "1Gbps",
            "Ethernet": "1Gbps"
        },
        "ports": {
            "api": 5031,
            "fastpath": 5041
        }
    },
    "macbook-m4": {
        "ip": "192.168.1.102",
        "interfaces": {
            "en0": "1Gbps",  # WiFi
            "en1": "1Gbps"   # USB-C Ethernet
        },
        "ports": {
            "api": 5032,
            "fastpath": 5042
        }
    }
}

# Variables d'environnement par nœud
ENV_CONFIGS = {
    "epyc-server": {
        "VRM_NODE_ROLE": "master",
        "VRM_CUDA_DEVICES": "0",      # RTX 3090
        "VRM_ROCM_DEVICES": "0",      # MI50
        "VRM_MEMORY_TIER_L1": "cuda:0",
        "VRM_MEMORY_TIER_L2": "rocm:0",
        "VRM_API_PORT": "5030",
        "VRM_FASTPATH_IF": "enp1s0",
        "VRM_MAX_MEMORY_MB": "200000",  # 200GB utilisables
        "VRM_CLUSTER_MASTER": "true"
    },
    "laptop-i5": {
        "VRM_NODE_ROLE": "worker",
        "VRM_CUDA_DEVICES": "0",      # RTX 4060 Ti
        "VRM_MEMORY_TIER_L1": "cuda:0",
        "VRM_API_PORT": "5031",
        "VRM_CLUSTER_MASTER_IP": "192.168.1.100",
        "VRM_MAX_MEMORY_MB": "12000"  # 12GB utilisables
    },
    "macbook-m4": {
        "VRM_NODE_ROLE": "worker",
        "VRM_MPS_DEVICE": "0",        # Apple Silicon
        "VRM_MEMORY_TIER_L1": "mps:0",
        "VRM_API_PORT": "5032", 
        "VRM_CLUSTER_MASTER_IP": "192.168.1.100",
        "VRM_MAX_MEMORY_MB": "16000",  # 16GB utilisables
        "VRM_POWER_PROFILE": "low_power"
    }
}

def generate_docker_compose():
    """Génère docker-compose.yml pour le cluster."""
    return """
version: '3.8'

services:
  vramancer-master:
    build: .
    hostname: epyc-server
    environment:
      - VRM_NODE_ROLE=master
      - VRM_CLUSTER_MASTER=true
      - VRM_API_PORT=5030
      - VRM_CUDA_DEVICES=0
      - VRM_ROCM_DEVICES=0
      - VRM_FASTPATH_IF=enp1s0
    ports:
      - "5030:5030"
      - "5000:5000"  # Dashboard
      - "5040:5040"  # FastPath
    volumes:
      - /dev/nvidia0:/dev/nvidia0
      - /dev/kfd:/dev/kfd
      - ./models:/app/models
    networks:
      - vramancer-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  vramancer-worker-laptop:
    build: .
    hostname: laptop-i5
    environment:
      - VRM_NODE_ROLE=worker
      - VRM_CLUSTER_MASTER_IP=epyc-server
      - VRM_API_PORT=5031
      - VRM_CUDA_DEVICES=0
    ports:
      - "5031:5031"
    volumes:
      - /dev/nvidia0:/dev/nvidia0
    networks:
      - vramancer-net
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  vramancer-worker-mac:
    build: .
    hostname: macbook-m4
    environment:
      - VRM_NODE_ROLE=worker
      - VRM_CLUSTER_MASTER_IP=epyc-server
      - VRM_API_PORT=5032
      - VRM_MPS_DEVICE=0
      - VRM_POWER_PROFILE=low_power
    ports:
      - "5032:5032"
    networks:
      - vramancer-net
    # Note: MPS devices nécessitent configuration spéciale sur macOS

networks:
  vramancer-net:
    driver: bridge
    ipam:
      config:
        - subnet: 192.168.100.0/24
"""

def generate_systemd_services():
    """Services systemd pour démarrage automatique."""
    return {
        "vramancer-master.service": """
[Unit]
Description=VRAMancer Master Node
After=network.target nvidia-persistenced.service

[Service]
Type=simple
User=vramancer
WorkingDirectory=/opt/vramancer
Environment=VRM_NODE_ROLE=master
Environment=VRM_CLUSTER_MASTER=true
Environment=VRM_API_PORT=5030
Environment=VRM_CUDA_DEVICES=0
Environment=VRM_ROCM_DEVICES=0
ExecStart=/opt/vramancer/.venv/bin/python -m vramancer.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
""",
        "vramancer-worker.service": """
[Unit]
Description=VRAMancer Worker Node
After=network.target

[Service]
Type=simple
User=vramancer
WorkingDirectory=/opt/vramancer
Environment=VRM_NODE_ROLE=worker
Environment=VRM_CLUSTER_MASTER_IP=192.168.1.100
Environment=VRM_API_PORT=5031
ExecStart=/opt/vramancer/.venv/bin/python -m vramancer.main
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    }

def generate_installation_script():
    """Script d'installation automatisée."""
    return """#!/bin/bash
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
"""

if __name__ == "__main__":
    import os
    
    # Créer les fichiers de configuration
    os.makedirs("config", exist_ok=True)
    
    # Docker Compose
    with open("config/docker-compose.yml", "w") as f:
        f.write(generate_docker_compose())
    
    # Services systemd
    services = generate_systemd_services()
    for name, content in services.items():
        with open(f"config/{name}", "w") as f:
            f.write(content)
    
    # Script d'installation
    with open("config/install_cluster.sh", "w") as f:
        f.write(generate_installation_script())
    os.chmod("config/install_cluster.sh", 0o755)
    
    # Configs par nœud
    for node, env_vars in ENV_CONFIGS.items():
        with open(f"config/{node}.env", "w") as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
    
    print("✅ Configuration générée dans config/")
    print("📁 Fichiers créés:")
    print("   • docker-compose.yml")
    print("   • install_cluster.sh")
    print("   • *.service (systemd)")
    print("   • *.env (par nœud)")