#!/usr/bin/env bash
# VRAMancer Linux/Mac One-Click Swarm Node Installer

set -e

# Couleurs pour faire joli
CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
cat << "EOF"
   __      __  _____       _      __  __                                 
   \ \    / / |  __ \     / \    |  \/  |                                
    \ \  / /  | |__) |   / _ \   | \  / |  __ _   _ __     ___    ___   _ __ 
     \ \/ /   |  _  /   / ___ \  | |\/| | / _` | | '_ \   / __|  / _ \ | '__|
      \  /    | | \ \  / ___ \ | |  | | | (_| | | | | | | (__  |  __/ | |   
       \/     |_|  \_\/_/   \_\|_|  |_|  \__,_| |_| |_|  \___|  \___| |_|   
EOF
echo -e "${NC}"
echo -e "${CYAN}==============================================================================${NC}"
echo -e "${GREEN}      Bienvenue dans l'Installeur VRAMancer Node (Zero-Config) Linux/Mac${NC}"
echo -e "${CYAN}==============================================================================${NC}\n"

echo -e "${YELLOW}[1] Vérification de l'environnement Python...${NC}"
if ! command -v python3 &> /dev/null
then
    echo -e "${RED}[ERREUR] python3 n'est pas trouvé. Veuillez l'installer (ex: sudo apt install python3 python3-venv).${NC}"
    exit 1
fi

echo -e "${YELLOW}[2] Création de l'environnement virtuel isolé (Mycelium Env)...${NC}"
if [ ! -d "venv_vramancer" ]; then
    python3 -m venv venv_vramancer
    echo -e "${GREEN}Environnement créé avec succès !${NC}"
else
    echo "L'environnement existe déjà."
fi

echo -e "${YELLOW}[3] Activation de l'environnement et installation des dépendances...${NC}"
source venv_vramancer/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install rich -q

# Pour tester la présence de Rust/Cargo et builder dynamiquement le data plane
if command -v cargo &> /dev/null; then
    echo -e "${GREEN}Rust détecté. Compilation du Data Plane Zero-Copy natif...${NC}"
    cd rust_core && pip install maturin && maturin develop --release && cd ..
else
    echo -e "${YELLOW}Rust n'est pas installé localement. VRAMancer utilisera son Fallback Python (un peu moins rapide mais tout aussi stable).${NC}"
fi

echo -e "\n${CYAN}==============================================================================${NC}"
echo -e "${GREEN}                        INSTALLATION TERMINEE AVEC SUCCES${NC}"
echo -e "${CYAN}==============================================================================${NC}\n"

echo -e "${YELLOW}Génération de votre clé P2P Swarm unique...${NC}"
python3 -c "from vramancer.cli.swarm_cli import ui_auth_generate; ui_auth_generate()"

echo -e "\n${YELLOW}[4] Démarrage du noeud VRAMancer...${NC}"
echo -e "Le nœud écoute désormais les paquets P2P ! Vous faites partie de l'essaim."
echo -e "Laissez ce terminal ouvert.\n"

python3 vramancer/main.py serve --backend auto
