🎮 VRAMancer
VRAMancer est un outil open‑source conçu pour optimiser l’utilisation de la mémoire vidéo (VRAM) sur des configurations multi‑GPU, même modestes.
Il permet de charger des modèles IA plus volumineux en répartissant intelligemment les blocs mémoire entre plusieurs cartes graphiques.
🚀 Objectif
Faciliter l’exécution locale de modèles LLM ou de génération d’images (type Stable Diffusion) sur des machines avec plusieurs GPU, sans dépendre du cloud ni investir dans du matériel haut de gamme.
📦 Structure du projet
text

Réduire
Enregistrer
Copier
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
VRAMancer/
├── README.md
├── vramancer.py
├── requirements.txt
├── data/
│   └── raw_data.csv
├── models/
│   └── model.pkl
├── utils/
│   └── helpers.py
├── core/
│   ├── model_splitter.py         # Découpe du modèle en blocs
│   ├── memory_balancer.py        # Répartition VRAM entre GPU
│   ├── scheduler.py              # Planification adaptative
│   ├── secure_layer.py           # Chiffrement & authentification
│   └── network/
│       ├── packets.py            # Format des paquets réseau
│       ├── transmission.py       # Logique d’échange inter‑GPU
│       ├── vramancer_link.py     # Protocole custom ultra‑léger
│       ├── cloud_bridge.py       # Extension VRAM via réseau
│       └── sfp_override.py       # Firmware réseau custom
├── premium/
│   ├── auto_tuner.py             # Optimisation dynamique
│   ├── huggingface_bridge.py     # Compatibilité HF
├── dashboard/
│   ├── app.py                    # Interface graphique (Flask ou PyQt)
│   └── visualizer.py             # Visualisation VRAM en temps réel
├── tests/
│   └── test_vramancer.py         # Tests unitaires
🧪 Installation
bash

Réduire
Enregistrer
Copier
1
2
3
git clone https://github.com/tonpseudo/VRAMancer.git
cd VRAMancer
pip install -r requirements.txt
🎯 Fonctionnalités
Répartition intelligente de la VRAM entre plusieurs GPU.
Planification adaptative du chargement des modèles.
Sécurité : chiffrement des échanges réseau, authentification par clé publique.
Interface graphique (Flask / PyQt) pour surveiller la consommation en temps réel.
Export des statistiques en CSV ou JSON.
Support multi‑GPU (CUDA, ROCm, M‑series).
Intégration avec Hugging Face et ComfyUI.
 🔧 Options Premium Disponibles
MODULE
 DESCRIPTION
 VRAMancer Link	
Protocole réseau ultra‑léger (SFP+/Ethernet) pour les échanges inter‑machines.
ZeroStack TCP‑Free Mode	
Bypass complet de la pile TCP/IP pour réduire la latence.
Cloud Fabric Custom	
Stack réseau propriétaire pour cloud distribué.
SFP Protocol Override	
Firmware dédié pour remplacer le protocole Ethernet par un protocole custom.
VRAMancer Memory Sync	
Synchronisation directe des blocs mémoire entre machines.
GPU Direct Dispatch	
Envoi direct de tâches entre GPU distants.
Cloud Bridge	
Extension dynamique de la VRAM via le réseau.
VRAMancer Lite	
Version allégée pour les configurations edge.
Auto Tuner	
Optimisation dynamique des paramètres système et réseau.
Hugging Face Bridge	
Compatibilité native avec les modèles Hugging Face.
Scheduler Intelligent	
Prédiction et préchargement adaptatif des modèles.
Secure Fabric Layer	
Chiffrement natif, authentification, gestion des sessions.
 🤝 Contribution
Les contributions sont les bienvenues !
Ouvrez une issue ou une pull request pour corriger un bug, ajouter une fonctionnalité ou simplement discuter d’une idée.
📜 Licence
MIT — libre d’utilisation, modification et distribution.
