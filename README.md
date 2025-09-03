# 🎮 VRAMancer

**VRAMancer** est un outil open source conçu pour optimiser l’utilisation de la mémoire vidéo (VRAM) sur des configurations multi-GPU, même modestes. Il permet de charger des modèles IA plus volumineux en répartissant intelligemment les blocs mémoire entre plusieurs cartes graphiques.

---

## 🚀 Objectif

Faciliter l’exécution locale de modèles LLM ou de génération d’images (type Stable Diffusion) sur des machines avec plusieurs GPU, sans dépendre du cloud ni investir dans du matériel haut de gamme.

---

## 📦 Structure du projet

```
VRAMancer/
├── README.md               # Description du projet
├── vramancer.py            # Script principal
├── requirements.txt        # Dépendances Python
├── data/                   # Données brutes
│   └── raw_data.csv
├── models/                 # Modèles sauvegardés
│   └── model.pkl
├── utils/                  # Fonctions utilitaires
│   └── helpers.py
└── tests/                  # Tests unitaires
    └── test_vramancer.py
```

---

## 🛠️ Fonctionnalités à venir

- 📊 Visualisation en temps réel de l’usage de la VRAM
- 🧠 Analyse des pics de consommation GPU
- 🗃️ Export des statistiques en CSV ou JSON
- 🖼️ Interface graphique simple (Tkinter ou PyQt)
- 🧩 Support multi-GPU et compatibilité CUDA
- 🔌 Intégration avec Hugging Face et ComfyUI

---

## 🧪 Installation

```bash
git clone https://github.com/tonpseudo/VRAMancer.git
cd VRAMancer
pip install -r requirements.txt
```

---

## 🤝 Vers une intégration native ?

VRAMancer a été imaginé pour démocratiser l’accès aux modèles IA volumineux sur des configurations modestes.  
Nous pensons que ce type d’approche pourrait enrichir l’écosystème NVIDIA, en complément des solutions haut de gamme comme CUDA, TensorRT ou DGX.

Si cette idée résonne chez des acteurs du secteur, nous serions ravis d’explorer une collaboration ou une intégration plus poussée.

---

## 📬 Contribuer

Les contributions sont les bienvenues !  
N’hésitez pas à ouvrir une issue ou une pull request pour proposer une amélioration, corriger un bug ou discuter d’une idée.

---

## 📜 Licence

MIT — libre d’utilisation, modification et distribution.



### 🔧 Options Premium Disponibles

Voici les modules et fonctionnalités avancées disponibles en option pour les utilisateurs premium :

#### 🚀 Protocoles & Réseaux

- **VRAMancer Link**  
  Protocole réseau ultra-léger et optimisé pour les échanges inter-machines (GPU ↔ GPU, machine ↔ machine), utilisant les ports SFP+ ou Ethernet sans passer par TCP/IP. Idéal pour les architectures cloud distribuées ou les clusters IA.

- **ZeroStack TCP-Free Mode**  
  Permet de bypasser complètement la stack TCP/IP pour des communications directes entre nœuds, réduisant la latence et augmentant le débit.

- **Cloud Fabric Custom**  
  Stack réseau propriétaire pour cloud distribué, avec gestion intelligente des flux, priorisation des tâches IA, et isolation des workloads.

- **SFP Protocol Override**  
  Firmware dédié pour cartes réseau et switchs compatibles, permettant de remplacer le protocole Ethernet par un protocole custom (VRAMancer Link).

#### 🧠 Mémoire & GPU

- **VRAMancer Memory Sync**  
  Synchronisation directe des blocs mémoire entre machines sans copie intermédiaire, via le protocole VRAMancer.

- **GPU Direct Dispatch**  
  Envoi direct de tâches entre GPU distants via le protocole custom, sans passer par le CPU ou le système d’exploitation.

- **Cloud Bridge**  
  Extension de la mémoire GPU via le réseau, permettant à plusieurs machines de partager dynamiquement leur VRAM.

#### ⚙️ Modules Optionnels

- **VRAMancer Lite**  
  Version allégée du protocole pour les machines modestes ou configurations edge.

- **Auto Tuner**  
  Optimisation dynamique des paramètres système et réseau en fonction de la charge et des performances observées.

- **Hugging Face Bridge**  
  Compatibilité native avec les modèles Hugging Face, incluant le chargement, l’inférence et la conversion.

- **Scheduler Intelligent**  
  Système de préchargement adaptatif des modèles et des données, basé sur les patterns d’usage et les prédictions de charge.

#### 🔐 Sécurité & Monitoring

- **Secure Fabric Layer**  
  Chiffrement natif des échanges sur le protocole VRAMancer, avec authentification par clé publique et gestion des sessions.


