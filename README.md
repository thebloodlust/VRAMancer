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
