# 🎮 VRAMancer

**FR 🇫🇷**  
VRAMancer est un moteur open‑source conçu pour optimiser l’utilisation de la mémoire vidéo (VRAM) sur des configurations multi‑GPU, même modestes. Il permet de charger des modèles IA volumineux en répartissant intelligemment les blocs entre GPU, RAM, NVMe et réseau.

**EN 🇬🇧**  
VRAMancer is an open-source engine designed to optimize video memory (VRAM) usage across multi-GPU setups — even modest ones. It enables large AI models to run locally by intelligently routing blocks across GPU, RAM, NVMe, and network.

---

## 🚀 Objectif / Purpose

**FR**  
Faciliter l’exécution locale de modèles LLM ou de génération d’images (Stable Diffusion, etc.) sans dépendre du cloud ni investir dans du matériel haut de gamme.

**EN**  
Make it easy to run LLMs or image generation models (e.g. Stable Diffusion) locally — no cloud, no expensive hardware.

---

## 📦 Structure du projet / Project Structure

```
VRAMancer/
├── core/                  # Moteur de routage et planification
│   ├── scheduler.py       # Planification adaptative
│   ├── block_router.py    # Routage dynamique multi-niveau
│   ├── block_metadata.py  # Poids et importance des blocs
│   ├── compute_engine.py  # Détection GPU/CPU/MPS/ROCm
│   ├── memory_monitor.py  # Surveillance RAM
│   ├── storage_manager.py # Fallback NVMe
│   └── network/
│       └── remote_executor.py  # Fallback réseau
├── utils/
│   └── gpu_utils.py       # Détection multi-GPU
├── dashboard/
│   ├── app.py             # Interface graphique (Flask)
│   └── visualizer.py      # Visualisation VRAM en temps réel
├── premium/               # Modules avancés
├── tests/                 # Tests unitaires et simulateurs
├── launcher.py            # Point d’entrée CLI
├── vrm                    # Alias CLI
├── README.md
├── setup.py
├── debian/                # Packaging .deb
```

---

## 🧪 Installation

```bash
git clone https://github.com/thebloodlust/VRAMancer.git
cd VRAMancer
pip install -r requirements.txt
```

---

## 🎯 Fonctionnalités / Features

**FR 🇫🇷**
- ✅ Répartition intelligente de la VRAM entre plusieurs GPU
- ✅ Routage dynamique vers GPU, CPU, RAM, NVMe ou réseau
- ✅ Support multi-backend : CUDA, ROCm, MPS, CPU
- ✅ Fallback automatique en cas de saturation mémoire
- ✅ Interface graphique pour visualiser l’exécution
- ✅ Export des statistiques en CSV ou JSON
- ✅ Intégration Hugging Face et ComfyUI
- ✅ Packaging `.deb` pour installation système

**EN 🇬🇧**
- ✅ Smart VRAM distribution across multiple GPUs
- ✅ Dynamic routing to GPU, CPU, RAM, NVMe, or network
- ✅ Multi-backend support: CUDA, ROCm, MPS, CPU
- ✅ Automatic fallback when memory is saturated
- ✅ Graphical interface to monitor execution in real time
- ✅ Export statistics to CSV or JSON
- ✅ Hugging Face and ComfyUI integration
- ✅ `.deb` packaging for system-wide installation

---

## 🔧 Modules Premium Disponibles / Premium Modules

| Module | Description |
|--------|-------------|
| VRAMancer Link | Protocole réseau ultra‑léger (SFP+/Ethernet) |
| ZeroStack TCP‑Free Mode | Bypass complet de la pile TCP/IP |
| Cloud Fabric Custom | Stack réseau propriétaire |
| SFP Protocol Override | Firmware Ethernet custom |
| VRAMancer Memory Sync | Synchronisation inter-machines |
| GPU Direct Dispatch | Envoi direct entre GPU distants |
| Cloud Bridge | Extension VRAM via réseau |
| VRAMancer Lite | Version edge allégée |
| Auto Tuner | Optimisation dynamique |
| Hugging Face Bridge | Compatibilité native HF |
| Scheduler Intelligent | Prédiction adaptative |
| Secure Fabric Layer | Chiffrement et sessions |

---

## 📦 Packaging `.deb`

```bash
dpkg-deb --build VRAMancer
```

Or with [fpm](https://fpm.readthedocs.io/en/latest/):

```bash
fpm -s dir -t deb -n vramancer -v 1.0 .
```

---

## 📊 Visualisation du Routage / Routing Visualization

- Timeline des blocs exécutés
- Affichage des périphériques utilisés (GPU, CPU, NVMe, réseau)
- Logs exportables en JSON

---

## 🤝 Contribution

**FR 🇫🇷**  
Les contributions sont les bienvenues ! Ouvrez une issue ou une pull request pour corriger un bug, ajouter une fonctionnalité ou discuter d’une idée.

**EN 🇬🇧**  
Contributions welcome! Open an issue or pull request to fix a bug, add a feature, or share an idea.

---

## 📜 Licence

MIT — libre d’utilisation, modification et distribution.
