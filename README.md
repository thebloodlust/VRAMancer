
# 🎮 VRAMancer

**FR 🇫🇷**
> VRAMancer optimise la VRAM sur multi-GPU, permet d’exécuter n’importe quel LLM localement, et offre un dashboard moderne (Qt, Tk, Web, CLI) avec packaging .deb prêt à l’emploi.

**EN 🇬🇧**
> VRAMancer optimizes VRAM across multi-GPU setups, runs any LLM locally, and provides a modern dashboard (Qt, Tk, Web, CLI) with ready-to-use .deb packaging.

---

## 🚀 Objectif / Purpose

**FR** : Exécuter localement des LLM ou modèles IA volumineux sur n’importe quelle config, sans cloud ni matériel hors de prix.

**EN** : Run large LLMs or AI models locally on any hardware, no cloud or expensive gear required.

---

## 📦 Structure du projet / Project Structure

```
VRAMancer/
├── core/         # Orchestrateur, découpeur, planificateur, gestion mémoire
├── dashboard/    # Dashboards Qt, Tk, Web, CLI, visualisation VRAM
├── cli/          # CLI alternative
├── premium/      # Modules avancés (réseau, tuning, bridge HF, etc.)
├── utils/        # Outils GPU, helpers
├── vramancer/    # Entrée principale, packaging
├── scripts/      # scripts/vramancer-launcher.sh (lanceur universel)
├── Debian/       # Fichiers .deb, .desktop, icônes
├── tests/        # Tests unitaires
├── README.md, setup.py, Makefile, ...
```

---

## 🧪 Installation

### 1. Locale (recommandé)
```bash
git clone https://github.com/thebloodlust/VRAMancer.git
cd VRAMancer
bash Install.sh
source .venv/bin/activate
make auto
```

### 2. Paquet `.deb` (Ubuntu/Debian)
```bash
sudo dpkg -i vramancer_1.0.deb
/usr/local/bin/vramancer-launcher.sh --mode auto
```

### 3. Archive portable
```bash
tar -xzf vramancer.tar.gz
cd VRAMancer
bash Install.sh
make auto
```

---

## 🖥️ Dashboards & Usage

Lancez le dashboard de votre choix :

```bash
# Mode auto (détection Qt > Tk > Web > CLI)
scripts/vramancer-launcher.sh --mode auto
# Forcer Qt
scripts/vramancer-launcher.sh --mode qt
# Forcer Tkinter
scripts/vramancer-launcher.sh --mode tk
# Forcer Web
scripts/vramancer-launcher.sh --mode web
# Mode CLI
scripts/vramancer-launcher.sh --mode cli
```

---

## 🤖 Compatibilité LLM universelle

- Découpe et routage dynamiques pour tous modèles HuggingFace, GPT, Llama, Mistral, etc.
- Aucune dépendance à un modèle unique : le splitter et l’orchestrateur détectent automatiquement la structure (L1-L6, etc.).
- Support multi-backend : CUDA, ROCm, MPS, CPU.

---

## 🎯 Fonctionnalités clés / Key Features

- Répartition VRAM multi-GPU, fallback RAM/NVMe/réseau
- Dashboard moderne (Qt, Tk, Web, CLI)
- Visualisation temps réel, logs exportables
- Intégration HuggingFace, ComfyUI, Llama.cpp, etc.
- Packaging .deb, installation simple
- Modules premium (réseau, tuning, bridge HF, etc.)

---

## 🔧 Premium Modules (en option)

| Module                  | Description                            |
|------------------------|----------------------------------------|
| VRAMancer Link         | Protocole réseau ultra‑léger (SFP+/Ethernet)  |
| ZeroStack TCP‑Free Mode| Bypass complet de la pile TCP/IP        |
| Cloud Fabric Custom    | Stack réseau propriétaire               |
| SFP Protocol Override  | Firmware Ethernet custom                |
| VRAMancer Memory Sync  | Synchronisation inter-machines          |
| GPU Direct Dispatch    | Envoi direct entre GPU distants         |
| Cloud Bridge           | Extension VRAM via réseau               |
| VRAMancer Lite         | Version edge allégée                    |
| Auto Tuner             | Optimisation dynamique                  |
| Hugging Face Bridge    | Compatibilité native HF                 |
| Scheduler Intelligent  | Prédiction adaptative                   |
| Secure Fabric Layer    | Chiffrement et sessions                 |

---

## 📦 Packaging & Build

```bash
make deb          # Build du paquet .deb
make release      # Build complet prêt à distribuer
```

---

## 📊 Visualisation & Logs

- Timeline des blocs exécutés
- Affichage des périphériques utilisés (GPU, CPU, NVMe, réseau)
- Logs exportables en JSON/CSV

---

## 🤝 Contribution

**FR** : Contributions bienvenues ! Ouvrez une issue ou une PR pour corriger un bug, ajouter une fonctionnalité ou discuter d’une idée.

**EN** : Contributions welcome! Open an issue or PR to fix a bug, add a feature, or discuss an idea.

---

## 📜 Licence

MIT — libre d’utilisation, modification et distribution.
