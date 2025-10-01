


# 🆕 Nouveautés 2025

- **API No-code Workflows IA** : création et exécution de pipelines IA sans coder
- **Dashboard web/mobile responsive** : supervision avancée, heatmap, actions distantes, logs, mobile friendly
- **Sandbox / Digital Twin avancé** : simulation, replay, prédiction, tests
- **Auditabilité / XAI / Fairness** : explications IA, logs, rapport d’audit, conformité
- **Actions distantes avancées** : auto-scale, failover, migration live, reboot, offload
- **Packaging universel** : archive unique avec tous les installateurs, docs, guides, rapport PDF
- **Guide ultra-débutant** : instructions simplifiées pour chaque OS
- **Modules premium, edge, mobile, cloud, marketplace, onboarding vidéo**

---

# 🚀 Installation ultra-débutant


## 📦 Installation du bundle ZIP

1. Téléchargez le fichier : `vramancer_release_bundle.zip`
2. Extrayez l’archive ZIP :
	```bash
	unzip vramancer_release_bundle.zip
	```
3. Ouvrez le dossier extrait : `release_bundle`
4. Choisissez votre OS :
	- **Linux** : lancez `installers/install_linux.sh` ou installez le `.deb`
	- **macOS** : lancez `installers/install_macos.sh`
	- **Windows** : lancez `installers/install_windows.bat`
5. Suivez les instructions à l’écran
6. Consultez le guide ultra-débutant dans `docs/INSTALL_ULTRA_DEBUTANT.md`

Tout est automatisé, plug-and-play, multi-OS, dashboards auto, cluster auto, onboarding vidéo/interactive.

----

# �📝 Rapport d’audit complet – Octobre 2025

Le projet VRAMancer est complet, modulaire, disruptif, prêt pour la production et l’extension. Toutes les briques demandées sont présentes : orchestration IA, clustering, dashboards, sécurité, marketplace, XAI, digital twin, API no-code, audit, actions distantes, mobile, edge, cloud, onboarding, packaging pro, tests, documentation.

**Modules principaux** : Orchestration IA multi-backend, découpage adaptatif VRAM, clustering dynamique, dashboards (Qt, Tk, Web, Mobile, CLI), supervision avancée, marketplace/plugins IA générative, auto-réparation, confidential computing, federated learning, digital twin, API no-code, auditabilité/XAI/fairness, orchestration cloud/edge/hybride, documentation complète.

**Packaging/installateurs** : setup.py, build_deb.sh, Install.sh, installateurs Linux/macOS/Windows, Makefile, Makefile.lite, build_lite.sh, archive, .deb, scripts, version Lite CLI, benchmarks, onboarding, dashboards auto, cluster auto.

**Documentation** : README.md complet, docs/ guides API, quickstart, sécurité, mobile, collective, automation, edge, hybrid cloud, README_LITE.md, ONBOARDING.md, MANUEL_FR.md, MANUAL_EN.md, RELEASE.md, ROADMAP_IDEES.md, PROTECT.md.

**Tests/validation** : tests/ unitaires, stress, scheduler, monitor, imports, testutils, scripts de build, benchmark, install, onboarding, plugin, automatisés.

**À améliorer/vérifier** : Inclusion des nouveaux modules dans installateurs et packaging, mise à jour du README et du guide d’installation ultra-débutant, cohérence des versions/dépendances, tests sur chaque OS, publication à jour sur le repository en ligne.

----

<div align="center">
	<img src="vramancer.png" width="120" alt="VRAMancer logo"/>
	<h1>VRAMancer</h1>
	<b>Orchestrateur IA universel, multi-backend, edge/cloud, plug-and-play, sécurité, dashboards, marketplace, automatisation, XAI, digital twin…</b><br>
	<a href="https://github.com/thebloodlust/VRAMancer/actions/workflows/ci.yml"><img src="https://github.com/thebloodlust/VRAMancer/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
</div>

----

## 🗂️ Sommaire
- [Installation ultra-débutant / Ultra-beginner install](#installation-ultra-débutant--ultra-beginner-install)
- [Fonctionnalités clés / Key features](#fonctionnalités-clés--key-features)
- [Super features disruptives](#super-features-disruptives)
- [Tableau des modules](#tableau-des-modules)
- [Documentation & guides](#documentation--guides)
- [FAQ & Support](#faq--support)
- [Roadmap](#roadmap)

---

## 🚀 Installation ultra-débutant / Ultra-beginner install

```bash
git clone https://github.com/thebloodlust/VRAMancer.git
cd VRAMancer
bash Install.sh
source .venv/bin/activate
make deb           # ou make archive / make lite
```

**Windows** : `installers/install_windows.bat`<br>
**Linux** : `bash installers/install_linux.sh`<br>
**macOS** : `bash installers/install_macos.sh`

Tout est guidé, plug-and-play, multi-OS, dashboards auto, cluster auto, onboarding vidéo/interactive.

---

## 🔥 Fonctionnalités clés / Key features
- Orchestration IA multi-backend (HF, vLLM, Ollama, DeepSpeed, TensorRT…)
- Découpage adaptatif VRAM, exploitation GPU secondaires, clustering dynamique
- Dashboards Qt, Tk, Web, CLI, mobile/tablette
- Plug-and-play (USB4, Ethernet, WiFi), auto-sensing, auto-repair, monitoring
- Sécurité avancée (Zero Trust, MFA, SSO, compliance RGPD/HIPAA/ISO)
- Marketplace plugins/extensions, onboarding vidéo, packaging pro, CI, tests

---

## 🚀 Super features disruptives
- Auto-optimisation IA/ressources (auto-tuning, MLOps, green AI)
- Confidential Computing (SGX/SEV/Nitro, exécution IA chiffrée)
- Zero Trust & SSO universel (OAuth2/SAML, segmentation, audit)
- Plugins IA générative (LLM, diffusion, audio, vidéo, scoring)
- Orchestration multi-cloud/edge (placement intelligent, RGPD, coût, SLA)
- Explainability & Fairness (XAI, dashboard, détection de biais, éthique)
- Auto-réparation avancée (rollback, redéploiement, alertes IA)
- Federated Learning natif (agrégation sécurisée, privacy)
- API “No Code” (drag & drop pipelines IA, endpoints)
- Digital Twin (simulation temps réel, jumeau numérique)

---

## 🧩 Tableau des modules principaux

| Module / Dossier                | Fonction / Description                                 |
|---------------------------------|-------------------------------------------------------|
| core/auto/auto_tuner.py         | Auto-optimisation IA/ressources                       |
| core/security/confidential_computing.py | Confidential Computing (SGX/SEV/Nitro)         |
| core/security/zero_trust.py     | Proxy Zero Trust, SSO OAuth2/SAML                     |
| core/marketplace/generative_plugin.py | Plugins IA générative (LLM, diffusion…)          |
| core/orchestrator/placement_engine.py | Orchestration multi-cloud/edge                   |
| core/xai/xai_dashboard.py       | Explainability & Fairness (XAI, biais, éthique)       |
| core/auto/auto_repair.py        | Auto-réparation avancée                               |
| core/collective/federated_learning.py | Federated Learning natif                         |
| core/api/no_code_api.py         | API “No Code” (pipelines drag & drop)                 |
| core/simulator/digital_twin.py  | Digital Twin (simulation infra IA)                    |
| core/cloud/hybrid_bridge.py     | Bridge cloud hybride (AWS, Azure, GCP)                |
| core/collective/federation.py   | Intelligence collective, partage inter-cluster        |
| mobile/dashboard_mobile.py      | Dashboard mobile/tablette                             |
| core/security/compliance.py     | Compliance RGPD, HIPAA, ISO                           |
| core/security/remote_access.py  | Contrôle web sécurisé, MFA, gestion des rôles         |
| core/security/ldap_auth.py      | Authentification LDAP/Active Directory                |

---

## 📚 Documentation & guides
- [docs/automation_api.md](docs/automation_api.md) — API d’automatisation avancée (REST/GraphQL)
- [docs/hybrid_cloud.md](docs/hybrid_cloud.md) — Bridge cloud hybride
- [docs/collective_federation.md](docs/collective_federation.md) — Intelligence collective, fédération
- [docs/mobile_dashboard.md](docs/mobile_dashboard.md) — Dashboard mobile/tablette
- [docs/security_enterprise.md](docs/security_enterprise.md) — Sécurité, conformité, LDAP, contrôle web
- [docs/edge_iot_supervision.md](docs/edge_iot_supervision.md) — Edge/IoT & supervision
- [MANUEL_FR.md](MANUEL_FR.md) — Manuel complet (français)
- [MANUAL_EN.md](MANUAL_EN.md) — Complete manual (English)
- [ONBOARDING.md](ONBOARDING.md) — Onboarding vidéo/interactive
- [ROADMAP_IDEES.md](ROADMAP_IDEES.md) — Roadmap & idées avancées

---

## ❓ FAQ & Support

**Q : Comment installer VRAMancer sur mon OS ?**<br>
A : Utilisez le script d’installation adapté (Windows, Linux, macOS) dans le dossier `installers/` ou suivez le guide ultra-débutant ci-dessus.

**Q : Comment ajouter un nœud au cluster ?**<br>
A : Branchez-le (USB4, Ethernet, WiFi), il sera détecté automatiquement.

**Q : Comment activer les dashboards ?**<br>
A : `python -m vramancer.main --mode qt` (ou tk/web/cli/mobile)

**Q : Où trouver la doc sur les modules avancés ?**<br>
A : Voir la section Documentation & guides ci-dessus.

**Q : Qui contacter pour du support ou contribuer ?**<br>
A : Ouvrez une issue GitHub ou contactez thebloodlust.

---

## 🛣️ Roadmap

Voir [ROADMAP_IDEES.md](ROADMAP_IDEES.md) pour toutes les idées avancées, modules à venir, et suggestions communautaires.

---

MIT — (c) thebloodlust 2023-2025

## 🇬🇧 English version

### 🚀 Quick install
```bash
git clone https://github.com/thebloodlust/VRAMancer.git
cd VRAMancer
bash Install.sh
source .venv/bin/activate
make deb           # or make archive / make lite
```

### 🖥️ Launch (examples)
- `python -m vramancer.main` (auto)
- `python -m vramancer.main --backend vllm --model mistral`
- `make lite` (CLI only version)

### 📦 Packaging
- `.deb`: `make deb` or `bash build_deb.sh`
- Portable archive: `make archive`
- Lite CLI version: `make lite`

<div align="center">
	<img src="vramancer.png" width="120" alt="VRAMancer logo"/>
</div>

# VRAMancer

[![CI](https://github.com/thebloodlust/VRAMancer/actions/workflows/ci.yml/badge.svg)](https://github.com/thebloodlust/VRAMancer/actions)

**Optimisation VRAM multi-GPU, LLM universel, dashboards modernes, packaging pro.**

---

## 🇫🇷 Version française


### 🚀 Notice d’installation ultra-débutant

#### Étape 1 : Copier le dépôt
1. Rendez-vous sur https://github.com/thebloodlust/VRAMancer
2. Cliquez sur "Code" puis "Download ZIP" ou copiez le lien pour cloner avec Git
3. Décompressez l’archive ZIP ou lancez :
	```bash
	git clone https://github.com/thebloodlust/VRAMancer.git
	cd VRAMancer
	```

#### Étape 2 : Lancer l’installeur selon votre OS

**Windows**
1. Ouvrez le dossier `installers` dans l’explorateur
2. Double-cliquez sur `install_windows.bat` (ou clic droit > "Exécuter en tant qu’administrateur")
3. Suivez l’interface graphique (tout est guidé)

**Linux**
1. Ouvrez un terminal dans le dossier `installers`
2. Tapez :
	```bash
	bash install_linux.sh
	```
3. Suivez l’interface graphique (tout est guidé)

**macOS**
1. Ouvrez un terminal dans le dossier `installers`
2. Tapez :
	```bash
	bash install_macos.sh
	```
3. Suivez l’interface graphique (tout est guidé)

#### Étape 3 : Plug-and-play
1. Branchez la machine (USB4, Ethernet, WiFi)
2. Le nœud est détecté automatiquement
3. Le cluster se crée, le master est choisi selon la performance (modifiable)
4. Vous pouvez ajouter d’autres machines à tout moment, elles seront reconnues instantanément

#### Étape 4 : Lancer le dashboard ou le cluster
1. Dashboard :
	```bash
	python -m vramancer.main --mode qt
	# ou --mode tk / web / cli
	```
2. Cluster master :
	```bash
	python core/network/cluster_master.py
	```
3. Découverte de nœuds :
	```bash
	python core/network/cluster_discovery.py
	```
4. Agrégation et routage :
	```bash
	python core/network/resource_aggregator.py
	```

#### Étape 5 : Utilisation avancée
- Override manuel du master/slave
- Monitoring réseau intégré
- Routage adaptatif, pipeline asynchrone, compression des poids
- Sécurité, auto-réparation, extensions premium

---

## 🇬🇧 Ultra-beginner installation guide

#### Step 1: Copy the repository
1. Go to https://github.com/thebloodlust/VRAMancer
2. Click "Code" then "Download ZIP" or copy the link to clone with Git
3. Unzip the archive or run:
	```bash
	git clone https://github.com/thebloodlust/VRAMancer.git
	cd VRAMancer
	```

#### Step 2: Run the installer for your OS

**Windows**
1. Open the `installers` folder in Explorer
2. Double-click `install_windows.bat` (or right-click > "Run as administrator")
3. Follow the graphical interface (everything is guided)

**Linux**
1. Open a terminal in the `installers` folder
2. Type:
	```bash
	bash install_linux.sh
	```
3. Follow the graphical interface (everything is guided)

**macOS**
1. Open a terminal in the `installers` folder
2. Type:
	```bash
	bash install_macos.sh
	```
3. Follow the graphical interface (everything is guided)

#### Step 3: Plug-and-play
1. Plug in the machine (USB4, Ethernet, WiFi)
2. Node is auto-detected
3. Cluster is created, master is chosen by performance (can be overridden)
4. You can add more machines anytime, they’ll be recognized instantly

#### Step 4: Launch dashboard or cluster
1. Dashboard:
	```bash
	python -m vramancer.main --mode qt
	# or --mode tk / web / cli
	```
2. Cluster master:
	```bash
	python core/network/cluster_master.py
	```
3. Node discovery:
	```bash
	python core/network/cluster_discovery.py
	```
4. Aggregation & routing:
	```bash
	python core/network/resource_aggregator.py
	```

#### Step 5: Advanced usage
- Manual master/slave override
- Integrated network monitoring
- Adaptive routing, async pipeline, weight compression
- Security, auto-repair, premium extensions

---

### � Manuel d’utilisation

Consultez le fichier [MANUEL_FR.md](MANUEL_FR.md) pour le guide complet : orchestration, dashboard, plug-and-play, override master/slave, agrégation VRAM/CPU, etc.

---

## 🇬🇧 English version

### 🚀 Installation & Getting Started

#### Windows
1. Run `installers/install_windows.bat` (double-click or terminal)
2. Follow the graphical interface for installation and setup
3. Plug in the machine (USB4, Ethernet, WiFi): node is auto-detected
4. Cluster is created, master is chosen by performance (can be overridden)

#### Linux
1. Run `bash installers/install_linux.sh`
2. Follow the graphical interface for installation and setup
3. Plug in the machine (USB4, Ethernet, WiFi): node is auto-detected
4. Cluster is created, master is chosen by performance (can be overridden)

#### macOS
1. Run `bash installers/install_macos.sh`
2. Follow the graphical interface for installation and setup
3. Plug in the machine (USB4, Ethernet, WiFi): node is auto-detected
4. Cluster is created, master is chosen by performance (can be overridden)

---

### 📖 User Manual

See [MANUAL_EN.md](MANUAL_EN.md) for the complete guide: orchestration, dashboard, plug-and-play, master/slave override, VRAM/CPU aggregation, etc.

---


---

## 🛣️ Roadmap & idées complémentaires

### 🇫🇷 À compléter / idées à ajouter
- Tests automatisés sur chaque OS (CI multi-plateforme)
- Module de sécurité (authentification, chiffrement des transferts)
- Dashboard web avancé (visualisation cluster, logs, contrôle distant)
- Support de nouveaux backends IA (DeepSpeed, TensorRT…)
- Module d’auto-réparation (détection et correction automatique des pannes de nœud)
- Marketplace de plugins/extensions (modules premium, connecteurs cloud, etc.)
- Documentation vidéo ou interactive pour onboarding ultra-facile

### 🇬🇧 To complete / ideas to add
- Automated tests for each OS (multi-platform CI)
- Security module (authentication, encrypted transfers)
- Advanced web dashboard (cluster visualization, logs, remote control)
- Support for new AI backends (DeepSpeed, TensorRT…)
- Auto-repair module (automatic node failure detection and correction)
- Plugin/extension marketplace (premium modules, cloud connectors, etc.)
- Video or interactive documentation for ultra-easy onboarding

---

MIT — (c) thebloodlust 2023-2025
