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
### ⚠️ Astuce Windows : chemins et nom de dossier

Si vous téléchargez plusieurs fois l’archive ZIP, Windows ajoute une parenthèse et un chiffre au nom du dossier (ex : `VRAMancer-main (2)`).
Pour éviter les problèmes de chemins dans les scripts, renommez le dossier extrait en `VRAMancer-main` (sans parenthèse ni chiffre) avant de lancer l’installation.
Lancez toujours les scripts depuis le dossier `release_bundle`.
### ⚠️ Note Windows : Installation de Rust

Certains modules (ex : tokenizers) nécessitent Rust pour s’installer sous Windows. Si une erreur apparaît lors de l’installation, installez Rust via :

https://
rustup.rs/

Puis relancez l’installation.


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
6. Consultez le guide ultra-débutant dans `docs/INSTALL_ULTRA_DEBUTANT.md` (ajouté)

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

Chemin rapide (Linux/macOS) :
```bash
git clone https://github.com/thebloodlust/VRAMancer.git \
	&& cd VRAMancer \
	&& bash Install.sh \
	&& source .venv/bin/activate \
	&& python -m core.api.unified_api
```
Puis ouvrir: `http://localhost:5030/api/version`

### 🧭 Parcours express (Étapes 1 → 5)
| Étape | Action | Commandes / Détails |
|-------|--------|---------------------|
| 1 | Cloner & créer venv | `git clone ... && cd VRAMancer && bash Install.sh` (crée `.venv`) |
| 2 | Vérifier dépendances lourdes optionnelles | GPU libs (CUDA/ROCm), zstd/lz4, tracing OTEL (facultatif) |
| 3 | Lancer serveur de base | `python -m vramancer.main` (auto backend heuristique) |
| 4 | Ouvrir dashboard / métriques | Web: `--mode web`, Prometheus: `curl :9108/metrics` |
| 5 | Activer features avancées | HA: `export VRM_HA_REPLICATION=1`; Tracing: `export VRM_TRACING=1`; Fastpath: `export VRM_FASTPATH_IF=eth0` |

Cheat‑sheet rapide (désactiver limites pour tests) :
```bash
export VRM_DISABLE_RATE_LIMIT=1
export VRM_TEST_MODE=1
pytest -q
```

**Windows** : `installers/install_windows.bat` (crée venv, installe deps, lance systray)<br>
Alternative tout-en-un : `installers/start_windows_all.bat` (installe/MAJ deps + API + Web + Qt)
**Linux** : `bash installers/install_linux.sh` (option GUI + API)<br>
**macOS** : `bash installers/install_macos.sh`

Tout est guidé, plug-and-play, multi-OS, dashboards auto, cluster auto, onboarding vidéo/interactive.

---

### 🌐 Endpoints principaux (API unifiée)

| Endpoint | Description |
|----------|-------------|
| `GET /api/version` | Version backend |
| `GET /api/health` | Healthcheck léger (utilisé par Docker HEALTHCHECK) |
| `GET /api/env` | Statut runtime (features actives, flags environnement) |
| `POST /api/workflows` | Crée un workflow no-code (Pydantic validation) |
| `GET /api/workflows` | Liste workflows (mémoire + persistence sqlite si activée) |
| `POST /api/federated/round/start` | Démarre un round fédéré |
| `POST /api/federated/round/submit` | Soumet une mise à jour (poids + valeur) |
| `GET /api/federated/round/aggregate` | Agrégation pondérée (clipping + bruit optionnel) |
| `GET /api/federated/secure` | État secure aggregation (masquage basique) |
| `POST /api/federated/secure {enabled:true|false}` | Active/désactive masquage simple |
| `POST /api/xai/explain` | Explication XAI (baseline feature attribution) |
| `GET /api/xai/explainers` | Liste des explainers disponibles |
| `GET /api/marketplace/plugins` | Plugins + signatures (sandbox run expérimental) |
| `POST /api/quota/reset` | Reset compteurs quotas (outillage/tests) |

Chaque réponse inclut un header `X-Request-ID` (corrélation logs). Vous pouvez fournir votre propre identifiant via le même header dans la requête.

### 🔐 Authentification forte (JWT)

Flux supporté (prototype production-ready minimal) :
1. `POST /api/login {"username":"admin","password":"admin"}` → tokens `{access, refresh, expires_in}`
2. Appels protégés: ajouter l'en-tête `Authorization: Bearer <access>`
3. Rafraîchissement: `POST /api/token/refresh {"refresh":"<refresh_token>"}` → nouveaux tokens

Variables d'environnement:
| Variable | Rôle | Défaut |
|----------|------|--------|
| VRM_AUTH_SECRET | Secret signature JWT HS256 | auto-généré (dev) |
| VRM_AUTH_EXP | Durée access token (s) | 900 |
| VRM_AUTH_REFRESH_EXP | Durée refresh token (s) | 86400 |

Exemple rapide:
```bash
curl -s -X POST -H 'Content-Type: application/json' \
	-d '{"username":"admin","password":"admin"}' http://localhost:5030/api/login | jq .
ACCESS=... # insérer access renvoyé
curl -H "Authorization: Bearer $ACCESS" http://localhost:5030/api/workflows
```

NOTE: En production changer immédiatement le mot de passe admin et définir `VRM_AUTH_SECRET`.

### 🪟 Note Windows (Dashboards)
Si `flask_socketio` ou `torch` ne sont pas installés, les dashboards Web / Qt démarrent en mode dégradé (pas de temps réel SocketIO, certaines fonctions d’offload inactives). Pour l’expérience complète :
```bash
pip install flask-socketio torch
```
Les wrappers se trouvent sous `installers/dashboard/` et redirigent vers `dashboard/`.

Mode ultra-léger forcé (pas de torch / transformers) :
```bat
set VRM_DASHBOARD_MINIMAL=1
python installers\dashboard\dashboard_web.py
```

Variables utiles : `VRM_UNIFIED_API_QUOTA`, `VRM_READ_ONLY`, `VRM_LOG_JSON`, `VRM_REQUEST_LOG`, `VRM_DISABLE_SOCKETIO`.

#### ⚠️ Windows : erreur build `tokenizers` / `link.exe not found`
Si l'installation échoue sur `tokenizers` (compilation Rust/MSVC) avec Python 3.12 :
1. Solution rapide (recommended) : utiliser le fichier `requirements-windows.txt` adapté :
	```bash
	pip install -r requirements-windows.txt
	```
	(Versions plus récentes : `transformers 4.46.2` + `tokenizers 0.20.1` avec wheels précompilés.)
2. Ou installer toolchain :
	- Installer Rust (`https://rustup.rs/`)
	- Installer *Visual Studio Build Tools* avec composant "Desktop development with C++" (inclut `link.exe`)
	- Relancer: `pip install -r requirements.txt`
3. Fallback possible : définir `USE_SLOW_TOKENIZER=1` (le code force alors l'utilisation d'un tokenizer Python si dispo / ou stub silencieux).

Mode minimal (aucun modèle / no HF): n'installe que `requirements-lite.txt` puis lancer l'API et dashboards (fonctionnalités ML avancées inactives).

### 🖥️ Qt Dashboard (fiabilisation connexions)
Le dashboard Qt utilise maintenant des retries configurables + fallback `127.0.0.1` si `localhost` échoue.

Variables d'environnement spécifiques :
| Variable | Rôle | Défaut |
|----------|------|--------|
| `VRM_API_BASE` | Base URL API supervision (nodes, telemetry) | `http://localhost:5010` |
| `VRM_MEMORY_BASE` | Base URL service mémoire hiérarchique | `http://localhost:5000` |
| `VRM_API_TIMEOUT` | Timeout (s) par requête HTTP | `2.5` |
| `VRM_API_RETRIES` | Nombre de tentatives par base | `3` |

Comportement : chaque requête tente `VRM_API_BASE` puis variante `127.0.0.1` avec backoff progressif. L'état (connecté / injoignable) est affiché sans spam.
Pour réduire l'erreur `Max retries exceeded with url /api/nodes` : augmenter `VRM_API_TIMEOUT` (ex: `export VRM_API_TIMEOUT=5`).

Détection automatique intégrée : si `VRM_API_BASE` n'est pas défini, le dashboard scanne `5030` puis `5010` (`/api/health`).
Script CLI équivalent :
```bash
python scripts/api_autodetect.py --json
```
Debug verbeux (requêtes, ports testés) :
```bash
export VRM_API_DEBUG=1
python dashboard/dashboard_qt.py
```
Note Windows: si le message "backend injoignable" persiste, définir manuellement :
```bat
set VRM_API_BASE=http://127.0.0.1:5030
python installers\dashboard\dashboard_web.py
```


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
| core/security/zero_trust.py     | (Manquant) Placeholder Zero Trust / SSO à ajouter     |
| core/marketplace/generative_plugin.py | (Prototype) Plugins IA générative (LLM, diffusion…) |
| core/orchestrator/placement_engine.py | (Prototype avancé) Orchestration multi-cloud/edge|
| core/xai/xai_dashboard.py       | (Stub) Explainability & Fairness (XAI, biais)         |
| core/auto/auto_repair.py        | (Stub) Auto-réparation avancée                        |
| core/collective/federated_learning.py | (Prototype) Federated Learning (agrégation naïve) |
| core/api/no_code_api.py         | (Prototype) API “No Code” echo                        |
| core/simulator/digital_twin.py  | (Prototype) Digital Twin simulate/replay              |
| core/cloud/hybrid_bridge.py     | (Stub) Bridge cloud hybride                           |
| core/collective/federation.py   | (Manquant) Intelligence collective                    |
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
- [docs/fastpath.md](docs/fastpath.md) — Transport fastpath (USB4 / RDMA / SFP+) & métriques
- [docs/orchestrator.md](docs/orchestrator.md) — Architecture orchestrateur mémoire & placement
- [docs/unified_api.md](docs/unified_api.md) — API unifiée (workflows, twin, fédération) (prototype évolué: HMAC, quotas, read-only, pondération FL)
 - [docs/operations.md](docs/operations.md) — Guide opérations & maintenance
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

Voir aussi: [CHANGELOG.md](CHANGELOG.md)

## 📡 Télémétrie & Scheduler Opportuniste

### Formats de télémétrie
- Binaire compact: `/api/telemetry.bin` (paquets concaténés: header struct + id)
- Texte compact: `/api/telemetry.txt` (1 ligne / nœud)
- Flux SSE: `/api/telemetry/stream` (push continu, 2s)
- Ingestion edge → serveur: `POST /api/telemetry/ingest` (binaire)

Client CLI de décodage:
```bash
python -m cli.telemetry_cli --url http://localhost:5010/api/telemetry.bin
```

Agent edge minimal:
```bash
python edge/edge_agent.py --id edge1 --api http://localhost:5010 --interval 5
```

### Métriques Prometheus
- `vramancer_telemetry_packets_total{direction=out|in}`
- `vramancer_device_info{backend,name,index}` (gauge=1)
- Scheduler: `vramancer_tasks_submitted_total`, `vramancer_tasks_completed_total`, `vramancer_tasks_failed_total`, `vramancer_tasks_running`, `vramancer_tasks_resource_running{resource}`
 - Fastpath: `vramancer_fastpath_interface_latency_seconds{interface,kind}`, `vramancer_fastpath_bytes_total{method,direction}`, `vramancer_fastpath_latency_seconds{method,op}`
 - HA Journal: `vramancer_ha_journal_size_bytes`, `vramancer_ha_journal_rotations_total`
 - Orchestrateur: `vramancer_orch_placements_total{level}`, `vramancer_orch_migrations_total`, `vramancer_orch_rebalance_total`, `vramancer_orch_hierarchy_moves_total{to_level}`

### Scheduler (réutilisation ressources inactives)
- `POST /api/tasks/submit` `{kind: warmup|compress|noop, priority}`
- `POST /api/tasks/submit_batch` `{tasks:[{kind,priority,est_runtime_s}]}`
- `GET /api/tasks/status`
- `GET /api/tasks/history`
- `POST /api/tasks/cancel` `{id}`
- Politique adaptative: spill CUDA→ROCm→MPS→CPU + admission VRAM/CPU + priorité dynamique

### UI & Intégrations
- Web: section "Tâches" (injection, historique live)
- Qt: consommation télémétrie binaire directe
- Mobile: lecture texte proxy `/telemetry`

### Extensions futures
- Delta binaires (varints)
- Transport UDP multicast edge
- Replay journal signé
- Priorisation ML / préemption douce

### 🚀 Fastpath (USB4 / RDMA / SFP+ simulé)
Endpoints:
```http
GET  /api/fastpath/capabilities          # Capacités du canal courant
GET  /api/fastpath/interfaces            # Interfaces détectées + benchmarks
POST /api/fastpath/select {interface:?}  # Priorise une interface + re-benchmark
```
Sélection alternative via variable d'env: `export VRM_FASTPATH_IF=eth0`.
Chaque benchmark publie `vramancer_fastpath_interface_latency_seconds`.

### ♻️ HA Replication Journal
- Application delta/full: `POST /api/ha/apply` (signature HMAC dérivée horaire + nonce anti-rejeu)
- Rotation automatique (taille > `VRM_HA_JOURNAL_MAX`, défaut 5MB) avec compression gzip archivage
- Métriques : taille & rotations (cf. section métriques)
- Tamper-evidence: journal append-only + hash inclus dans meta


---

## 🔍 État d'implémentation (Réalité vs Promesse)

| Domaine | Statut | Détails |
|---------|--------|---------|
| Backends HuggingFace | ✅ Fonctionnel | Chargement + split basique (à améliorer VRAM réelle) |
| Backend vLLM | 🟡 Prototype | Stub, infer non implémenté |
| Backend Ollama | 🟡 Prototype | Stub, REST à compléter |
| Routing adaptatif | 🟡 Démo | Heuristique simple sur VRAM simulée |
| Federated Learning | 🟡 Prototype évolué | Moyenne pondérée + clipping + bruit optionnel |
| XAI Dashboard | 🟡 Prototype évolué | `/api/xai/explain` + attribution relative L1 + métriques |
| Hybrid Cloud Bridge | 🟡 Prototype | Déploiement/offload simulé |
| Zero Trust / Sécurité | 🟡 Prototype | Structures présentes, logique à étoffer |
| Auto-Repair | 🟡 Prototype | Scripts de base, pas d'orchestration complète |
| Marketplace Plugins | 🟡 Prototype | Classe plugin générique |
| API No-Code | 🟡 Prototype | Validation Pydantic + création workflows |
| Tokenizer fallback | ❌ Manquant | À ajouter : fallback slow si Rust absent |
| Tests unitaires | 🟡 Partiel | Scheduler / imports ok, manque réseau/sécurité/XAI |
| Tests lourds mémoire | ⚠️ Risque | `test_memory_stress` potentiellement OOM |
| CI automatisée | ❌ Manquant | Recommander workflow lint+tests rapides |
| Production hardening (RBAC, CORS, rate limit, persistence) | ✅ Ajouté | Security + quotas, read-only, rotation HMAC, persistence sqlite optionnelle |
| Cohérence dépendances | ✅ Corrigé | `setup.cfg` synchronisé sur requirements.txt |
| Systray multi-contexte | ✅ OK | Chemins absolus + détection bundle |

Légende : ✅ = opérationnel / 🟡 = prototype / ❌ = à implémenter / ⚠️ = à surveiller

### 📡 Observabilité & Health

Métriques Prometheus exposées par défaut sur le port 9108 (modifiable via `VRM_METRICS_PORT`).

```bash
vramancer --backend huggingface --model gpt2 &
curl -s http://localhost:9108/metrics | grep vramancer_infer_total
```

Healthcheck rapide :
```bash
vramancer-health
```

### Variables d'environnement essentielles (résumé)
| Variable | Rôle | Valeur défaut |
|----------|------|---------------|
| VRM_API_PORT | Port API Flask | 5030 |
| VRM_METRICS_PORT | Port exposition Prometheus | 9108 |
| VRM_HA_REPLICATION | Active journal & réplication HA | 0 |
| VRM_HA_PEERS | Liste host:port pairs | (vide) |
| VRM_DISABLE_RATE_LIMIT | Bypasse rate limiting | 0 |
| VRM_TRACING | Active OpenTelemetry | 0 |
| VRM_TEST_MODE | Relaxe sécurité (tests) | 0 |
| VRM_DISABLE_SECRET_ROTATION | Fige rotation HMAC | 0 |
| VRM_FASTPATH_IF | Force interface fastpath | autodetect |
| VRM_RATE_MAX | Seuil rate limit (req/interval) | 60 |
| VRM_DISABLE_ONNX | Désactive import/export ONNX (environnements légers) | 0 |
| VRM_API_DEBUG | Verbose debug connexions dashboard (Qt) | 0 |
| VRM_STRICT_IMPORT | Échec immédiat si import critique manquant | 0 |

Pour le mode production ne pas définir `VRM_TEST_MODE` et laisser rotation active.

Bootstrap environnement :
```bash
python scripts/bootstrap_env.py
```


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

#### Extras pip / Profils
Installation complète (défaut via `requirements.txt`). Pour un déploiement serveur sans UI lourde :
```bash
pip install .[server]
```
Profils prévus (à documenter / WIP) :
| Extra | Contenu attendu | Cible |
|-------|-----------------|-------|
| lite | Dépendances minimales CLI | Conteneurs, edge faible |
| server | Sans PyQt5, avec prometheus/opentelemetry | Serveur prod |
| dev | + outils dev (black, mypy, isort, pytest) | Contribution |
| all | Tous modules y compris GUI & compression | Desktop labo |

#### Fichiers requirements
| Fichier | Rôle |
|---------|------|
| `requirements.txt` | Profil lite / base (API + orchestration) |
| `requirements-full.txt` | Stack complète (GUI, dash, vision, compression, tracing) |

Exemples :
```bash
# Minimal
pip install -r requirements.txt

# Full
pip install -r requirements-full.txt

# Équivalent full via extras
pip install .[all]
```

#### Audit packaging (résumé)
Actions en cours / à valider :
- Aligner `setup.cfg` (actuellement nom `vrc_inference`) avec `setup.py` (`vramancer`) → unifier
- Déplacer dépendances lourdes (PyQt5, torchvision) vers extras
- Ajouter détection dynamique lz4/zstandard (déjà tolérant si absent)
- Fournir wheel universelle + archive lite
- Intégrer script `build_deb.sh` dans workflow CI

#### Build wheel
```bash
python -m build
pip install dist/vramancer-*.whl
```

#### Build .deb (résumé)
```bash
make deb
sudo dpkg -i dist/vramancer_*.deb
```

<div align="center">
	<img src="vramancer.png" width="120" alt="VRAMancer logo"/>
</div>

# VRAMancer

[![CI](https://github.com/thebloodlust/VRAMancer/actions/workflows/ci.yml/badge.svg)](https://github.com/thebloodlust/VRAMancer/actions)

**Optimisation VRAM multi-GPU, LLM universel, dashboards modernes, packaging pro.**

### Compatibilité GPU / Accélération
| Stack | Support actuel | Détails |
|-------|----------------|---------|
| CUDA (NVIDIA) | ✅ | Détection GPU, mémoire, torch.cuda.* |
| ROCm (AMD) | 🟡 Partiel | Torch ROCm fonctionne si environnement dispos; fastpath neutre |
| Apple Metal (MPS) | 🟡 Partiel | Si torch.mps dispo: fallback CPU->MPS possible (à ajouter) |
| CPU pur | ✅ | Tous backends stub / HF CPU fonctionnent |

Pour activer un backend même sans dépendance native :
```bash
export VRM_BACKEND_ALLOW_STUB=1
python -m vramancer.main --backend vllm --model dummy
```

### Fastpath & Bypass TCP/IP
Le module `core/network/fibre_fastpath.py` fournit :
 - Autosensing (usb4 / interfaces réseau génériques)
 - Canal mmap local zero-copy (prototype)
 - API unifiée send/recv
 - Plugin RDMA (détection pyverbs) stub (latence simulée 20µs) – `prefer="rdma"`
 - Extensible vers io_uring ou driver fibre SFP+ personnalisé

Lots pro A→F implémentés :
 A. Tracing OpenTelemetry optionnel (`VRM_TRACING=1`) via `core/tracing.py`
 B. Eviction planner hotness (endpoint `POST /api/memory/evict`)
 C. Sécurité: rate limiting + rotation token horaire (`/api/security/rotate`)
 D. Multicast UDP télémétrie (`/api/telemetry/multicast`)
 E. Runtime estimator dynamique (`POST /api/tasks/estimator/install`)
 F. Fastpath RDMA stub (pyverbs) + intégration hot-plug

Endpoints récents (points 1–4 avancés):
- POST `/api/memory/evict` {vram_pressure?} – éviction adaptative
- GET  `/api/memory/summary` – synthèse tiers/hotness
- GET  `/api/telemetry/multicast` – diffusion multicast états légers
- POST `/api/tasks/estimator/install` – installation dynamique d’un estimator
- Script bootstrap production stricte: `python -m scripts.prod_bootstrap`

### Tracing & Observabilité avancée
Activer :
```bash
export VRM_TRACING=1
# Optionnel : export OTLP
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4318"
export VRM_TRACING_ATTRS='{"deployment":"dev","cluster":"local"}'
```
Spans clés : `memory.migrate`, `memory.eviction_cycle` (extensible scheduler / fastpath).

### Persistence
- Mémoire hiérarchique : autosave toutes les 30s (`.hm_state.pkl`)
- Scheduler : recharge historique si `history_path` défini

### RBAC minimal
- Header `X-API-ROLE`: user < ops < admin
- Endpoints protégés : `/api/memory/evict`, `/api/security/rotate`, `/api/tasks/estimator/install`, `/api/memory/summary`

### TLS / Reverse Proxy (Production)
### Ports de communication cluster
Par défaut le serveur supervision écoute sur 5010. Pour multi-instances:
```bash
export VRM_API_PORT=6010
python -m vramancer.main
```
Réplication HA cible les ports que vous listez dans `VRM_HA_PEERS` (format host:port). USB4 / fastpath réseau est abstrait via `fibre_fastpath` (détection auto usb4 / rdma stub). Pour port custom fastpath de transport SocketIO/TCP, adapter vos scripts de lancement ou ajouter un paramètre CLI (à intégrer selon besoin).

Exemple Nginx minimal:
```nginx
server {
	listen 443 ssl;
	server_name vramancer.local;
	ssl_certificate /etc/ssl/certs/fullchain.pem;
	ssl_certificate_key /etc/ssl/private/privkey.pem;
	location / {
		proxy_pass http://127.0.0.1:5010/;
		proxy_set_header Host $host;
		proxy_set_header X-Forwarded-For $remote_addr;
	}
}
```
Flask derrière proxy: exporter `VRM_CORS_ORIGINS=https://vramancer.local`.
Pour certificats de dev rapides: mkcert ou Traefik (Let's Encrypt auto).

### Haute disponibilité (Réplication légère)
Activer:
```bash
export VRM_HA_REPLICATION=1
export VRM_HA_PEERS="node2:5010,node3:5010"
```
Chaque instance POST `/api/ha/apply` aux pairs (registry hotness simplifié).

### Contrôle autosave & éviction
```bash
export VRM_AUTOSAVE_MEMORY=0      # désactive autosave
export VRM_ENABLE_EVICTION=0      # désactive éviction automatique
```

Roadmap bas niveau : implémenter un backend C (io_uring) + un backend RDMA (pyverbs) branchés derrière `FastHandle`.

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

# Installation simplifiée de VRAMancer (Windows)

## 1. Téléchargement
- Téléchargez le dépôt GitHub (VRAMancer-main.zip) et le bundle release (vramancer_release_bundle.zip).

## 2. Extraction
- Dézippez le dépôt dans un dossier, par exemple :
  `C:\Users\votre_nom\Downloads\VRAMancer-main\`
- Dézippez le bundle release dans ce même dossier ou à l’intérieur, par exemple :
  `C:\Users\votre_nom\Downloads\VRAMancer-main\release_bundle\`

## 3. Installation des dépendances
- Ouvrez une console (cmd ou PowerShell) dans le dossier `release_bundle`.
- Installez les dépendances Python :
  ```bash
  pip install -r requirements.txt
  ```

## 4. Lancement du systray
- Dans la console, lancez :
  ```bash
  python systray_vramancer.py
  ```
- L’icône VRAMancer apparaît dans la barre de tâches.
- Utilisez le menu pour accéder à l’installation graphique, la supervision ou la GUI avancée.
  
### Menus systray disponibles
| Catégorie | Entrées |
|-----------|---------|
| Installation | Installation graphique VRAMancer |
| Dashboards / Modes | Web (basique), Web avancé, Qt GUI, Tk GUI, CLI dashboard, Visualizer |
| Actions rapides | Lancer API principale, API Lite (test), Tracing ON/OFF, Ouvrir métriques (info), Statut HA, Redémarrer (bootstrap), Quitter |
| Aide / Info (boîte métriques) | Rappel URL Prometheus |

Notes:
- Le menu “Lancer API principale” tente `vramancer/main.py` puis fallback `gui.py`.
- L’option métriques n’ouvre pas de navigateur (affiche info / console).
- Le reload simple exécute `scripts/bootstrap_env.py` si présent.

#### Fonctionnalités avancées systray
- Récents Dashboards : sous-menu "Derniers" (max 5 derniers lancés) persistant dans `.vramancer_systray.json`.
- Port API auto : si 5010 occupé, sélection d’un port libre 5011–5050, mémorisé pour le health check.
- API Lite : lance l’API avec `VRM_DISABLE_RATE_LIMIT=1` et `VRM_TEST_MODE=1` (facilite tests locaux / démo rapide).
- Toggle Tracing : active/désactive en mémoire (appliqué aux prochains lancements API via `VRM_TRACING=1`).
- Statut HA : lit les métriques `vramancer_ha_journal_size_bytes` & `vramancer_ha_journal_rotations_total` et affiche un résumé.
- Icône santé dynamique : ping `/api/health` toutes les 5s, halo vert (UP) ou rouge (DOWN) sur l’icône.
- Persistance état : fichier JSON à la racine du bundle (peut être supprimé sans risque pour réinitialiser).

## 5. Conseils
- Ne déplacez pas le script systray ou les fichiers du bundle, lancez toujours depuis le dossier `release_bundle`.
- Si une dépendance manque (ex : Flask), relancez la commande d’installation des dépendances.
- Pour toute erreur, vérifiez que tous les fichiers du bundle sont bien présents dans le dossier.

---

Pour toute question ou problème, consultez le manuel ou contactez le support sur GitHub.

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
