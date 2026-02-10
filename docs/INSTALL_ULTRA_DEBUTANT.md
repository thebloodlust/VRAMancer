# Guide d'installation VRAMancer — Du zéro à l'inférence

Ce guide vous accompagne pas à pas. Aucune connaissance préalable n'est nécessaire.

---

## Pré-requis

| Élément | Minimum | Recommandé |
|---------|---------|------------|
| OS | Windows 10, macOS 12, Ubuntu 20.04 | Ubuntu 22.04+ / Windows 11 |
| Python | 3.10 | 3.11 ou 3.12 |
| GPU | Aucun (CPU fonctionne) | NVIDIA (CUDA 12+) ou AMD (ROCm 6+) ou Apple Silicon |
| RAM | 8 GB | 16 GB+ |
| Disque | 2 GB (+ espace modèle) | SSD NVMe recommandé |

> **Pas de GPU ?** VRAMancer fonctionne en mode CPU. Les tests passent à 100% sans aucun GPU.

---

## 1. Télécharger le projet

**Option A — Git (recommandé)**
```bash
git clone https://github.com/thebloodlust/VRAMancer.git
cd VRAMancer
```

**Option B — ZIP (pas de git)**
1. Allez sur https://github.com/thebloodlust/VRAMancer
2. Bouton vert **Code** → **Download ZIP**
3. Décompressez et ouvrez un terminal dans le dossier

---

## 2. Installer

### Linux / macOS

```bash
# Créer un environnement isolé
python3 -m venv .venv
source .venv/bin/activate

# Installer VRAMancer
pip install -e .
```

### Windows (PowerShell)

```powershell
# Créer un environnement isolé
python -m venv .venv
.venv\Scripts\Activate.ps1

# Installer VRAMancer
pip install -e .
```

### Docker (production — le plus simple)

```bash
export VRM_API_TOKEN=mon-token-secret
docker compose up -d
```

C'est tout. Quatre services démarrent :

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:5030 | Inference OpenAI-compatible |
| Grafana | http://localhost:3000 | Dashboards (admin / vramancer) |
| Prometheus | http://localhost:9090 | Métriques temps réel |
| Alertmanager | http://localhost:9093 | Alertes GPU/API/latence |

---

## 3. Vérifier l'installation

```bash
# Vérifier que tout est importable
python -c "import core; print(f'VRAMancer v{core.__version__} OK')"

# Lancer les tests (aucun GPU requis, ~30 secondes)
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 \
  pytest tests/ -q --no-cov
```

Résultat attendu : **436 passed, 9 skipped**.

---

## 4. Premier lancement

### Démarrer le serveur API

```bash
export VRM_API_TOKEN=mon-token-secret
python -m vramancer.main --api
```

Le serveur démarre sur http://localhost:5000.

### Charger un modèle

```bash
curl -X POST http://localhost:5000/api/models/load \
  -H "Content-Type: application/json" \
  -H "X-API-TOKEN: $VRM_API_TOKEN" \
  -d '{"model": "gpt2", "num_gpus": 1}'
```

> **Premier modèle ?** Commencez par `gpt2` (500 MB). Pour du sérieux : `meta-llama/Llama-3.1-8B` (2 GPUs, ~16 GB VRAM total).

### Générer du texte

```bash
# Complétion simple
curl http://localhost:5000/v1/completions \
  -H "Content-Type: application/json" \
  -H "X-API-TOKEN: $VRM_API_TOKEN" \
  -d '{"prompt": "La vie est", "max_tokens": 50}'

# Chat (format OpenAI)
curl http://localhost:5000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-TOKEN: $VRM_API_TOKEN" \
  -d '{
    "messages": [{"role": "user", "content": "Explique le machine learning en 3 phrases"}],
    "max_tokens": 200
  }'
```

---

## 5. Multi-GPU (le cœur de VRAMancer)

Si vous avez 2+ GPUs (même de marques/tailles différentes) :

```bash
# Charger un modèle sur 2 GPUs — split automatique
curl -X POST http://localhost:5000/api/models/load \
  -H "Content-Type: application/json" \
  -H "X-API-TOKEN: $VRM_API_TOKEN" \
  -d '{"model": "meta-llama/Llama-3.1-8B", "num_gpus": 2}'
```

VRAMancer détecte automatiquement la VRAM libre de chaque GPU et distribue les couches proportionnellement. Un RTX 3090 (24GB) reçoit 60% des couches, un RTX 5070 Ti (16GB) reçoit 40%.

### Vérifier la répartition

```bash
curl http://localhost:5000/api/pipeline/status \
  -H "X-API-TOKEN: $VRM_API_TOKEN" | python -m json.tool
```

---

## 6. Multi-machine (cluster)

### Machine A (master)

```bash
python -m vramancer.main --api --cluster-master
```

### Machine B (worker — se connecte tout seul via mDNS)

```bash
python -m vramancer.main --cluster-worker
```

> Les machines doivent être sur le même réseau local. La découverte est automatique via mDNS.

### Vérifier les nœuds

```bash
curl http://localhost:5000/api/nodes \
  -H "X-API-TOKEN: $VRM_API_TOKEN"
```

---

## 7. Monitoring

### Sans Docker

Les métriques Prometheus sont exposées automatiquement sur le port 9108 :

```bash
curl http://localhost:9108/metrics | grep vramancer_
```

47 métriques disponibles : GPU memory, inference latency, VRAM lending, batcher throughput, etc.

### Avec Docker (recommandé)

```bash
docker compose up -d
```

Ouvrez http://localhost:3000 → connectez-vous (`admin` / `vramancer`) → dashboard "VRAMancer — Multi-GPU Inference" avec 24 panneaux en temps réel.

16 règles d'alerte surveillent : GPU memory >90%, inference errors >5%, API down, task queue backlog, etc.

### Dashboard CLI (sans navigateur)

```bash
python -m dashboard.cli_dashboard
```

Affichage ASCII temps réel des GPUs, mémoire et santé du cluster.

---

## 8. Installer PyTorch pour votre GPU

VRAMancer a besoin de PyTorch. La version dépend de votre matériel :

### NVIDIA (CUDA)

```bash
# CUDA 12.1 (le plus courant)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### AMD (ROCm)

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
```

### Apple Silicon (MPS)

```bash
pip install torch  # MPS activé automatiquement sur macOS ARM
```

### CPU uniquement

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 9. Guide de dépannage

| Problème | Solution |
|----------|----------|
| `ModuleNotFoundError: No module named 'core'` | Lancez `pip install -e .` depuis la racine du projet |
| `torch.cuda.is_available()` retourne `False` | Installez les drivers NVIDIA + PyTorch CUDA (voir §8) |
| `Address already in use` | Un autre serveur tourne. `kill $(lsof -ti:5000)` ou changez le port : `VRM_API_PORT=5050` |
| `401 Unauthorized` | Ajoutez le header : `-H "X-API-TOKEN: $VRM_API_TOKEN"` |
| `tokenizers` échoue sur Windows | `pip install tokenizers --no-build-isolation` ou installez Rust : https://rustup.rs |
| Tests échouent | Vérifiez : `VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 pytest tests/ -q` |
| `ImportError: psutil` | `pip install psutil` |
| GPU non détecté après ajout à chaud | VRAMancer détecte le hot-plug, vérifiez `/api/gpu` |

---

## 10. Prochaines étapes

- **Benchmarker votre setup** : `curl -X POST http://localhost:5000/api/benchmark -H "X-API-TOKEN: $VRM_API_TOKEN" -d '{"mode": "synthetic"}'`
- **Configurer les alertes** : modifiez `monitoring/alertmanager.yml` pour recevoir les notifications (Slack, PagerDuty, email)
- **Explorer la config** : `cat config.yaml` — toutes les options sont documentées
- **Lire la doc API** : `docs/api.md` ou `docs/quickstart.md`

---

## Désinstallation

```bash
# Si installé avec pip
pip uninstall vramancer

# Si Docker
docker compose down -v

# Nettoyage complet
rm -rf .venv/
```

---

## Résumé des commandes

```bash
# Installation
git clone https://github.com/thebloodlust/VRAMancer.git && cd VRAMancer
python3 -m venv .venv && source .venv/bin/activate
pip install -e .

# Tests
VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1 pytest tests/ -q

# Lancement
export VRM_API_TOKEN=mon-token
python -m vramancer.main --api

# Chargement modèle + génération
curl -X POST http://localhost:5000/api/models/load \
  -H "Content-Type: application/json" -H "X-API-TOKEN: $VRM_API_TOKEN" \
  -d '{"model": "gpt2", "num_gpus": 1}'

curl http://localhost:5000/v1/completions \
  -H "Content-Type: application/json" -H "X-API-TOKEN: $VRM_API_TOKEN" \
  -d '{"prompt": "Hello", "max_tokens": 50}'
```
