# 🚀 Guide de Migration : Dev → Production

Ce guide explique comment migrer de la version "simple" (prototype) vers la version production-ready de VRAMancer.

## 📋 Table des matières

1. [Changements principaux](#changements-principaux)
2. [Migration de l'API](#migration-de-lapi)
3. [Migration du logging](#migration-du-logging)
4. [Configuration des secrets](#configuration-des-secrets)
5. [Tests de validation](#tests-de-validation)
6. [Déploiement](#déploiement)

---

## 🔄 Changements principaux

### Avant (Version Simple)

```python
# api_simple.py
print("🚀 Démarrage API...")
app.run(host='0.0.0.0', port=5030, debug=True)  # ❌ debug=True
```

### Après (Version Production)

```python
# core/production_api.py
from core.logger import get_logger
logger = get_logger('api.production')
logger.info("Démarrage API...")
app.run(host='0.0.0.0', port=5030, debug=False)  # ✅ debug=False
```

---

## 📦 Migration de l'API

### Étape 1 : Utiliser le nouveau point d'entrée

**Avant** :
```bash
python api_simple.py
```

**Après** :
```bash
# Mode production (recommandé)
export VRM_PRODUCTION=1
python api.py

# OU directement
python -m core.production_api
```

### Étape 2 : Vérifier les variables d'environnement

```bash
# Obligatoire pour production
export VRM_AUTH_SECRET=$(openssl rand -hex 32)
export VRM_API_DEBUG=0

# Recommandé
export VRM_LOG_JSON=1
export VRM_LOG_LEVEL=INFO
export VRM_DISABLE_DEFAULT_ADMIN=1
```

### Étape 3 : Tester l'API

```bash
# Health check
curl http://localhost:5030/health

# Readiness check
curl http://localhost:5030/ready

# Status
curl http://localhost:5030/api/status
```

---

## 📝 Migration du logging

### Dans vos fichiers Python

**Avant** (❌ À éviter) :
```python
print("Démarrage du service...")
print(f"Erreur: {error}")
```

**Après** (✅ Production-ready) :
```python
from core.logger import get_logger

logger = get_logger('mon_module')

logger.info("Démarrage du service...")
logger.error(f"Erreur: {error}", exc_info=True)
```

### Niveaux de logging

```python
logger.debug("Message de debug (développement)")
logger.info("Information générale")
logger.warning("Avertissement")
logger.error("Erreur non-fatale")
logger.critical("Erreur critique")
```

### Logging avec contexte

```python
from core.logger import log_with_context

log_with_context(
    logger,
    'info',
    'Requête traitée',
    user_id=123,
    endpoint='/api/gpu',
    duration_ms=45
)
```

---

## 🔐 Configuration des secrets

### Fichier `.env.production` (recommandé)

Créez un fichier `.env.production` :

```bash
# Authentification (OBLIGATOIRE)
VRM_AUTH_SECRET=votre_secret_32_caracteres_minimum_aleatoires

# API
VRM_API_HOST=0.0.0.0
VRM_API_PORT=5030
VRM_API_DEBUG=0

# Logging
VRM_LOG_LEVEL=INFO
VRM_LOG_JSON=1
VRM_LOG_DIR=/var/log/vramancer

# Sécurité
VRM_DISABLE_DEFAULT_ADMIN=1
VRM_RATE_MAX=200

# Quotas
VRM_UNIFIED_API_QUOTA=1
VRM_QUOTA_DEFAULT=1000
```

**⚠️ IMPORTANT** : Ne JAMAIS versionner ce fichier !

Ajoutez à `.gitignore` :
```
.env
.env.production
.env.local
*.env
```

### Charger les variables

```bash
# Méthode 1 : export manuel
export $(cat .env.production | xargs)

# Méthode 2 : source
set -a
source .env.production
set +a

# Méthode 3 : docker-compose
docker-compose --env-file .env.production up
```

---

## ✅ Tests de validation

### 1. Validation de configuration

```bash
# Exécuter le script de validation
./scripts/check_production_ready.sh

# Résultat attendu :
# ✅ Configuration production validée avec succès !
```

### 2. Test de l'API

```bash
# Test des endpoints critiques
curl http://localhost:5030/health | jq .
# {"status": "healthy", "service": "vramancer-api", "version": "1.0.0"}

curl http://localhost:5030/api/gpu | jq .
# {"cuda_available": true, "device_count": 1, ...}
```

### 3. Test d'authentification

```bash
# Login
RESPONSE=$(curl -s -X POST http://localhost:5030/api/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"VotreNouveauMotDePasse"}')

# Extraire le token
ACCESS_TOKEN=$(echo $RESPONSE | jq -r '.access')

# Test endpoint protégé
curl -H "Authorization: Bearer $ACCESS_TOKEN" \
  http://localhost:5030/api/workflows | jq .
```

### 4. Test de charge (optionnel)

```bash
# Installer apache bench
sudo apt-get install apache2-utils

# Test 1000 requêtes, 10 concurrent
ab -n 1000 -c 10 http://localhost:5030/health

# Vérifier rate limiting
ab -n 300 -c 50 http://localhost:5030/api/status
```

---

## 🚀 Déploiement

### Déploiement local (systemd)

Créez `/etc/systemd/system/vramancer.service` :

```ini
[Unit]
Description=VRAMancer AI Orchestrator
After=network.target

[Service]
Type=simple
User=vramancer
Group=vramancer
WorkingDirectory=/opt/vramancer
EnvironmentFile=/opt/vramancer/.env.production
ExecStart=/opt/vramancer/.venv/bin/python -m core.production_api
Restart=always
RestartSec=10

# Sécurité
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/log/vramancer /opt/vramancer/data

[Install]
WantedBy=multi-user.target
```

Activer :
```bash
sudo systemctl daemon-reload
sudo systemctl enable vramancer
sudo systemctl start vramancer
sudo systemctl status vramancer
```

### Déploiement Docker

Créez `Dockerfile.production` :

```dockerfile
FROM python:3.11-slim

# Variables d'environnement
ENV VRM_PRODUCTION=1 \
    VRM_API_DEBUG=0 \
    VRM_LOG_JSON=1 \
    PYTHONUNBUFFERED=1

# Créer utilisateur non-root
RUN useradd -m -u 1000 vramancer

WORKDIR /app

# Installer dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code
COPY --chown=vramancer:vramancer . .

# Changer vers utilisateur non-root
USER vramancer

# Port
EXPOSE 5030

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5030/health')"

# Démarrage
CMD ["python", "-m", "core.production_api"]
```

Build et run :
```bash
docker build -f Dockerfile.production -t vramancer:1.0.0 .

docker run -d \
  --name vramancer \
  --env-file .env.production \
  -p 127.0.0.1:5030:5030 \
  --restart unless-stopped \
  vramancer:1.0.0
```

### Déploiement Kubernetes

Créez `k8s/deployment.yaml` :

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vramancer
  namespace: ai-services
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vramancer
  template:
    metadata:
      labels:
        app: vramancer
    spec:
      containers:
      - name: vramancer
        image: vramancer:1.0.0
        ports:
        - containerPort: 5030
        env:
        - name: VRM_PRODUCTION
          value: "1"
        - name: VRM_AUTH_SECRET
          valueFrom:
            secretKeyRef:
              name: vramancer-secrets
              key: auth-secret
        livenessProbe:
          httpGet:
            path: /health
            port: 5030
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 5030
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: vramancer
  namespace: ai-services
spec:
  selector:
    app: vramancer
  ports:
  - port: 5030
    targetPort: 5030
  type: ClusterIP
```

Déployer :
```bash
# Créer le secret
kubectl create secret generic vramancer-secrets \
  --from-literal=auth-secret=$(openssl rand -hex 32) \
  -n ai-services

# Déployer
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/vramancer -n ai-services
```

---

## 📊 Monitoring

### Logs

```bash
# Voir les logs en temps réel
tail -f /var/log/vramancer/api.log

# Logs JSON (parsable)
tail -f /var/log/vramancer/api.log | jq .

# Filtrer les erreurs
tail -f /var/log/vramancer/api.log | jq 'select(.level=="ERROR")'
```

### Métriques (Prometheus)

```bash
# Métriques disponibles
curl http://localhost:9108/metrics

# Exemple de requêtes Prometheus
rate(api_latency_seconds_sum[5m])
```

---

## 🔄 Rollback

Si problème en production :

### Rollback rapide

```bash
# Systemd
sudo systemctl stop vramancer
sudo systemctl start vramancer-old

# Docker
docker stop vramancer
docker start vramancer-old

# Kubernetes
kubectl rollout undo deployment/vramancer -n ai-services
```

### Rollback complet

```bash
# Retour à api_simple.py
export VRM_PRODUCTION=0
python api.py
```

---

## 📚 Ressources

- **Documentation sécurité** : [SECURITY_PRODUCTION.md](SECURITY_PRODUCTION.md)
- **Script de validation** : `scripts/check_production_ready.sh`
- **Changelog** : [CHANGELOG.md](CHANGELOG.md)
- **Issues** : https://github.com/thebloodlust/VRAMancer/issues

---

## ❓ FAQ

### Q: Dois-je migrer immédiatement ?

**R**: Pour le développement, `api_simple.py` reste acceptable. Pour la production, la migration est **OBLIGATOIRE**.

### Q: La version production est-elle compatible avec l'ancienne ?

**R**: Oui, les endpoints sont 100% compatibles. Seule l'implémentation change.

### Q: Comment tester sans impacter la production ?

**R**: Utilisez des environnements séparés (dev, staging, prod) avec des `.env` différents.

### Q: Les performances sont-elles affectées ?

**R**: Le logging structuré a un overhead minime (~1-2%). L'authentification JWT ajoute ~5ms par requête.

---

**Version du guide** : 1.0.0  
**Dernière mise à jour** : 2025-10-15
