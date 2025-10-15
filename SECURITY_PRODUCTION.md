"""
VRAMancer - Guide de Sécurité pour Production

⚠️  ATTENTION : Actions OBLIGATOIRES avant déploiement en production
"""

# =============================================================================
# 🔒 SÉCURITÉ CRITIQUE
# =============================================================================

## 1. Changer le mot de passe admin par défaut

Le compte admin par défaut utilise `admin/admin` - **DANGEREUX EN PRODUCTION**

### Solution :

```bash
# Méthode 1 : Via l'API au premier démarrage
curl -X POST http://localhost:5030/api/users/change-password \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "old_password": "admin",
    "new_password": "VOTRE_MOT_DE_PASSE_FORT_ICI"
  }'

# Méthode 2 : Variable d'environnement (désactive le compte par défaut)
export VRM_DISABLE_DEFAULT_ADMIN=1
export VRM_ADMIN_USERNAME=votre_admin
export VRM_ADMIN_PASSWORD_HASH=$(python3 -c "from core.auth_strong import _hash_password; import secrets; salt=secrets.token_hex(8); print(salt + ':' + _hash_password('VotreMotDePasseFort', salt))")
```

## 2. Définir VRM_AUTH_SECRET (OBLIGATOIRE)

Le secret JWT est auto-généré en mode dev. **INACCEPTABLE EN PRODUCTION**

### Solution :

```bash
# Générer un secret fort (32+ caractères)
export VRM_AUTH_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# OU utiliser openssl
export VRM_AUTH_SECRET=$(openssl rand -hex 32)

# Ajouter au fichier .env (JAMAIS versionner ce fichier !)
echo "VRM_AUTH_SECRET=$(openssl rand -hex 32)" >> .env.production
```

## 3. Désactiver le mode debug

### Vérifier :

```bash
# Ces variables DOIVENT être à 0 en production
export VRM_API_DEBUG=0
export VRM_TEST_MODE=0
export VRM_TEST_RELAX_SECURITY=0
export VRM_DISABLE_RATE_LIMIT=0
```

## 4. Configurer HTTPS/TLS

### Pour Nginx (recommandé) :

```nginx
server {
    listen 443 ssl http2;
    server_name vramancer.votre-domaine.com;

    ssl_certificate /etc/letsencrypt/live/votre-domaine.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/votre-domaine.com/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    location / {
        proxy_pass http://127.0.0.1:5030;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 5. Rate Limiting et Quotas

```bash
# Configurer les limites
export VRM_RATE_MAX=100                    # Max requêtes par fenêtre
export VRM_UNIFIED_API_QUOTA=1             # Activer les quotas
export VRM_QUOTA_DEFAULT=1000              # Quota par défaut
```

## 6. Logging sécurisé

```bash
# Logs structurés JSON (parsable par ELK, Splunk, etc.)
export VRM_LOG_JSON=1
export VRM_LOG_LEVEL=INFO                  # Pas DEBUG en prod !
export VRM_LOG_DIR=/var/log/vramancer
export VRM_REQUEST_LOG=0                   # Ne pas logger toutes les requêtes
```

## 7. Isolation réseau (Docker)

```yaml
# docker-compose.production.yml
version: '3.8'

services:
  vramancer:
    image: vramancer:1.0.0
    restart: always
    environment:
      - VRM_AUTH_SECRET=${VRM_AUTH_SECRET}
      - VRM_API_DEBUG=0
      - VRM_LOG_JSON=1
      - VRM_DISABLE_DEFAULT_ADMIN=1
    networks:
      - vramancer_internal
    ports:
      - "127.0.0.1:5030:5030"  # N'écoute que sur localhost
    volumes:
      - vramancer_logs:/var/log/vramancer
      - vramancer_data:/app/data
    read_only: true
    tmpfs:
      - /tmp
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE

networks:
  vramancer_internal:
    driver: bridge
    internal: true

volumes:
  vramancer_logs:
  vramancer_data:
```

# =============================================================================
# 📋 CHECKLIST PRÉ-PRODUCTION
# =============================================================================

Avant de déployer, vérifiez :

- [ ] Mot de passe admin changé
- [ ] VRM_AUTH_SECRET défini (32+ caractères aléatoires)
- [ ] VRM_API_DEBUG=0
- [ ] VRM_TEST_MODE=0
- [ ] HTTPS/TLS configuré (certificat valide)
- [ ] Rate limiting activé
- [ ] Logs sécurisés (JSON, rotation configurée)
- [ ] Firewall configuré (ports minimaux ouverts)
- [ ] Backups automatiques configurés
- [ ] Monitoring activé (Prometheus/Grafana)
- [ ] Alerting configuré (erreurs critiques)
- [ ] Documentation opérationnelle à jour

# =============================================================================
# 🔐 VARIABLES D'ENVIRONNEMENT - RÉFÉRENCE COMPLÈTE
# =============================================================================

## Authentification (CRITIQUE)

```bash
VRM_AUTH_SECRET                 # Secret JWT (32+ chars) - OBLIGATOIRE
VRM_AUTH_EXP                    # Durée access token (s) - défaut: 900
VRM_AUTH_REFRESH_EXP            # Durée refresh token (s) - défaut: 86400
VRM_DISABLE_DEFAULT_ADMIN       # Désactiver admin/admin - défaut: 0
```

## Sécurité

```bash
VRM_API_TOKEN                   # Token API (HMAC) - optionnel
VRM_DISABLE_SECRET_ROTATION     # Désactiver rotation HMAC - défaut: 0
VRM_DISABLE_RATE_LIMIT          # Désactiver rate limit - défaut: 0
VRM_RATE_MAX                    # Max requêtes/fenêtre - défaut: 200
VRM_TEST_MODE                   # Mode test (relax sécurité) - défaut: 0
VRM_TEST_RELAX_SECURITY         # Bypass token/HMAC - défaut: 0
```

## API

```bash
VRM_API_HOST                    # Host API - défaut: 0.0.0.0
VRM_API_PORT                    # Port API - défaut: 5030
VRM_API_DEBUG                   # Mode debug - défaut: 0
VRM_API_BASE                    # Base URL - défaut: http://localhost:5030
```

## Logging

```bash
VRM_LOG_LEVEL                   # Niveau (DEBUG/INFO/WARNING/ERROR) - défaut: INFO
VRM_LOG_JSON                    # Format JSON - défaut: 0
VRM_LOG_CONSOLE                 # Sortie console - défaut: 1
VRM_LOG_DIR                     # Répertoire logs - défaut: logs/
VRM_REQUEST_LOG                 # Logger toutes requêtes - défaut: 0
```

## Quotas

```bash
VRM_UNIFIED_API_QUOTA           # Activer quotas - défaut: 0
VRM_QUOTA_DEFAULT               # Quota par défaut - défaut: 1000
```

## Haute Disponibilité

```bash
VRM_HA_REPLICATION              # Activer réplication - défaut: 0
VRM_HA_PEERS                    # Liste peers (host:port,host:port)
VRM_READ_ONLY                   # Mode lecture seule - défaut: 0
```

## Monitoring

```bash
VRM_METRICS_PORT                # Port Prometheus - défaut: 9108
VRM_TRACING                     # OpenTelemetry - défaut: 0
```

# =============================================================================
# 🚨 INCIDENTS COURANTS ET SOLUTIONS
# =============================================================================

## Incident 1 : "Admin password is default"

**Cause** : Le mot de passe admin n'a pas été changé

**Solution** :
```bash
curl -X POST http://localhost:5030/api/users/change-password \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"old_password": "admin", "new_password": "NouveauMotDePasse"}'
```

## Incident 2 : "VRM_AUTH_SECRET not set"

**Cause** : Variable d'environnement manquante

**Solution** :
```bash
export VRM_AUTH_SECRET=$(openssl rand -hex 32)
# Redémarrer l'API
```

## Incident 3 : "Rate limit exceeded"

**Cause** : Trop de requêtes

**Solution** :
```bash
# Augmenter la limite temporairement
export VRM_RATE_MAX=500
# OU désactiver (DEV SEULEMENT)
export VRM_DISABLE_RATE_LIMIT=1
```

## Incident 4 : "Token expired"

**Cause** : Token JWT expiré (défaut: 15 min)

**Solution** :
```bash
# Utiliser le refresh token
curl -X POST http://localhost:5030/api/token/refresh \
  -d '{"refresh": "VOTRE_REFRESH_TOKEN"}'
```

# =============================================================================
# 📞 SUPPORT ET ESCALADE
# =============================================================================

## Niveaux de support

1. **Documentation** : SECURITY_PRODUCTION.md (ce fichier)
2. **Issues GitHub** : https://github.com/thebloodlust/VRAMancer/issues
3. **Logs** : Vérifier /var/log/vramancer/*.log
4. **Monitoring** : http://localhost:9108/metrics (Prometheus)

## Logs critiques à vérifier

```bash
# Erreurs d'authentification
grep "auth.*error" /var/log/vramancer/api.log

# Tentatives de connexion échouées
grep "login.*failed" /var/log/vramancer/api.log

# Rate limiting
grep "rate limit" /var/log/vramancer/api.log

# Erreurs critiques
grep "CRITICAL" /var/log/vramancer/*.log
```

# =============================================================================
# ✅ VALIDATION FINALE
# =============================================================================

Script de validation automatique :

```bash
#!/bin/bash
# check_production_ready.sh

echo "🔍 Vérification configuration production..."

# Vérifier VRM_AUTH_SECRET
if [ -z "$VRM_AUTH_SECRET" ]; then
    echo "❌ VRM_AUTH_SECRET non défini"
    exit 1
fi

# Vérifier longueur secret
if [ ${#VRM_AUTH_SECRET} -lt 32 ]; then
    echo "❌ VRM_AUTH_SECRET trop court (< 32 caractères)"
    exit 1
fi

# Vérifier mode debug
if [ "$VRM_API_DEBUG" == "1" ]; then
    echo "❌ VRM_API_DEBUG=1 (doit être 0)"
    exit 1
fi

# Vérifier test mode
if [ "$VRM_TEST_MODE" == "1" ]; then
    echo "❌ VRM_TEST_MODE=1 (doit être 0)"
    exit 1
fi

# Vérifier rate limiting
if [ "$VRM_DISABLE_RATE_LIMIT" == "1" ]; then
    echo "⚠️  Warning: Rate limiting désactivé"
fi

# Tester l'API
HEALTH=$(curl -s http://localhost:5030/health | jq -r '.status')
if [ "$HEALTH" != "healthy" ]; then
    echo "❌ API health check failed"
    exit 1
fi

echo "✅ Configuration production validée"
exit 0
```

Exécuter :
```bash
chmod +x check_production_ready.sh
./check_production_ready.sh
```

# =============================================================================
# 🎓 FORMATION ÉQUIPE OPS
# =============================================================================

## Procédure de déploiement (résumé)

1. **Pré-déploiement** :
   - Backup de la config actuelle
   - Vérifier secrets définis
   - Tester en staging

2. **Déploiement** :
   - Déployer nouvelle version
   - Attendre health checks OK
   - Vérifier logs (5 premières minutes)

3. **Post-déploiement** :
   - Changer mot de passe admin
   - Valider endpoints critiques
   - Activer monitoring
   - Brief équipe

## Contacts d'urgence

- **Ops Lead** : [À définir]
- **Security Lead** : [À définir]
- **On-call** : [À définir]

---

**Document version** : 1.0.0
**Dernière mise à jour** : 2025-10-15
**Auteur** : VRAMancer Team
