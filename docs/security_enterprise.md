# Sécurité, conformité et interopérabilité entreprise

## 1. Certification et conformité (RGPD, HIPAA, ISO)
- **Module** : `core/security/compliance.py`
- **Fonctions** :
  - Logging sécurisé des accès et actions
  - Audit des événements sensibles
  - Anonymisation des données (masquage des identifiants)
  - Gestion des accès et consentements utilisateurs
- **Exemple** :
```python
from core.security.compliance import ComplianceLogger
compliance = ComplianceLogger()
compliance.log_access("alice", "read", "model_weights")
compliance.log_audit("export", {"user": "bob", "file": "data.csv"})
print(compliance.anonymize({"user_id": "bob", "data": 42}))
```

## 2. Accès distant sécurisé (poste de contrôle web)
- **Module** : `core/security/remote_access.py`
- **Fonctions** :
  - Authentification forte (MFA)
  - Gestion des rôles (admin, user...)
  - API Flask pour contrôle distant sécurisé
- **Exemple** :
```bash
python3 core/security/remote_access.py
```
Puis POST sur `/login` avec JSON `{ "user": "admin", "password": "adminpass", "mfa": "123456" }`

## 3. Interopérabilité entreprise (LDAP/Active Directory)
- **Module** : `core/security/ldap_auth.py`
- **Fonctions** :
  - Authentification via LDAP/AD
  - Récupération des rôles utilisateurs
- **Exemple** :
```python
from core.security.ldap_auth import LDAPAuthenticator
ldap = LDAPAuthenticator("ldap://ldap.entreprise.fr", "ou=users,dc=entreprise,dc=fr")
if ldap.authenticate("alice", "motdepasse"):
    print("Authentifié!")
```

---

**Intégration dashboard/web** :
- ComplianceLogger peut être appelé à chaque action sensible (export, accès modèle, etc.)
- Le poste de contrôle web (`remote_access.py`) peut servir de portail d’admin distant (MFA, rôles, audit)
- LDAPAuthenticator s’intègre dans le backend d’authentification (API, dashboard, etc.)

**À adapter selon les besoins de conformité et d’intégration de chaque client.**
