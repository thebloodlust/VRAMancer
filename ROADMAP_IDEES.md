
# Roadmap & Idées complémentaires VRAMancer

## Sécurité & conformité
- Authentification des nœuds (clé, LDAP, Active Directory)
- Chiffrement des transferts (AES, TLS)
- Gestion des accès, rôles, conformité RGPD/HIPAA/ISO

## Accès distant & supervision
- Poste de contrôle web sécurisé (MFA, gestion des rôles)
- Dashboard centralisé multi-cluster, alertes, reporting, SIEM
- Interface mobile/tablette pour monitoring et contrôle

## Interopérabilité entreprise
- Support LDAP/Active Directory
- API REST/GraphQL pour automatisation et DevOps

## Edge/IoT & cloud hybride
- Support nœuds légers (Raspberry Pi, Jetson, etc.)
- Orchestration IA en périphérie
- Bridge natif vers AWS, Azure, GCP
- Bascule dynamique local/cloud

## Intelligence collective & partage
- Fédération de clusters, P2P, partage de modèles/datasets/résultats

## Automatisation avancée
- Auto-réparation, haute disponibilité, reconfiguration dynamique
- Déploiement automatisé, gestion des tâches

## Marketplace & extensions
- Modules premium, connecteurs cloud, plugins tiers
- API publique, documentation vidéo/interactive

## Documentation & onboarding
- Tutoriels vidéo, guides interactifs, FAQ
- Accessibilité multilingue et pour tous

---

Toutes ces idées sont réalisables et peuvent rendre VRAMancer exceptionnel, universel et leader du marché IA distribué !

---

## Backlog (prochaine itération ciblée)

### Auth & Sécurité avancées
- Endpoint gestion utilisateurs (create/delete/change password, rôle) + audit des connexions.
- MFA (TOTP) + WebAuthn (clé FIDO2) pour comptes admin.
- Révocation / liste des refresh tokens + rotation clé JWT programmée.
- Secure aggregation réelle (masques pairwise multi-parties, protocole de somme masquée) + preuve d’intégrité.
- Sandboxing plugins renforcé (AST parse, policy listes blanches, quotas CPU/mémoire/tempdir isolé).

### Observabilité & Fiabilité
- Journaux JSON structurés enrichis (corrélation request_id, trace_id, user).
- Middleware OpenTelemetry (spans API + orchestrateur) + export OTLP configurable.
- Endpoint /api/metrics/meta (liste des métriques exposées + description).
- Tests charge / soak (fédération, workflows massifs, XAI intensif).

### Persistence & Données
- Persistance utilisateurs + rôles (SQLite ou Postgres optionnel).
- Journal immuable signé (append-only) pour audit sécurité + réplication HA.
- Snapshots orchestrateur (placement, mémoire, twin) + restore.

### Orchestrateur & Performance
- Planification DAG workflows (dépendances, retries, timeouts).
- Placement cost-aware (latence réseau, énergie, coût cloud, densité VRAM).
- Secure offload chiffré (en transit + at-rest) NVMe / réseau.
- Accélération RDMA / SFP+ réelle (binding lib native plus tard).

### Marketplace & Extensions
- Notation communautaire signée + réputation plugin.
- Signature cryptographique (clé privée éditeur) + vérification publique.
- Mode sandbox isolé (process séparé + policy seccomp / firejail expérimental).

### Expérience & Tooling
- Assistant CLI interactif (bootstrap cluster, diagnostics guidés).
- Générateur de plugins (scaffold) avec tests auto.
- Workflow CI release automatique (GitHub Release + SBOM attaché + artefacts wheels/deb).

### Sécurité réseau / Zero Trust
- Attestation mutuelle nœuds (challenge HMAC + ephemeral keys).
- Politique RBAC fine (ressource + action) + configuration YAML.
- Limitation plus granulaire (quotas par rôle/endpoint, burst tokens).

### XAI & Conformité
- Multiples explainers (Integrated Gradients, SHAP, LIME fallback).
- Rapport d’audit exportable (PDF/JSON) consolidant XAI + fairness + logs.
- Détection biais basique (stat parity) sur jeux de tests fournis.

---
Ces éléments constituent la base d’une feuille de route 0.3.x / 0.4.x structurée (sécurité ++, orchestrateur avancé, haute confiance, auditabilité totale).
