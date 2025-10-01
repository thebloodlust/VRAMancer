# API d’automatisation avancée (REST & GraphQL)

## Lancer l’API
```bash
python3 core/api/automation_api.py
```

## Endpoints REST
- **GET /api/jobs** : liste tous les jobs
- **POST /api/jobs** : crée un job
  - Body JSON : `{ "name": "Mon job" }`
- **GET /api/jobs/<id>** : récupère un job
- **DELETE /api/jobs/<id>** : supprime un job

### Exemple curl
```bash
curl -X POST http://localhost:5002/api/jobs -H "Content-Type: application/json" -d '{"name": "test job"}'
curl http://localhost:5002/api/jobs
```

## Endpoint GraphQL
- **POST /graphql**
  - Body JSON : `{ "query": "{ jobs { id name status } }" }`

### Exemple curl
```bash
curl -X POST http://localhost:5002/graphql -H "Content-Type: application/json" -d '{"query": "{ jobs { id name status } }"}'
```

## Intégration DevOps
- Déploiement automatisé de jobs/tâches IA
- Intégration possible dans vos pipelines CI/CD (GitLab, GitHub Actions, Jenkins…)
- Monitoring et gestion centralisée des tâches

---

**À adapter selon vos besoins (auth, sécurité, orchestration, etc.)**
