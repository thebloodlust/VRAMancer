# Roadmap Immédiate (Mémoire & Cluster)

| Étape | Description | Statut |
|-------|-------------|--------|
| 1 | UI mémoire dynamique (Web/Qt) | ✅ Done |
| 2 | Sécurisation API mémoire (token) | ✅ Done (`export VRM_API_TOKEN=secret`) |
| 3 | Fastpath natif RDMA / USB4 (zero-copy) | 🔄 À faire |
| 4 | Promotion guidée par accès réels (hooks backend infer) | 🔄 À faire |
| 5 | Backends vLLM/Ollama final + tests promotion/demotion | 🔄 À faire |
| 6 | Tests unitaires benchmark & policies | 🔄 À faire |
| 7 | Supervision cluster icônes nœuds temps réel (all dashboards) | 🔄 À faire |

## Utilisation de l'API mémoire sécurisée

```bash
export VRM_API_TOKEN=secret
curl -H "X-API-TOKEN: secret" http://localhost:5000/api/memory
```

Promotion/Demotion manuelle :
```bash
curl -H "X-API-TOKEN: secret" "http://localhost:5000/api/memory/promote?id=<short_id>"
curl -H "X-API-TOKEN: secret" "http://localhost:5000/api/memory/demote?id=<short_id>"
```

