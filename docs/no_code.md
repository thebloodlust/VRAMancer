# API No-Code (Prototype)

Expose des endpoints pour créer et exécuter des workflows simples.
Fichiers: `core/api/no_code_workflow.py`, `core/api/no_code_api.py`.

## Concepts
Workflow: liste ordonnée de tâches JSON `{type: str, params: {...}}`.
Statuts: created -> running -> finished.

## Endpoints principaux
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | /api/workflows/templates | Liste templates de tâches |
| POST | /api/workflows | Création workflow `{tasks: [...]}` |
| GET | /api/workflows/<id> | Récupérer definition & statut |
| POST | /api/workflows/<id>/run | Lancer exécution asynchrone |
| GET | /api/workflows/<id>/logs | Logs d'exécution |
| GET | /api/workflows/<id>/export | Export minimal |

## Limitations actuelles
- Pas de validation schéma stricte
- Pas de dépendances entre tâches (DAG linéaire uniquement)
- Pas de retries ni rollback
- Pas de persistance disque

## Améliorations suggérées
- Pydantic pour validation
- Graph DAG + ordonnancement topologique
- Ajout estimateur runtime pour scheduler opportuniste
- Stockage persistant (sqlite ou plugin S3)
- Accès via API unifiée pour centralisation
