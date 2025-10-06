# Digital Twin (Prototype)

Fichier: `core/simulator/digital_twin.py`

Fonctions fournies:
- `simulate(action)` : applique une action abstraite (migration, scale, etc.) et enregistre un résultat + prédiction simple.
- `replay()` : renvoie l'historique des actions simulées.
- `predict(future_action)` : alias simple autour de simulate (à enrichir).

## Modèle d'action (exemples)
```json
{ "type": "migration", "block_id": "b12", "src": "gpu0", "dst": "gpu1" }
{ "type": "scale", "nodes": 5 }
```

## Limitations
- Pas de modèle de coût/latence réel
- Aucune notion de topologie réseau / bande passante
- Pas de corrélation avec l'état mémoire live (snapshot manuel requis)

## Roadmap proposée
1. Introduire un objet ClusterState structuré (nœuds, liens, blocs, charges).
2. Ajouter simulateurs de latence (matrice adjacency) et de bande passante.
3. Injecter un snapshot automatique depuis l'orchestrateur mémoire.
4. Calculer des scores d'impact (latence moyenne, répartition VRAM post-action).
5. Exposer `/api/twin/state` et `/api/twin/impact` (prévision).
