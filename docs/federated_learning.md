# Federated Learning (Prototype)

Fichier: `core/collective/federated_learning.py`

## Composant actuel
`FederatedLearner.aggregate(updates)` : moyenne simple des valeurs reçues.

## Limitations
- Pas de pondération par taille de dataset
- Pas de rounds coordonnés multi-peers réels
- Pas de chiffrement / secure aggregation
- Pas de compression gradients

## Intégration API
API unifiée (`core/api/unified_api.py`) expose :
- POST `/api/federated/round/start` : init round
- POST `/api/federated/round/submit` : ajouter une update (champ `value`)
- GET `/api/federated/round/aggregate` : calcule la moyenne

## Roadmap
1. Ajout pondération updates (`weight` dans payload)
2. Stratégies (FedAvg, FedProx) via interface `Strategy`
3. Secure aggregation (masquage additif, cryptographie) optionnelle
4. Export métriques Prometheus (latence round, taille updates)
5. Gestion échecs & timeouts (exclusion clients lents)
