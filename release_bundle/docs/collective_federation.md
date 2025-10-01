# Intelligence collective & fédération

## Module : `core/collective/federation.py`
- Partage de modèles, datasets, résultats entre clusters (fédération, P2P)
- Publication, synchronisation, découverte de pairs

### Exemple d’utilisation
```python
from core.collective.federation import FederationNode
node = FederationNode("node1", "10.0.0.1")
node.publish_model({"name": "bert", "version": "1.0"})
node.sync_dataset({"name": "images", "size": 1000})
peers = node.discover_peers()
```

---
À compléter pour la fédération réelle (P2P, discovery, sécurité).
