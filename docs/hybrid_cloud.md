# Bridge cloud hybride

## Module : `core/cloud/hybrid_bridge.py`
- Permet de basculer dynamiquement entre local et cloud (AWS, Azure, GCP)
- API unifiée pour déploiement, offload, monitoring

### Exemple d’utilisation
```python
from core.cloud.hybrid_bridge import HybridCloudBridge
bridge = HybridCloudBridge("aws", credentials={"key": "..."})
bridge.deploy("model", {"instance_type": "g4dn.xlarge"})
bridge.offload({"data": "..."})
bridge.monitor()
```

---
À adapter selon le provider et les besoins (API cloud réelles à intégrer).
