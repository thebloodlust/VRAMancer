# Edge / IoT & Supervision

## Edge / IoT
Le module `edge_iot.py` permet d’intégrer des nœuds périphériques (Raspberry Pi, Jetson, ARM, etc.) dans le cluster VRAMancer. Il détecte automatiquement les capacités du matériel, s’adapte à l’architecture, et lance des tâches IA ou monitoring en local.

- Détection automatique : CPU, RAM, OS, type (edge/standard)
- Orchestration locale : exécution de tâches IA, monitoring, offload
- Usage typique : inference, monitoring, worker réseau

## Supervision
Le module `supervision.py` gère l’enregistrement, le suivi et la supervision des nœuds (edge ou standards) du cluster.

- Enregistrement dynamique des nœuds
- Suivi du statut (actif/inactif)
- Monitoring en temps réel (thread, logs)
- Extensible pour alertes, auto-réparation, reporting

### Exemple d’utilisation
```python
from core.network.edge_iot import EdgeNode
from core.network.supervision import NodeSupervisor

sup = NodeSupervisor()
n1 = EdgeNode(host="raspberrypi")
n2 = EdgeNode(host="jetson")
sup.register_node(n1)
sup.register_node(n2)
# Lancer la supervision en arrière-plan
import threading
threading.Thread(target=sup.monitor, daemon=True).start()
n1.run_task("inference")
n2.run_task("monitoring")
```

---

**Edge/IoT** = orchestration IA en périphérie, faible consommation, déploiement massif.
**Supervision** = monitoring, reporting, auto-réparation, haute disponibilité.
