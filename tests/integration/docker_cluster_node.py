#!/usr/bin/env python3
"""Nœud de simulation cross-nœud (D3.c) — tourne DANS un conteneur Docker.

Démarre ClusterDiscovery, journalise join/leave, et écrit un état JSON toutes
les 2 s dans /out/<nom>.json pour que le script hôte puisse l'observer.
"""
import json
import os
import sys
import time

sys.path.insert(0, "/app")

from experimental.cluster_discovery import ClusterDiscovery  # noqa: E402

NAME = os.environ.get("NODE_NAME", "node")
DURATION = int(os.environ.get("DURATION", "60"))
OUT = f"/out/{NAME}.json"
events = []

# UDP_PORT différent d'un nœud à l'autre = la voie UDP broadcast ne peut PAS
# les apparier ; toute découverte restante vient forcément de mDNS. C'est ainsi
# qu'on isole les deux voies (sinon on ne sait pas laquelle a marché).
PORT = int(os.environ.get("UDP_PORT", "55555"))
disco = ClusterDiscovery(port=PORT, heartbeat_interval=2.0, node_timeout=8.0)
disco.on_join(lambda info: events.append({"t": round(time.time(), 2), "event": "join",
                                          "node": info.get("hostname")}))
disco.on_leave(lambda info: events.append({"t": round(time.time(), 2), "event": "leave",
                                           "node": info.get("hostname")}))
disco.start()
print(f"[{NAME}] discovery démarrée port={PORT}, mDNS actif = {disco._stats.get('mdns_active')}", flush=True)

t0 = time.time()
while time.time() - t0 < DURATION:
    nodes = disco.get_nodes()
    state = {
        "name": NAME,
        "t": round(time.time() - t0, 1),
        "known_nodes": sorted(n.get("hostname", "?") for n in nodes),
        "node_count": len(nodes),
        "leader": disco._leader,
        "stats": dict(disco._stats),
        "events": events,
    }
    with open(OUT, "w") as f:
        json.dump(state, f, indent=2)
    print(f"[{NAME}] t={state['t']}s nodes={state['known_nodes']} leader={state['leader']}", flush=True)
    time.sleep(2)

disco.stop()
print(f"[{NAME}] terminé", flush=True)
