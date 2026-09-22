# D3.c — simulation cross-nœud, 2 conteneurs Docker (bridge)

> Exécuté le 2026-09-22T22:10:28+02:00 · durée 60 s · image python:3.12-slim
> Mode : zeroconf installé = 1 · port UDP node-b = 55555
> (un port UDP différent sur node-b rend l'appariement UDP broadcast impossible :
> toute découverte restante vient alors de mDNS — c'est ainsi qu'on isole les 2 voies)
> **Ceci ne remplace PAS D3.b** (2 machines réelles) : un bridge Docker n'a ni la
> latence, ni le MTU, ni le comportement multicast d'un vrai LAN. C'est un
> dégrossissage de `experimental/cluster_discovery.py`, rien de plus.

## État final node-a (JSON brut)
```json
{
  "name": "node-a",
  "t": 56.0,
  "known_nodes": [
    "node-a"
  ],
  "node_count": 1,
  "leader": "node-a",
  "stats": {
    "nodes_joined": 2,
    "nodes_left": 1,
    "heartbeats_sent": 29,
    "heartbeats_failed": 0,
    "udp_errors": 0,
    "mdns_active": true
  },
  "events": [
    {
      "t": 1790107769.77,
      "event": "join",
      "node": "node-a"
    },
    {
      "t": 1790107777.35,
      "event": "join",
      "node": "node-b"
    },
    {
      "t": 1790107797.45,
      "event": "leave",
      "node": "node-b"
    }
  ]
}
```

## Logs conteneur node-a
```
2026-09-22 20:09:37,351 | INFO | vramancer.connectome | [Connectome] Nouveau lien vers node-b (172.23.0.3)
2026-09-22 20:09:37,351 | INFO | vramancer.discovery | Leader elected: node-b (gpus=0)
[node-a] t=6.0s nodes=['node-a', 'node-b'] leader=node-b
[node-a] t=8.0s nodes=['node-a', 'node-b'] leader=node-b
[node-a] t=10.0s nodes=['node-a', 'node-b'] leader=node-b
[node-a] t=12.0s nodes=['node-a', 'node-b'] leader=node-b
[node-a] t=14.0s nodes=['node-a', 'node-b'] leader=node-b
[node-a] t=16.0s nodes=['node-a', 'node-b'] leader=node-b
[node-a] t=18.0s nodes=['node-a', 'node-b'] leader=node-b
[node-a] t=20.0s nodes=['node-a', 'node-b'] leader=node-b
[node-a] t=22.0s nodes=['node-a', 'node-b'] leader=node-b
[node-a] t=24.0s nodes=['node-a', 'node-b'] leader=node-b
2026-09-22 20:09:57,445 | INFO | vramancer.discovery | Node left (confirmed dead): node-b
2026-09-22 20:09:57,445 | INFO | vramancer.discovery | Leader elected: node-a (gpus=0)
[node-a] t=26.0s nodes=['node-a'] leader=node-a
[node-a] t=28.0s nodes=['node-a'] leader=node-a
[node-a] t=30.0s nodes=['node-a'] leader=node-a
[node-a] t=32.0s nodes=['node-a'] leader=node-a
[node-a] t=34.0s nodes=['node-a'] leader=node-a
[node-a] t=36.0s nodes=['node-a'] leader=node-a
[node-a] t=38.0s nodes=['node-a'] leader=node-a
[node-a] t=40.0s nodes=['node-a'] leader=node-a
[node-a] t=42.0s nodes=['node-a'] leader=node-a
[node-a] t=44.0s nodes=['node-a'] leader=node-a
[node-a] t=46.0s nodes=['node-a'] leader=node-a
[node-a] t=48.0s nodes=['node-a'] leader=node-a
[node-a] t=50.0s nodes=['node-a'] leader=node-a
[node-a] t=52.0s nodes=['node-a'] leader=node-a
[node-a] t=54.0s nodes=['node-a'] leader=node-a
[node-a] t=56.0s nodes=['node-a'] leader=node-a
```
