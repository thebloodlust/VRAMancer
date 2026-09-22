# D3.c — cross-nœud simulé : 2 conteneurs Docker sur un réseau bridgé

> Session du 2026-09-22. Reproductible : `./tests/integration/docker_cluster_sim.sh [durée]`
> (sortie brute regénérée dans `D3C_DOCKER_CLUSTER_SIM_RAW.md`).
>
> **Ceci ne remplace PAS D3.b.** Un bridge Docker n'a ni la latence, ni le MTU, ni le
> comportement multicast d'un vrai LAN, et les deux « nœuds » partagent un seul noyau.
> Ce qui marche ici peut échouer entre deux machines ; ce qui échoue ici échouera
> probablement aussi là-bas. C'est un dégrossissage de `experimental/cluster_discovery.py`,
> pas une validation cross-nœud.

## Protocole

Deux conteneurs `python:3.12-slim` (repo monté en lecture seule, aucune dépendance
VRAMancer installée — tous les imports `core.*` de `cluster_discovery.py` sont sous
`try/except` et retombent sur leurs stubs), réseau bridge dédié, `ClusterDiscovery`
avec `heartbeat_interval=2 s` et `node_timeout=8 s` (au lieu de 10/30 s, pour observer
en une minute). node-a démarre seul, node-b le rejoint 6 s plus tard, puis node-b est
tué brutalement (`docker rm -f`, pas d'arrêt propre) et on observe 45 s.

## Ce qui marche

| Point | Résultat mesuré |
|---|---|
| Découverte de node-b par node-a | **oui**, ~1 s après le démarrage de node-b (join à t=6 s) |
| Découverte réciproque | oui, node-b voit node-a dans le même intervalle |
| Heartbeats | 29 envoyés sur 56 s, **0 échec, 0 erreur UDP** |
| Élection de leader | node-b élu (départage par nombre de GPU puis hostname) |
| Mort brutale d'un nœud | **détectée** : « Node left (confirmed dead): node-b » |
| Bascule de leader après la mort | immédiate à l'éviction : node-a réélu dans la même seconde |
| Journal de membership | événements join/leave écrits |

**UDP broadcast traverse le bridge Docker** — c'était le risque principal identifié dans
le plan D. Il ne s'est pas matérialisé.

**mDNS aussi.** Vérifié en isolant les deux voies : en donnant à node-b un port UDP
différent (55556 vs 55555), l'appariement par broadcast devient impossible, et node-b
est **quand même découvert** → la découverte est bien passée par mDNS/zeroconf. À noter :
`zeroconf` n'est pas dans l'image de base ; sans lui, `mdns_active=false` et le code
retombe **silencieusement** sur UDP broadcast. En lisant seulement les logs on croirait
que mDNS fonctionne alors qu'il n'a jamais démarré — c'est le piège à connaître.

## Ce qu'il faut savoir avant de compter dessus

1. **La détection de mort n'est pas immédiate, et ce n'est pas un bug.** Le
   `_cleanup_loop` exige `age > node_timeout`, puis 3 cycles manqués (un cycle =
   `node_timeout/3`), puis un probe TCP en échec. Mesuré ici : **~10 s** après le kill
   avec `node_timeout=8 s` ; avec les valeurs par défaut (30 s) il faut compter
   **plusieurs dizaines de secondes**. Une première observation à 14 s avait conclu à
   tort à un échec — la fenêtre était trop courte.
2. **Un nœud mort reste leader pendant toute cette fenêtre.** Entre le kill et
   l'éviction, node-a continue de désigner node-b comme leader. Si du travail est routé
   vers le leader, il part dans le vide pendant ~10 s (ou ~1 min en configuration par
   défaut). À traiter avant tout sharding cross-nœud.
3. Les deux conteneurs partagent le noyau de l'hôte : rien ici ne teste une vraie pile
   réseau distincte, un MTU différent, ni la perte de paquets.

## Reste à faire sur vraie 2e machine (D3.b)

- mDNS à travers un vrai routeur/switch (le multicast y est souvent filtré, contrairement
  au bridge Docker).
- Bande passante et latence inter-nœud réelles (LAN vs Thunderbolt).
- Comportement en perte de paquets et en partition réseau (deux leaders ?).
