# Ajouter un nœud au réseau VRAMancer (mDNS auto-discovery)

VRAMancer découvre automatiquement les autres machines VRAMancer du **même réseau
local**, sans configuration d'IP — via **mDNS / ZeroConf** (le même mécanisme que les
imprimantes ou Chromecast), avec repli **UDP broadcast** si zeroconf n'est pas installé.

## Prérequis
```bash
pip install 'vramancer[cluster]'      # apporte zeroconf (mDNS)
```
(Inclus dans `vramancer[full]`. Sans zeroconf, le repli UDP broadcast fonctionne quand même.)

## Ajouter un nœud — c'est automatique
**Démarrer un serveur annonce le nœud sur le réseau** (mDNS), par défaut :
```bash
vramancer serve Qwen/Qwen2.5-14B-Instruct
#   Cluster: ce noeud est annonce via mDNS (port 5030).
```
Pour ne PAS annoncer : `vramancer serve ... --no-cluster`.

## Voir les nœuds du réseau
Depuis n'importe quelle autre machine du LAN :
```bash
vramancer discover --timeout 5
#   [machine-A] {hostname, ip, gpus: [...], gpu_count, ...}
#   [machine-B] {...}
```

## Ce qui est échangé
Chaque nœud annonce : hostname, IP, OS/plateforme, **liste des GPU + VRAM**, port API,
rôle (edge/serveur), heartbeat. Le `connectome` câble automatiquement les nœuds
découverts comme synapses (join/leave gérés).

## Thunderbolt / USB4 (lien rapide)

Quand la découverte tourne (`serve`/`discover`), VRAMancer surveille aussi les
connexions **Thunderbolt/USB4** (`USB4HotPlug`, pyudev temps-réel sur Linux ou polling
`/sys/bus/thunderbolt`, IOKit sur macOS). Brancher un câble TB/USB4 **ré-annonce** le
nœud sur la nouvelle interface — le pair est alors découvert via mDNS/IP-over-Thunderbolt.

Pourquoi ça compte : Thunderbolt/USB4 (~16–20 Gbps) est **le seul transport assez rapide**
pour du travail cross-nœud sérieux (mesuré : ~2.5 GB/s, proche du local ; le 1 GbE est
100× plus lent). Voir `docs/history/phase7-tiering-2026-06/resultat_cross_node.md`.

**État honnête** : la détection est embarquée, **câblée** (démarre avec la découverte) et
tourne sans erreur. La détection d'un **vrai périphérique TB** n'est PAS testée sur le
matériel de dev (la VM n'a pas de contrôleur Thunderbolt) — à valider sur 2 machines
reliées en TB. `pip install 'vramancer[cluster]'` apporte pyudev (temps-réel Linux).

## Vérifié
- `experimental/cluster_discovery.py` : mDNS (`_vramancer._tcp.local.`) prioritaire,
  repli UDP broadcast, heartbeat, registre thread-safe, multi-OS (Linux/macOS/Windows).
- Test : un browser zeroconf tiers voit bien le service annoncé ; `vramancer discover`
  liste le nœud local avec ses 2 GPU. (`benchmarks/` — test mDNS fonctionnel.)
