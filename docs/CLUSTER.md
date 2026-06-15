# Cluster VRAMancer — data-parallel multi-process (et au-delà)

`vramancer cluster serve` distribue les requêtes d'inférence sur plusieurs GPU (ou,
demain, plusieurs vendeurs / plusieurs machines) **sans réécrire le moteur** : chaque
worker garde `accelerate`/`torch` standard ; VRAMancer orchestre **au-dessus**.

## Démarrer (local, aujourd'hui)
```bash
vramancer cluster serve Qwen/Qwen2.5-14B-Instruct
#  → 1 worker par GPU (process isolé), routeur data-parallel
#  → API OpenAI : POST http://localhost:5040/v1/completions
#  → /health  · dashboard cluster : http://localhost:5041/dash
```
Options : `--gpus 0,1` · `--port 5040` · `--host 0.0.0.0`.

```bash
curl http://localhost:5040/v1/completions -H "Content-Type: application/json" \
  -d '{"prompt":"Write a CSV parser","max_tokens":128}'
# → {"choices":[{"text":...}], "vramancer":{"gpu_id":1,"seconds":2.3}}
```

## Comment ça marche
- **1 worker = 1 process = 1 GPU** (`CUDA_VISIBLE_DEVICES` par worker, posé avant torch).
  Process, pas threads : le GIL ne bride pas (threads mesurés ×0.97, process **×1.97**).
- **File de travail partagée** : les workers se *volent* les requêtes (work-stealing) →
  équilibrage automatique vers le GPU le plus rapide, zéro placement manuel.
- **Requêtes entières** routées (data-parallel) → 0 transfert d'activation entre GPU.

## Cross-nœud (plusieurs machines) — passerelle
Chaque machine (laptop, Mac, desktop) fait tourner `vramancer serve <model>` **avec son
propre backend** (CUDA, MPS…). Une passerelle route les requêtes entières vers le nœud le
moins chargé (data-parallel, 0 crossing). **Le cross-nœud contourne le problème
d'interpréteur du cross-vendor** : chaque machine a son propre torch/venv, la passerelle ne
parle qu'en HTTP.

```bash
# Sur chaque machine (nœud) :
vramancer serve Qwen/Qwen2.5-7B-Instruct --port 5040

# Sur la passerelle (n'importe quelle machine) :
vramancer cluster gateway --nodes http://laptop:5040,http://mac:5040
#   ou auto-découverte mDNS :
vramancer cluster gateway --discover
# → API agrégée sur :5050, health-aware, least-loaded
```
Un nœud Mac (MPS) + un nœud CUDA fonctionnent ensemble — **cross-backend** sans venv partagé.

## Résilience & observabilité
- **Health-check** : un worker mort (OOM, crash) est **relancé automatiquement** (`/health`
  expose `alive`/`restarts`). Les requêtes continuent sur les autres workers entre-temps.
- **Historique** : chaque requête est enregistrée → `vramancer history` (tok/s, OOM, tendances).
- **Alertes** : worker mort → webhook si `VRM_ALERT_WEBHOOK` est défini (Telegram/Discord/Slack).

## Mesuré
| | data-parallel |
|---|---|
| par threads | ×0.97 (artefact GIL) |
| par process (ClusterRouter), 32 req | **×1.97** (équilibrage 16/16) |

## La même brique, 3 usages
L'insight clé (mesuré) : **un build torch est mono-vendeur** (CUDA *xor* ROCm) → un process
ne pilote qu'un vendeur. Donc cross-vendor = multi-process = **la même architecture** que
cross-nœud. Un seul `ClusterRouter` :

1. **Local multi-process** (NVIDIA, plusieurs GPU) — ✅ disponible, ×1.97.
2. **Cross-vendor** (NVIDIA + AMD même machine) — ⬜ *amorcé* : la variable par vendeur est
   faite (`HIP_VISIBLE_DEVICES`), mais un worker AMD exige un **torch ROCm = interpréteur
   Python séparé** (venv ROCm). À finir **quand un GPU AMD est disponible** (non testable avant).
3. **Cross-nœud** (Thunderbolt/USB4) — ⬜ attend une 2e machine + le lien (~16–20 Gbps,
   le seul transport assez rapide ; cf. `docs/history/phase7-tiering-2026-06/resultat_cross_node.md`).

## Portée honnête
Le data-parallel local (×2 sur 2 GPU) est **utile mais standard** — vLLM le fait aussi.
La vraie nouveauté potentielle, c'est ce que cette brique **débloque** : faire coopérer des
GPU de **vendeurs différents** (NVIDIA+AMD) ou de **machines différentes** (Thunderbolt), ce
que les moteurs standard ne font pas. Ces deux-là restent **à prouver sur matériel** —
on ne les revendique pas comme acquis.

## Cross-nœud : le bon mode
Mesuré (`bench_cross_node_transfer.py`) : le transfert d'activation décode est minuscule
(~10 KB, <1% du calcul) ; le frein du single-requête, c'est la **sérialisation
autorégressive** (B attend A). Donc cross-nœud = **débit** (data-parallel : requête entière
par nœud), **jamais** latence single-requête. Le par-couche interleavé est **mort** (réfuté).
