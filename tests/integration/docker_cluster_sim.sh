#!/usr/bin/env bash
# D3.c — simulation cross-nœud avec 2 conteneurs Docker sur un réseau bridgé.
#
# BUT : dégrossir cluster_discovery.py AVANT l'arrivée d'une 2e machine physique.
# Ceci NE REMPLACE PAS D3.b (2 vraies machines) : un bridge Docker n'a ni la
# latence, ni le MTU, ni le comportement multicast d'un LAN réel.
#
# Usage :  ./tests/integration/docker_cluster_sim.sh [durée_secondes]
# Sortie :  docs/sessions/D3C_DOCKER_CLUSTER_SIM.md (chiffres bruts)
set -u
DUR="${1:-40}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NET=vrm-sim-net
IMG=python:3.12-slim
OUTDIR=$(mktemp -d)
trap 'docker rm -f vrm-node-a vrm-node-b >/dev/null 2>&1; docker network rm $NET >/dev/null 2>&1; rm -rf "$OUTDIR"' EXIT

command -v docker >/dev/null || { echo "docker absent"; exit 1; }
docker network rm $NET >/dev/null 2>&1
docker network create --driver bridge $NET >/dev/null || exit 1
echo "réseau $NET créé ($(docker network inspect $NET -f '{{(index .IPAM.Config 0).Subnet}}'))"

# ZEROCONF=1 (défaut) installe zeroconf dans le conteneur pour tester la voie mDNS ;
# ZEROCONF=0 saute l'install (hors-ligne) et ne teste que le fallback UDP broadcast.
ZEROCONF="${ZEROCONF:-1}"
run_node() { # run_node <nom> <conteneur>
  local cmd="python /app/tests/integration/docker_cluster_node.py"
  [ "$ZEROCONF" = "1" ] && cmd="pip install -q zeroconf 2>/dev/null; $cmd"
  docker run -d --rm --name "$2" --network $NET --hostname "$1" \
    -e NODE_NAME="$1" -e DURATION="$DUR" -e VRM_MINIMAL_TEST=1 -e UDP_PORT="${3:-55555}" \
    -e VRM_MEMBERSHIP_LOG=/tmp/membership.jsonl \
    -v "$ROOT":/app:ro -v "$OUTDIR":/out \
    $IMG sh -c "$cmd" >/dev/null
}

echo "démarrage node-a…"; run_node node-a vrm-node-a
sleep 6
echo "démarrage node-b (doit être découvert par node-a)…"; run_node node-b vrm-node-b "${NODE_B_UDP_PORT:-55555}"
sleep 10

echo "--- état à mi-parcours ---"
for f in "$OUTDIR"/*.json; do echo "== $(basename "$f")"; cat "$f"; done

# La détection de mort n'est PAS immédiate : le cleanup_loop tourne toutes les
# node_timeout/3 s (≈2.7 s ici) et exige age > node_timeout PUIS 3 cycles manqués
# PUIS un probe TCP en échec, soit ≈ 8 + 3×2.7 + probe ≈ 20 s minimum. On observe
# donc 45 s après le kill, sinon on conclurait à tort à un bug.
echo "kill node-b (test leave / timeout, fenêtre d'observation 45 s)…"
docker rm -f vrm-node-b >/dev/null 2>&1
sleep 45

echo "--- état final node-a ---"
FINAL_A=$(cat "$OUTDIR/node-a.json" 2>/dev/null)
echo "$FINAL_A"

# ---- rapport ----
REPORT="$ROOT/docs/sessions/D3C_DOCKER_CLUSTER_SIM_RAW.md"
{
  echo "# D3.c — simulation cross-nœud, 2 conteneurs Docker (bridge)"
  echo
  echo "> Exécuté le $(date -Iseconds) · durée $DUR s · image $IMG"
  echo "> Mode : zeroconf installé = $ZEROCONF · port UDP node-b = ${NODE_B_UDP_PORT:-55555}"
  echo "> (un port UDP différent sur node-b rend l'appariement UDP broadcast impossible :"
  echo "> toute découverte restante vient alors de mDNS — c'est ainsi qu'on isole les 2 voies)"
  echo "> **Ceci ne remplace PAS D3.b** (2 machines réelles) : un bridge Docker n'a ni la"
  echo "> latence, ni le MTU, ni le comportement multicast d'un vrai LAN. C'est un"
  echo "> dégrossissage de \`experimental/cluster_discovery.py\`, rien de plus."
  echo
  echo '## État final node-a (JSON brut)'
  echo '```json'
  echo "$FINAL_A"
  echo '```'
  echo
  echo '## Logs conteneur node-a'
  echo '```'
  docker logs vrm-node-a 2>&1 | tail -30
  echo '```'
} > "$REPORT"
echo "rapport écrit : $REPORT"
