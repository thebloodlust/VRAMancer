#!/usr/bin/env bash
#
# serve_qwen36.sh — Lance l'API VRAMancer avec Qwen3.6-35B-A3B (GGUF Q4_K_M)
# pré-chargé sur 2 GPUs. API OpenAI-compatible sur http://localhost:5030.
#
# Usage:
#   ./serve_qwen36.sh                  # 2 GPUs, port 5030
#   ./serve_qwen36.sh --gpus 1         # 1 GPU (tient sur la RTX 3090 24 Go)
#   ./serve_qwen36.sh --port 8000      # autre port
#   VRM_API_TOKEN=monsecret ./serve_qwen36.sh   # protéger l'API par token
#
# Endpoints (compatibles OpenAI) :
#   POST http://localhost:5030/v1/chat/completions
#   POST http://localhost:5030/v1/completions
#   GET  http://localhost:5030/health
#
set -euo pipefail

# ── Répertoire du projet ───────────────────────────────────────────────
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# ── venv ───────────────────────────────────────────────────────────────
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# ── Modèle (overridable via $QWEN36_GGUF) ──────────────────────────────
MODEL="${QWEN36_GGUF:-$HOME/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
if [[ ! -f "$MODEL" ]]; then
  echo "ERREUR : modèle GGUF introuvable : $MODEL" >&2
  echo "Définis QWEN36_GGUF=/chemin/vers/model.gguf ou télécharge le modèle." >&2
  exit 1
fi

# ── Arguments (gpus / port) ────────────────────────────────────────────
GPUS=2
PORT=5030
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

# ── Environnement local (usage perso, pas de rate limit) ───────────────
export VRM_API_PORT="$PORT"
export VRM_DISABLE_RATE_LIMIT="${VRM_DISABLE_RATE_LIMIT:-1}"
# Continuous batching = meilleur débit en usage interactif/concurrent
export VRM_CONTINUOUS_BATCHING="${VRM_CONTINUOUS_BATCHING:-1}"
# Serveur mono-process : le modèle est pré-chargé en CUDA dans ce process,
# et gunicorn forke des workers (CUDA ne survit pas au fork). Obligatoire ici.
export VRM_NO_GUNICORN="${VRM_NO_GUNICORN:-1}"

echo "════════════════════════════════════════════════════════════"
echo "  VRAMancer · Qwen3.6-35B-A3B (GGUF Q4_K_M)"
echo "  Modèle : $MODEL"
echo "  GPUs   : $GPUS   Port : $PORT"
echo "  API    : http://localhost:$PORT/v1/chat/completions"
if [[ -n "${VRM_API_TOKEN:-}" ]]; then
  echo "  Auth   : token requis (Authorization: Bearer \$VRM_API_TOKEN)"
else
  echo "  Auth   : désactivée (mode local)"
fi
echo "════════════════════════════════════════════════════════════"

exec python -m core.production_api \
  --model "$MODEL" \
  --backend llamacpp \
  --gpus "$GPUS" \
  --host 0.0.0.0 \
  --port "$PORT" \
  "${EXTRA[@]}"
