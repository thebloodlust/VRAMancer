#!/usr/bin/env bash
# D3.d — garde-fou AVANT tout investissement cross-vendor maison.
# Mesure ce que llama.cpp Vulkan donne DÉJÀ sur la paire RTX 3090 + RX 7900 XT.
#
# Prérequis :
#   1. module nvidia chargé      → nvidia-smi doit répondre
#      (kernel 6.8.0-139 : sudo apt install linux-modules-nvidia-595-open-6.8.0-139-generic
#       puis sudo modprobe nvidia nvidia_uvm)
#   2. amdgpu sain               → runtime_status = active
#   3. un build llama.cpp Vulkan → détecté automatiquement (voir plus bas)
#
# Usage :  ./benchmarks/bench_vulkan_pair.sh [modele.gguf]
# Sortie :  benchmarks/results/vulkan_pair_<date>.md — chiffres BRUTS (règle d'honnêteté).
set -u

MODEL="${1:-$HOME/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
OUT="$(dirname "$0")/results/vulkan_pair_$(date +%Y%m%d_%H%M).md"
mkdir -p "$(dirname "$OUT")"

# --- build Vulkan : celui téléchargé par VRAMancer, sinon ~/tools ---
LLAMA=""
for cand in "$HOME"/.cache/vramancer/bin/*linux-vulkan*/llama-*/ "$HOME/tools/llama-vulkan-b11026"; do
  [ -x "$cand/llama-bench" ] && { LLAMA="${cand%/}"; break; }
done
[ -z "$LLAMA" ] && { echo "Aucun build llama.cpp Vulkan trouvé (cherché dans ~/.cache/vramancer/bin et ~/tools)."; exit 1; }
export LD_LIBRARY_PATH="$LLAMA${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "Build utilisé : $LLAMA"

# --- pré-vol ---
nvidia-smi -L >/dev/null 2>&1 || {
  echo "ATTENTION : nvidia-smi ne répond pas — la moitié NVIDIA de la mesure sera absente."
  echo "            (c'est la panne de module noyau documentée dans conclusion7900xt.md)"
}

DEV_LIST=$("$LLAMA/llama-bench" --list-devices 2>/dev/null)
echo "$DEV_LIST"
# Identifie les devices par NOM plutôt que par position (l'ordre change avec les pilotes).
NV_DEV=$(echo "$DEV_LIST" | grep -iE "Vulkan[0-9]+:.*(NVIDIA|RTX|GeForce)" | head -1 | grep -oE "Vulkan[0-9]+")
AMD_DEV=$(echo "$DEV_LIST" | grep -iE "Vulkan[0-9]+:.*(AMD|Radeon|RADV)" | head -1 | grep -oE "Vulkan[0-9]+")
echo "NVIDIA=${NV_DEV:-absent}  AMD=${AMD_DEV:-absent}"

run() { # run <titre> <args llama-bench...>
  echo -e "\n## $1\n" | tee -a "$OUT"
  echo '```' >> "$OUT"
  "$LLAMA/llama-bench" -m "$MODEL" -p 512 -n 128 -fa on -r 2 -o md "${@:2}" 2>&1 | tee -a "$OUT"
  echo '```' >> "$OUT"
}

{
  echo "# Bench llama.cpp Vulkan — paire 3090 + 7900 XT ($(date -Iseconds))"
  echo
  echo "Build : $(basename "$LLAMA") · Modèle : $MODEL"
  echo
  echo '## Devices détectés'
  echo '```'
  echo "$DEV_LIST"
  echo '```'
} | tee "$OUT" >/dev/null

# 1. Chaque carte seule (références)
[ -n "${AMD_DEV:-}" ] && run "7900 XT seule (Vulkan/RADV) — ngl 38, le max qui tient" -dev "$AMD_DEV" -ngl 38
[ -n "${NV_DEV:-}"  ] && run "3090 seule (Vulkan) — ngl 99" -dev "$NV_DEV" -ngl 99

# 2. LA mesure qui décide : la paire mixte
if [ -n "${AMD_DEV:-}" ] && [ -n "${NV_DEV:-}" ]; then
  run "Paire 3090+7900XT (split par défaut)" -ngl 99
  run "Paire 3090+7900XT (split 24:20, proportionnel VRAM)" -ngl 99 -ts 24/20
  run "Paire 3090+7900XT (split row)" -ngl 99 -sm row
  echo -e "\n> Lecture D3.d : si la paire mixte tient la comparaison avec la meilleure carte seule,\n> le cross-vendor maison est un non-sujet (leçon A1). Sinon, la niche existe." >> "$OUT"
else
  echo -e "\n> **Paire NON mesurée** : il manque ${NV_DEV:+}${NV_DEV:-la carte NVIDIA}${AMD_DEV:+}. D3.d reste OUVERT." >> "$OUT"
fi

echo
echo "Résultats bruts : $OUT"
