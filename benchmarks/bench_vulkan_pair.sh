#!/usr/bin/env bash
# D3.d — Garde-fou AVANT tout investissement cross-vendor maison (PLAN_D_POST_LANCEMENT.md).
# Mesure ce que llama.cpp Vulkan donne DÉJÀ sur la paire RTX 3090 + RX 7900 XT.
#
# Prérequis (voir procédure de réparation du 2026-09-18) :
#   - module nvidia chargé (nvidia-smi OK)
#   - amdgpu sain (cat /sys/class/drm/renderD129/device/power/runtime_status != error)
#   - build Vulkan : ~/tools/llama-vulkan-b11026 (RADV pour AMD, driver proprio pour NVIDIA)
#
# Usage : ./benchmarks/bench_vulkan_pair.sh [modele.gguf]
# Résultats : benchmarks/results/vulkan_pair_$(date).md — chiffres BRUTS, règle honnêteté.
set -u
LLAMA=~/tools/llama-vulkan-b11026
MODEL="${1:-$HOME/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
OUT="$(dirname "$0")/results/vulkan_pair_$(date +%Y%m%d_%H%M).md"
mkdir -p "$(dirname "$OUT")"
export LD_LIBRARY_PATH="$LLAMA"

run() { # run <titre> <args llama-bench...>
  echo -e "\n## $1\n" | tee -a "$OUT"
  echo '```' >> "$OUT"
  "$LLAMA/llama-bench" -m "$MODEL" -p 512 -n 128 -fa on -o md "${@:2}" 2>&1 | tee -a "$OUT"
  echo '```' >> "$OUT"
}

echo "# Bench llama.cpp Vulkan — paire 3090 + 7900 XT ($(date -Iseconds))" | tee "$OUT"
echo -e "\nBuild: b11026 vulkan-x64 (précompilé) · Modèle: $MODEL\n" | tee -a "$OUT"

echo -e "\n## Devices détectés\n\`\`\`" | tee -a "$OUT"
"$LLAMA/llama-bench" --list-devices 2>&1 | tee -a "$OUT"
echo '```' >> "$OUT"
DEVICES=$("$LLAMA/llama-bench" --list-devices 2>/dev/null | grep -c "Vulkan")
if [ "$DEVICES" -lt 2 ]; then
  echo "ATTENTION: $DEVICES device(s) Vulkan détecté(s) au lieu de 2 — corriger avant de bencher." | tee -a "$OUT"
  [ "$DEVICES" -eq 0 ] && exit 1
fi

# 1. AMD seule (le modèle 21 GB dépasse ses 20 GB → -ngl partiel attendu, noter le max qui tient)
run "7900 XT seule (Vulkan/RADV)" -dev Vulkan1 -ngl 40
# 2. 3090 seule en Vulkan (contrôle : pénalité Vulkan vs CUDA connue de BENCHMARK_RESULTS.md)
run "3090 seule (Vulkan/proprio)" -dev Vulkan0 -ngl 99
# 3. LA mesure qui décide : paire mixte, split par défaut puis proportionnel VRAM (24:20)
run "Paire 3090+7900XT (split défaut)" -ngl 99
run "Paire 3090+7900XT (split 24:20)" -ngl 99 -ts 24/20
# 4. Split par lignes (parfois meilleur en multi-GPU Vulkan)
run "Paire 3090+7900XT (split row)" -ngl 99 -sm row

echo -e "\nRésultats bruts dans: $OUT"
echo "Rappel D3.d : si la paire mixte marche bien ici → le cross-vendor maison est un non-sujet (leçon A1)."
