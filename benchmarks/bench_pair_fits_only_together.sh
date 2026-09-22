#!/usr/bin/env bash
# LE cas manquant du verdict D3.d : un modèle TROP GROS pour chaque carte prise
# séparément, mais qui TIENT ENTIÈREMENT dans la VRAM cumulée de la paire.
#
#   Qwen3.6-35B-A3B Q6_K — 27.2 GB, 40 couches
#   RTX 3090 = 24 GB · RX 7900 XT = 20 GB · cumulé = 44 GB
#
# C'est le seul régime où le cross-vendor peut réellement payer : chaque carte
# seule doit déborder sur le CPU, la paire non.
set -u
L=$(ls -d ~/.cache/vramancer/bin/*linux-vulkan*/llama-*/ | head -1)
export LD_LIBRARY_PATH="$L"
M="${1:-$HOME/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K.gguf}"
B="$L/llama-bench"
OUT="$(dirname "$0")/results/pair_fits_only_together_$(date +%Y%m%d_%H%M).md"
mkdir -p "$(dirname "$OUT")"

DEV=$("$B" --list-devices 2>/dev/null)
NV=$(echo "$DEV" | grep -iE "Vulkan[0-9]+:.*(NVIDIA|RTX)" | head -1 | grep -oE "Vulkan[0-9]+")
AMD=$(echo "$DEV" | grep -iE "Vulkan[0-9]+:.*(AMD|Radeon|RADV)" | head -1 | grep -oE "Vulkan[0-9]+")

{
  echo "# Modèle qui ne tient QUE sur la paire — $(date -Iseconds)"
  echo
  echo "Modèle : $M"
  echo "Devices : NVIDIA=$NV AMD=$AMD"
  echo '```'
  echo "$DEV"
  echo '```'
} | tee "$OUT" >/dev/null

run() { echo -e "\n## $1\n" | tee -a "$OUT"; echo '```' >> "$OUT"
        "$B" -m "$M" -p 512 -n 128 -fa on -r 2 -o md "${@:2}" 2>&1 | tee -a "$OUT"; echo '```' >> "$OUT"; }

run "CPU seul (référence)"                    -ngl 0
run "3090 seule — max de couches qui tient"   -dev "$NV"  -ngl 30
run "7900 XT seule — max de couches qui tient" -dev "$AMD" -ngl 26
run "PAIRE — modèle ENTIER sur les deux cartes" -ngl 99
run "PAIRE — split 24:20"                     -ngl 99 -ts 24/20

echo "Résultats : $OUT"
