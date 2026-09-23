#!/usr/bin/env bash
# Mesures de l'étage CPU/DRAM à refaire après passage de la VM en `cpu: host`.
#
# Contexte (2026-09-23) : la VM exposait un « QEMU Virtual CPU 2.5+ » sans AVX/AVX2/FMA,
# 8 vCPU — llama.cpp chargeait sa variante SSE4.2 et l'étage CPU plafonnait à ~21 Go/s
# effectifs sur un EPYC 7402 (8 canaux DDR4, ~200 Go/s). Ce script refait, à l'identique,
# les mesures bridées pour pouvoir comparer avant / après.
#
# Usage : ./benchmarks/bench_cpu_tier_host.sh      (≈ 30-45 min, GPU et CPU sollicités)
set -u
L=$(ls -d ~/.cache/vramancer/bin/*linux-vulkan*/llama-*/ | head -1); export LD_LIBRARY_PATH="$L"
OUT="$(dirname "$0")/results/cpu_tier_host_$(date +%Y%m%d_%H%M).md"
mkdir -p "$(dirname "$OUT")"
T=$(nproc)
Q6=~/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q6_K.gguf
D6=~/models/Qwen2.5-Coder-32B-GGUF/Qwen2.5-Coder-32B-Instruct-Q6_K.gguf
DS=~/models/DeepSeek-V4-Flash-IQ2_XS/IQ2_XS-XL/DeepSeek-V4-Flash-IQ2_XS-XL-00001-of-00002.gguf

{
  echo "# Étage CPU après passage en cpu: host — $(date -Iseconds)"
  echo
  echo "- CPU vu par la VM : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)"
  echo "- vCPU : $T"
  echo "- AVX2 : $(grep -m1 flags /proc/cpuinfo | grep -qw avx2 && echo oui || echo NON)"
  echo "- FMA : $(grep -m1 flags /proc/cpuinfo | grep -qw fma && echo oui || echo NON)"
  echo "- AVX-512 : $(grep -m1 flags /proc/cpuinfo | grep -qw avx512f && echo oui || echo non)"
  echo "- variante CPU chargée par llama.cpp : $("$L/llama-bench" --list-devices 2>&1 | grep -o 'libggml-cpu-[a-z0-9]*' | head -1)"
  echo
  echo "Référence AVANT (VM bridée, SSE4.2, 8 vCPU) entre crochets."
  echo
} | tee "$OUT"

run() { # run <libellé> <référence avant> <modèle> <args…>
  printf "%-52s" "$1" | tee -a "$OUT"
  "$L/llama-bench" -m "$3" -fa on -t "$T" -o md "${@:4}" 2>&1 | grep -E "^\| (qwen|deepseek)|failed|alloc" \
    | awk -F'|' '/qwen|deepseek/{printf "  %s=%s", $(NF-2), $(NF-1)} /failed|alloc/{printf "  ÉCHEC"}' | tee -a "$OUT"
  echo "   [avant : $2]" | tee -a "$OUT"
}

echo "## 1. Étage CPU seul (recalibration)" | tee -a "$OUT"
run "MoE Q6_K, CPU seul"                          "tg 7.46"       "$Q6" -ngl 0  -p 512 -n 64 -r 1
echo "## 2. Débordement par couches (3090 seule)" | tee -a "$OUT"
GGML_VK_VISIBLE_DEVICES=0 run "MoE Q6_K, 10 couches en CPU"          "tg 16.1-20.7" "$Q6" -ngl 30 -p 512 -n 128 -r 2
GGML_VK_VISIBLE_DEVICES=0 run "dense Q6_K, 8 couches en CPU"         "tg 3.72"      "$D6" -ngl 56 -p 512 -n 128 -r 2
echo "## 3. Experts seuls en RAM (3090 seule)" | tee -a "$OUT"
GGML_VK_VISIBLE_DEVICES=0 run "MoE Q6_K, experts de 10 couches en RAM" "tg 39.8"    "$Q6" -ngl 99 -ncmoe 10 -p 512 -n 128 -r 2
GGML_VK_VISIBLE_DEVICES=0 run "MoE Q6_K, experts de 20 couches en RAM" "tg 25.5"    "$Q6" -ngl 99 -ncmoe 20 -p 512 -n 128 -r 2
echo "## 4. DeepSeek-V4-Flash 81 GiB, tous les experts en RAM" | tee -a "$OUT"
GGML_VK_VISIBLE_DEVICES=0 run "3090 seule, experts en RAM"            "tg 1.04"      "$DS" -ngl 99 -ncmoe 43 -p 128 -n 32 -r 1
run "paire, experts de 36 couches en RAM"                             "tg 1.33"      "$DS" -ngl 99 -ncmoe 36 -p 128 -n 32 -r 1
echo | tee -a "$OUT"
echo "Résultats : $OUT"
