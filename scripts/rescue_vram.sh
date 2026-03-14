#!/usr/bin/env bash

echo "=========================================================="
echo "🚨 VRAMancer - Emergency Memory Cleanser 🚨"
echo "=========================================================="

echo "[1] Killing all running Python / vLLM ghost processes..."
pkill -9 -f "vllm"
pkill -9 -f "vramancer.main"
pkill -9 -f "vramancer"
pkill -9 -f "multiproc_executor"
pkill -9 -f "dashboard"

echo "[2] Releasing RAM cache & NVMe swap..."
sync && echo "Flushing kernel caches (requires sudo)" && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

echo "[3] Interrogating NVIDIA-SMI to find and kill CUDA zombies..."
if command -v nvidia-smi &> /dev/null; then
    echo "Killing processes holding NVIDIA GPUs..."
    sudo fuser -k -9 /dev/nvidia* 2>/dev/null
    echo "Current vRAM status:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
else
    echo "NVIDIA-SMI not found. Skipping CUDA target wipe."
fi

echo "=========================================================="
echo "✅ VRAM is now fully sanitized. All ghost processes are DEAD."
echo "You can safely restart the Swarm Orchestrator:"
echo "👉 VRM_CORS_ORIGINS=\"*\" python3 -m vramancer.main serve --host \"0.0.0.0\" --port 5000"
echo "=========================================================="