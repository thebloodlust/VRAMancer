#!/usr/bin/env python3
"""Mesure réelle de la bande passante de transfert inter-GPU
RTX 3090 (cuda:0, Ampere) <-> RTX 5070 Ti (cuda:1, Blackwell).

Contexte : `nvidia-smi topo` indique P2P = NS (Not Supported) entre ces deux
GPU consumer — les transferts passent donc par la RAM hôte (CPU-staged). Cette
mesure établit la baseline pour DeepSeek P2.10 (auto-tuning de la taille de
chunk) et compare :
  1. torch naïf  : d.copy_(t) cross-device (staging géré par CUDA runtime)
  2. VRAMancer   : GpuPipeline (triple-buffered pinned) via bench_gpu_transfer,
                   en balayant chunk_mb pour trouver l'optimum.

Tourne en cohabitation avec le serveur (petits tenseurs, VRAM libre vérifiée).
"""
import json
import time
from pathlib import Path

import torch

SIZES_MB = [4, 16, 64, 256]
CHUNKS_MB = [4, 8, 16, 32, 64]
ITERS = 30
WARMUP = 10
OUT = Path("benchmarks/results/deepseek_p2.10_pcie_bw")


def bench_torch(src: int, dst: int, size_mb: int) -> float:
    """GB/s effectif pour copy_ cross-device pré-alloué (transfert pur)."""
    n = size_mb * 1024 * 1024 // 4
    t = torch.empty(n, dtype=torch.float32, device=f"cuda:{src}")
    d = torch.empty(n, dtype=torch.float32, device=f"cuda:{dst}")
    for _ in range(WARMUP):
        d.copy_(t)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        d.copy_(t)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS
    del t, d
    torch.cuda.empty_cache()
    return (size_mb / 1024) / dt  # GB/s


def bench_vramancer(src: int, dst: int, size_mb: int, chunk_mb: int):
    import vramancer_rust as v
    res = v.bench_gpu_transfer(src, dst, size_mb, chunk_mb, WARMUP, ITERS)
    # res est un dict[str,str] ; on récupère un champ de bande passante
    return res


def main():
    print(f"torch {torch.__version__}, devices: "
          f"{torch.cuda.get_device_name(0)} / {torch.cuda.get_device_name(1)}")

    results = {"torch_baseline": {}, "vramancer_pipeline": {}, "raw_vramancer": {}}

    # 1. Baseline torch naïf, les deux directions
    print("\n== torch naïf (copy_ cross-device) ==")
    for size in SIZES_MB:
        bw01 = bench_torch(0, 1, size)
        bw10 = bench_torch(1, 0, size)
        results["torch_baseline"][f"{size}MB"] = {
            "0->1_gbps": round(bw01, 3), "1->0_gbps": round(bw10, 3)
        }
        print(f"  {size:4d} MB : 0->1 {bw01:6.3f} GB/s | 1->0 {bw10:6.3f} GB/s")

    # 2. VRAMancer GpuPipeline, sweep chunk_mb
    print("\n== VRAMancer GpuPipeline (bench_gpu_transfer), sweep chunk ==")
    try:
        import vramancer_rust as v
        for size in SIZES_MB:
            best_bw, best_chunk, per_chunk = 0.0, None, {}
            for chunk in CHUNKS_MB:
                if chunk > size:
                    continue
                try:
                    res = v.bench_gpu_transfer(0, 1, size, chunk, WARMUP, ITERS)
                    results["raw_vramancer"][f"{size}MB_chunk{chunk}"] = res
                    # Lire les giga-OCTETS/s. Clé explicite bandwidth_gbyte_s
                    # (fallback bandwidth_gbs, legacy). NE PAS lire bandwidth_gbps
                    # /_gbit_s qui sont en giga-BITS/s (x8, trompeur).
                    bw = float(res.get("bandwidth_gbyte_s")
                               or res.get("bandwidth_gbs", 0.0))
                    method = res.get("method", "?")
                    per_chunk[f"chunk{chunk}"] = bw
                    if bw > best_bw:
                        best_bw, best_chunk = bw, chunk
                    print(f"  {size:4d} MB chunk {chunk:2d} MB -> "
                          f"{bw:6.3f} GB/s ({method}, {res.get('avg_ms')} ms)")
                except Exception as e:
                    print(f"  {size} MB chunk {chunk}: erreur {e}")
            results["vramancer_pipeline"][f"{size}MB"] = {
                "best_chunk_mb": best_chunk, "best_gbps": round(best_bw, 3),
                "per_chunk": per_chunk,
            }
    except Exception as e:
        print(f"  GpuPipeline indisponible: {e}")
        results["vramancer_pipeline"]["error"] = str(e)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{OUT}.json").write_text(json.dumps(results, indent=2))
    print(f"\nÉcrit {OUT}.json")
    return results


if __name__ == "__main__":
    main()
