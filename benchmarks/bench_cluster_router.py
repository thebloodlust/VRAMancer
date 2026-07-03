#!/usr/bin/env python3
"""Mesure ClusterRouter : data-parallel en PROCESS scale-t-il ~×2 (vs l'artefact GIL des threads) ?

Plus tôt, le data-parallel par threads donnait ×0.97 (GIL bloque la boucle de décode).
Ici, des PROCESS isolés (1 GPU/worker) → vraie parallélisme. On attend ~×2.

C'est la validation de la brique ClusterRouter (qui débloque cross-vendor + cross-nœud).

Usage: python benchmarks/bench_cluster_router.py [modele]
"""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
N_REQ = int(sys.argv[2]) if len(sys.argv) > 2 else 8
MAXTOK = 48
PROMPTS = [f"Write a short Python function example #{i} about lists." for i in range(N_REQ)]


def run(gpu_ids):
    from core.cluster_router import ClusterRouter
    r = ClusterRouter(MODEL, gpu_ids=gpu_ids)
    info = r.start()
    # warmup (1 requête par worker)
    r.submit_batch(PROMPTS[:len(gpu_ids)], max_tokens=4)
    t0 = time.perf_counter()
    res = r.submit_batch(PROMPTS, max_tokens=MAXTOK)
    dt = time.perf_counter() - t0
    r.shutdown()
    total_tok = sum(x["gen_tokens"] for x in res if x.get("ok"))
    by_gpu = {}
    for x in res:
        by_gpu[x["gpu_id"]] = by_gpu.get(x["gpu_id"], 0) + 1
    return {"workers": len(gpu_ids), "wall_s": round(dt, 2),
            "total_tokens": total_tok, "agg_tok_s": round(total_tok / dt, 1),
            "req_par_gpu": by_gpu, "all_ok": all(x.get("ok") for x in res)}


def main():
    import torch
    n = torch.cuda.device_count()
    print(f"[setup] modèle={MODEL} n_req={N_REQ} maxtok={MAXTOK} GPUs={n}\n", flush=True)
    if n < 2:
        print("Besoin de 2 GPU."); return

    one = run([0])
    print(f"[1 worker ] {one}", flush=True)
    two = run([0, 1])
    print(f"[2 workers] {two}", flush=True)

    speedup = round(two["agg_tok_s"] / one["agg_tok_s"], 2) if one["agg_tok_s"] else 0
    print("\nRESULT_JSON:" + json.dumps({"one": one, "two": two, "speedup": speedup}))
    print("\n=== VERDICT ===")
    print(f"1 worker : {one['agg_tok_s']} tok/s | 2 workers : {two['agg_tok_s']} tok/s -> ×{speedup}")
    print(f"répartition 2 workers : {two['req_par_gpu']}")
    if speedup >= 1.6:
        print("-> data-parallel MULTI-PROCESS scale (~×2). L'artefact GIL des threads est levé.")
        print("   La brique ClusterRouter est validée : même archi -> cross-vendor + cross-nœud.")
    else:
        print(f"-> scaling ×{speedup} < attendu. À investiguer (modèle trop petit ? overhead spawn ?).")


if __name__ == "__main__":
    main()
