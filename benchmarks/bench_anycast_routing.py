#!/usr/bin/env python3
"""Anycast strategy comparison benchmark (simulated peers)."""
import argparse
import json
import random
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--peers", type=int, default=5)
    p.add_argument("--rounds", type=int, default=1000)
    p.add_argument("--out", default="bench_anycast_routing.json")
    args = p.parse_args()

    try:
        from core.network.anycast_balancer import AnycastBalancer
    except ImportError as exc:
        print(f"AnycastBalancer unavailable: {exc}")
        return

    rng = random.Random(42)
    peers = [
        {"node_id": f"peer-{i}",
         "latency_ms": rng.uniform(1, 50),
         "strength": rng.uniform(0.1, 1.0)}
        for i in range(args.peers)
    ]

    results = {}
    for strategy in ("weighted", "least_latency", "round_robin"):
        balancer = AnycastBalancer(strategy=strategy)
        for pr in peers:
            try:
                balancer.update_node_health(
                    pr["node_id"],
                    latency_ms=pr["latency_ms"],
                    strength=pr["strength"],
                )
            except Exception:
                pass
        chosen = []
        t0 = time.perf_counter()
        for _ in range(args.rounds):
            try:
                sel = balancer.select_peer()
            except Exception:
                sel = None
            if sel:
                chosen.append(sel)
        elapsed = time.perf_counter() - t0
        results[strategy] = {
            "success_rate": len(chosen) / args.rounds,
            "wall_seconds": round(elapsed, 3),
            "selections_per_sec": round(args.rounds / elapsed, 1) if elapsed else 0,
        }

    with open(args.out, "w") as f:
        json.dump({"peers": peers, "results": results}, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
