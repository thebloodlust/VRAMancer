#!/usr/bin/env python3
"""Mesure cross-nœud LIVE — débit agrégé via la passerelle (data-parallel réel).

À lancer une fois que les nœuds tournent (`vramancer serve`) et la passerelle aussi
(`vramancer cluster gateway --discover`). Envoie N requêtes concurrentes, mesure le
débit agrégé + la répartition par nœud + les latences.

Pour voir le SCALING : lance la passerelle avec 1 nœud, note le débit ; relance-la avec
2 nœuds, compare. (Ou pointe --url directement sur un nœud unique pour la baseline.)

Usage: python benchmarks/bench_cross_node_live.py [gateway_url] [n_req] [max_tokens]
  ex:  python benchmarks/bench_cross_node_live.py http://localhost:5050 16 64
"""
import sys, json, time, threading, urllib.request, statistics

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5050"
N_REQ = int(sys.argv[2]) if len(sys.argv) > 2 else 16
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 64
PROMPT = "Write a concise Python function that deduplicates a list while keeping order."


def one(i, out):
    body = json.dumps({"prompt": PROMPT, "max_tokens": MAXTOK}).encode()
    req = urllib.request.Request(URL + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.loads(r.read().decode())
        dt = time.perf_counter() - t0
        tokens = d.get("usage", {}).get("completion_tokens", 0)
        node = d.get("vramancer", {}).get("node", "?")
        out[i] = {"ok": "choices" in d, "dt": dt, "tokens": tokens, "node": node}
    except Exception as e:
        out[i] = {"ok": False, "error": str(e), "dt": time.perf_counter() - t0}


def main():
    # pré-vol
    try:
        with urllib.request.urlopen(URL + "/health", timeout=5) as r:
            h = json.loads(r.read().decode())
        print(f"[health] {URL} → {json.dumps(h)[:200]}", flush=True)
    except Exception as e:
        print(f"[health] {URL} injoignable: {e}"); return 1

    print(f"[bench] {N_REQ} requêtes concurrentes, max_tokens={MAXTOK}…", flush=True)
    out = [None] * N_REQ
    threads = [threading.Thread(target=one, args=(i, out)) for i in range(N_REQ)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.perf_counter() - t0

    ok = [o for o in out if o and o.get("ok")]
    total_tokens = sum(o["tokens"] for o in ok)
    lats = sorted(o["dt"] for o in ok)
    by_node = {}
    for o in ok:
        by_node[o["node"]] = by_node.get(o["node"], 0) + 1

    print(f"\n=== RÉSULTAT ({len(ok)}/{N_REQ} OK) ===")
    print(f"débit agrégé   : {total_tokens/wall:.1f} tok/s  ({total_tokens} tok en {wall:.2f}s)")
    if lats:
        p50 = lats[len(lats)//2]; p95 = lats[min(len(lats)-1, int(len(lats)*0.95))]
        print(f"latence/req    : p50={p50:.2f}s  p95={p95:.2f}s")
    print(f"répartition    : {by_node}  ({len(by_node)} nœud(s))")
    print("\n-> Compare ce débit avec 1 seul nœud pour voir le scaling (idéal ~×nb_noeuds).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
