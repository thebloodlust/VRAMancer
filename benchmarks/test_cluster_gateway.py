#!/usr/bin/env python3
"""Test passerelle cross-nœud : route data-parallel vers des nœuds distants (simulés).

test_nodepool_least_loaded : unitaire, sans réseau (collectable pytest).
main() : intégration — 2 faux nœuds HTTP + gateway, vérifie la répartition.
"""
import os, sys, json, threading, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.cluster_gateway import NodePool


def test_nodepool_least_loaded():
    pool = NodePool(["http://a", "http://b"])
    n1 = pool.pick(); assert n1["inflight"] == 1        # a (égalité -> premier)
    n2 = pool.pick(); assert n2["url"] == "http://b"    # b (moins chargé)
    n3 = pool.pick()                                     # égalité 1/1 -> a
    assert n3["url"] == "http://a"
    pool.done(n1, True); pool.done(n2, True); pool.done(n3, True)
    snap = {n["url"]: n for n in pool.snapshot()}
    assert snap["http://a"]["served"] == 2 and snap["http://b"]["served"] == 1


def _make_node(name, port):
    from flask import Flask, jsonify
    from werkzeug.serving import make_server
    app = Flask(name)

    @app.route("/health")
    def h():
        return jsonify({"ok": True})

    @app.route("/v1/completions", methods=["POST"])
    def c():
        time.sleep(0.25)  # simule du calcul
        return jsonify({"choices": [{"text": f"hello from {name}"}], "vramancer": {"gpu_id": 0}})

    srv = make_server("127.0.0.1", port, app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    from core.cluster_gateway import cluster_gateway
    n1 = _make_node("node-laptop", 5061)
    n2 = _make_node("node-mac", 5062)
    threading.Thread(target=lambda: cluster_gateway(
        nodes=["http://127.0.0.1:5061", "http://127.0.0.1:5062"], port=5063), daemon=True).start()
    time.sleep(2)  # gateway + health loop

    results = []

    def fire(i):
        try:
            req = urllib.request.Request("http://127.0.0.1:5063/v1/completions",
                data=json.dumps({"prompt": f"hi {i}", "max_tokens": 8}).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=10) as r:
                results.append(json.loads(r.read().decode()))
        except Exception as e:
            results.append({"error": str(e)})

    threads = [threading.Thread(target=fire, args=(i,)) for i in range(6)]
    t0 = time.perf_counter()
    for t in threads: t.start()
    for t in threads: t.join()
    dt = time.perf_counter() - t0

    nodes_used = {}
    for r in results:
        nd = r.get("vramancer", {}).get("node", r.get("error", "?"))
        nodes_used[nd] = nodes_used.get(nd, 0) + 1
    print(f"6 requêtes en {dt:.2f}s — répartition: {nodes_used}")
    ok = (len([r for r in results if "choices" in r]) == 6 and len(nodes_used) == 2)
    print(f"VERDICT gateway cross-nœud (2 nœuds servis, parallèle): {'OUI' if ok else 'NON'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
