"""Cross-nœud : passerelle HTTP data-parallel vers des `vramancer serve` distants.

Le cross-nœud **contourne** le problème d'interpréteur du cross-vendor : chaque machine
a son propre torch/venv (CUDA, MPS, ROCm…). La passerelle ne parle qu'en HTTP — elle se
fiche du backend de chaque nœud. Elle route des **requêtes entières** (data-parallel,
0 crossing d'activation) vers le nœud le moins chargé.

    vramancer cluster gateway --nodes http://laptop:5040,http://mac:5040
    vramancer cluster gateway --discover        # auto via mDNS

Chaque nœud fait juste tourner `vramancer serve <model>`.
"""
from __future__ import annotations
import json
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional


class NodePool:
    """Liste de nœuds + routage least-loaded (in-flight par nœud)."""

    def __init__(self, urls: List[str]):
        self.nodes = [{"url": u.rstrip("/"), "inflight": 0, "ok": True,
                       "served": 0, "errors": 0} for u in urls]
        self._lock = threading.Lock()

    def pick(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            healthy = [n for n in self.nodes if n["ok"]]
            if not healthy:
                healthy = self.nodes  # tente quand même
            if not healthy:
                return None
            n = min(healthy, key=lambda x: x["inflight"])
            n["inflight"] += 1
            return n

    def done(self, n: Dict[str, Any], ok: bool):
        with self._lock:
            n["inflight"] = max(0, n["inflight"] - 1)
            if ok:
                n["served"] += 1
            else:
                n["errors"] += 1

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [dict(n) for n in self.nodes]


def _http_post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _http_get(url: str, timeout: float = 3.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _health_loop(pool: NodePool, interval: float = 5.0):
    while True:
        for n in pool.nodes:
            try:
                _http_get(n["url"] + "/health", timeout=3.0)
                n["ok"] = True
            except Exception:
                n["ok"] = False
        time.sleep(interval)


def _discover_nodes(timeout: float = 5.0) -> List[str]:
    """Découvre les nœuds vramancer du LAN via mDNS → URLs."""
    import os
    os.environ.setdefault("VRM_EXPERIMENTAL", "1")
    from experimental.cluster_discovery import ClusterDiscovery
    d = ClusterDiscovery()
    d.start()
    time.sleep(timeout)
    nodes = list(d.get_nodes() or [])
    d.stop()
    urls = []
    for nd in nodes:
        ip = nd.get("ip")
        port = nd.get("vramancer_port", 5040)
        if ip:
            urls.append(f"http://{ip}:{port}")
    return urls


def _resolve_urls(nodes: Optional[List[str]], discover: bool) -> List[str]:
    urls = list(nodes or [])
    if discover:
        print("[gateway] découverte mDNS des nœuds…", flush=True)
        urls += _discover_nodes()
    return sorted(set(urls))


def probe(nodes: Optional[List[str]] = None, discover: bool = False) -> int:
    """Pré-vol : liste les nœuds trouvés + leur santé (et modèle chargé), puis quitte."""
    urls = _resolve_urls(nodes, discover)
    if not urls:
        print("[check] aucun nœud (utilise --nodes url1,url2 ou --discover)."); return 1
    print(f"[check] {len(urls)} nœud(s) :")
    reachable = 0
    for u in urls:
        try:
            h = _http_get(u + "/health", timeout=3.0)
            model = h.get("model", "?")
            wk = h.get("alive", h.get("workers", "?"))
            print(f"  ✅ {u}  · modèle={model} · workers={wk}")
            reachable += 1
        except Exception as e:
            print(f"  ❌ {u}  · injoignable ({type(e).__name__})")
    print(f"[check] {reachable}/{len(urls)} joignable(s). "
          f"{'Prêt pour la passerelle.' if reachable else 'Vérifie serve + firewall (port + mDNS UDP 5353).'}")
    return 0 if reachable else 1


def cluster_gateway(nodes: Optional[List[str]] = None, discover: bool = False,
                    host: str = "0.0.0.0", port: int = 5050, req_timeout: float = 300.0) -> None:
    from flask import Flask, request, jsonify
    from werkzeug.serving import make_server

    urls = _resolve_urls(nodes, discover)
    if not urls:
        print("[gateway] aucun nœud (utilise --nodes url1,url2 ou --discover).")
        return
    pool = NodePool(urls)
    print(f"[gateway] {len(urls)} nœud(s) : {', '.join(urls)}", flush=True)
    threading.Thread(target=_health_loop, args=(pool,), daemon=True).start()

    app = Flask(__name__)

    @app.route("/health")
    def health():
        return jsonify({"ok": True, "nodes": pool.snapshot()})

    @app.route("/api/cluster/nodes")
    def cluster_nodes():
        snap = pool.snapshot()
        return jsonify({"ok": True, "node_count": len(snap), "nodes": snap})

    @app.route("/v1/completions", methods=["POST"])
    @app.route("/api/generate", methods=["POST"])
    def completions():
        body = request.get_json(silent=True) or {}
        n = pool.pick()
        if n is None:
            return jsonify({"error": "no node available"}), 503
        t0 = time.perf_counter()
        try:
            out = _http_post(n["url"] + "/v1/completions", body, req_timeout)
            pool.done(n, True)
            out.setdefault("vramancer", {})["node"] = n["url"]
            out["vramancer"]["gateway_s"] = round(time.perf_counter() - t0, 3)
            return jsonify(out)
        except Exception as e:
            pool.done(n, False)
            return jsonify({"error": f"node {n['url']}: {e}"}), 502

    print(f"[gateway] API: http://{host}:{port}/v1/completions  ·  /health", flush=True)
    srv = make_server(host, port, app, threaded=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[gateway] arrêt.")
