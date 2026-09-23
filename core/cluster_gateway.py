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
    """Routage « moins de connexions PONDÉRÉ » par la vitesse mesurée de chaque nœud.

    Mesuré le 2026-09-23 (RTX 3090 + RX 7900 XT, même modèle sur chaque nœud,
    16 requêtes, 4 en parallèle) avec l'ancien routage « least-loaded » pur :
    3090 seule 150.6 tok/s ; passerelle 3090 + 7900 XT **104.1 tok/s** (−31 %),
    p95 16.2 s au lieu de 5.8 s. Ajouter un nœud lent FAISAIT BAISSER le débit : le
    nœud lent recevait autant de travail que le rapide dès qu'il avait moins de
    requêtes en cours. On minimise désormais (en_cours + 1) / vitesse, où la vitesse
    est une moyenne glissante des tokens/s observés sur les réponses de CE nœud.

    Deuxième défaut corrigé : le « trou noir ». Un nœud mort échoue instantanément,
    donc paraît toujours le moins chargé et aspirait le trafic jusqu'au prochain
    contrôle de santé (5 s) — 12 requêtes sur 16 perdues en test. Un échec le retire
    désormais immédiatement ; le contrôle de santé le réintègre quand il répond.
    """

    EWMA = 0.3

    def __init__(self, urls: List[str]):
        self.nodes = [{"url": u.rstrip("/"), "inflight": 0, "ok": True,
                       "served": 0, "errors": 0, "tok_s": None} for u in urls]
        self._lock = threading.Lock()

    def pick(self, exclude=()) -> Optional[Dict[str, Any]]:
        with self._lock:
            cand = [n for n in self.nodes if n["ok"] and n["url"] not in exclude]
            if not cand:
                cand = [n for n in self.nodes if n["url"] not in exclude]  # tente quand même
            if not cand:
                return None
            # nœud jamais mesuré : on le teste une fois (sinon sa vitesse reste inconnue)
            unknown = [n for n in cand if n["tok_s"] is None and n["inflight"] == 0]
            if unknown:
                n = unknown[0]
            else:
                known = [n["tok_s"] for n in cand if n["tok_s"]]
                default = sum(known) / len(known) if known else 1.0
                n = min(cand, key=lambda x: (x["inflight"] + 1) / (x["tok_s"] or default))
            n["inflight"] += 1
            return n

    def done(self, n: Dict[str, Any], ok: bool, tokens: int = 0, seconds: float = 0.0):
        with self._lock:
            n["inflight"] = max(0, n["inflight"] - 1)
            if ok:
                n["served"] += 1
                if tokens > 0 and seconds > 0:
                    speed = tokens / seconds
                    n["tok_s"] = speed if n["tok_s"] is None else \
                        (1 - self.EWMA) * n["tok_s"] + self.EWMA * speed
            else:
                n["errors"] += 1
                n["ok"] = False          # retiré tout de suite ; _health_loop le réintègre

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

    def _proxy(path: str):
        """Transmet la requête au meilleur nœud ; en cas d'échec, UN nouvel essai ailleurs."""
        body = request.get_json(silent=True) or {}
        tried: List[str] = []
        last_err = "no node available"
        for _attempt in range(2):
            n = pool.pick(exclude=tried)
            if n is None:
                break
            tried.append(n["url"])
            t0 = time.perf_counter()
            try:
                out = _http_post(n["url"] + path, body, req_timeout)
                dt = time.perf_counter() - t0
                toks = int((out.get("usage") or {}).get("completion_tokens") or 0)
                pool.done(n, True, tokens=toks, seconds=dt)
                out.setdefault("vramancer", {})["node"] = n["url"]
                out["vramancer"]["gateway_s"] = round(dt, 3)
                if len(tried) > 1:
                    out["vramancer"]["retried_from"] = tried[0]
                return jsonify(out)
            except Exception as e:
                pool.done(n, False)
                last_err = f"node {n['url']}: {e}"
        return jsonify({"error": last_err}), 502 if tried else 503

    @app.route("/v1/completions", methods=["POST"])
    @app.route("/api/generate", methods=["POST"])
    def completions():
        return _proxy("/v1/completions")

    # Les agents de code (Aider, Cline, Continue…) parlent /v1/chat/completions.
    # Sans cette route, le cas d'usage phare du projet ne passait pas par le cluster (404).
    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions():
        return _proxy("/v1/chat/completions")

    @app.route("/v1/models")
    def models():
        for n in pool.snapshot():
            if n["ok"]:
                try:
                    return jsonify(_http_get(n["url"] + "/v1/models", timeout=5.0))
                except Exception:
                    continue
        return jsonify({"object": "list", "data": []})

    print(f"[gateway] API: http://{host}:{port}/v1/completions  ·  /health", flush=True)
    srv = make_server(host, port, app, threaded=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[gateway] arrêt.")
