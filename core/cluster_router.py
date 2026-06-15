"""ClusterRouter v0 — data-parallel multi-process (la brique qui débloque cross-vendor + cross-nœud).

Insight (mesuré) : un build torch est mono-vendeur (CUDA xor ROCm) → 1 process = 1 vendeur.
Donc cross-vendor (NVIDIA+AMD locales) = MULTI-PROCESS, même archi que le cross-nœud.
Une seule brique : des **workers isolés** (chacun possède 1 GPU via CUDA_VISIBLE_DEVICES),
une **file de travail partagée** (work-stealing = least-loaded automatique), le routeur
distribue des **requêtes entières** (0 crossing par token).

v0 = local multi-process CUDA (testable sans AMD ni 2e machine). Les mêmes workers
peuvent ensuite vivre sur d'autres vendeurs (ROCm) ou d'autres machines (Thunderbolt).

    r = ClusterRouter("Qwen/Qwen2.5-0.5B-Instruct", gpu_ids=[0, 1]); r.start()
    outs = r.submit_batch(["prompt A", "prompt B", ...], max_tokens=64); r.shutdown()
"""
from __future__ import annotations
import os
import queue as _queue
import threading
import time
from typing import Any, Dict, List, Optional

import multiprocessing as _mp


def _worker_main(gpu_id: int, model_name: str, dtype: str, req_q, res_q, ready_q):
    """Process worker : possède 1 GPU, charge le modèle, sert les requêtes de la file."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # AVANT torch -> le worker ne voit que SON GPU
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        td = torch.float16 if dtype == "float16" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=td).to("cuda:0").eval()
        ready_q.put({"gpu_id": gpu_id, "ok": True})
    except Exception as e:  # pragma: no cover
        ready_q.put({"gpu_id": gpu_id, "ok": False, "error": repr(e)})
        return

    import queue as _q
    while True:
        try:
            item = req_q.get(timeout=2.0)
        except _q.Empty:
            if os.getppid() == 1:  # parent mort (SIGKILL/SIGTERM) -> orphelin adopté par init
                break
            continue
        if item is None:  # sentinelle d'arrêt propre
            break
        req_id, prompt, max_tokens = item
        try:
            import torch
            ids = tok(prompt, return_tensors="pt").input_ids.to("cuda:0")
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            gen = out.shape[1] - ids.shape[1]
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            res_q.put({"req_id": req_id, "gpu_id": gpu_id, "text": text,
                       "gen_tokens": int(gen), "dt": dt, "ok": True})
        except Exception as e:  # pragma: no cover
            res_q.put({"req_id": req_id, "gpu_id": gpu_id, "ok": False, "error": repr(e)})


class ClusterRouter:
    """Route des requêtes entières vers des workers mono-GPU isolés (data-parallel)."""

    def __init__(self, model_name: str, gpu_ids: List[int], dtype: str = "float16"):
        self.model_name = model_name
        self.gpu_ids = gpu_ids
        self.dtype = dtype
        self._ctx = _mp.get_context("spawn")  # OBLIGATOIRE pour CUDA (fork casse CUDA)
        self._req_q = self._ctx.Queue()
        self._res_q = self._ctx.Queue()
        self._procs: List[Any] = []
        self._started = False
        # Démux concurrent : un collector lit res_q et réveille le bon appelant par req_id.
        self._pending: Dict[int, Dict[str, Any]] = {}
        self._counter = 0
        self._lock = threading.Lock()
        self._collector: Optional[threading.Thread] = None

    def start(self, timeout: float = 300.0) -> Dict[str, Any]:
        ready_q = self._ctx.Queue()
        for gid in self.gpu_ids:
            p = self._ctx.Process(target=_worker_main,
                                  args=(gid, self.model_name, self.dtype,
                                        self._req_q, self._res_q, ready_q), daemon=True)
            p.start(); self._procs.append(p)
        ready = []
        t0 = time.perf_counter()
        while len(ready) < len(self.gpu_ids):
            if time.perf_counter() - t0 > timeout:
                raise RuntimeError(f"workers non prêts ({len(ready)}/{len(self.gpu_ids)})")
            ready.append(ready_q.get(timeout=timeout))
        failed = [r for r in ready if not r.get("ok")]
        if failed:
            self.shutdown()
            raise RuntimeError(f"worker(s) en échec: {failed}")
        self._started = True
        self._collector = threading.Thread(target=self._collect_loop, daemon=True)
        self._collector.start()
        return {"workers": len(self._procs), "gpu_ids": self.gpu_ids}

    def _collect_loop(self) -> None:
        while self._started:
            try:
                r = self._res_q.get(timeout=1.0)
            except _queue.Empty:
                continue
            except Exception:
                break
            if r is None:
                break
            with self._lock:
                slot = self._pending.pop(r["req_id"], None)
            if slot is not None:
                slot["result"] = r
                slot["event"].set()

    def _new_slot(self, prompt: str, max_tokens: int):
        slot = {"event": threading.Event(), "result": None}
        with self._lock:
            rid = self._counter
            self._counter += 1
            self._pending[rid] = slot
        self._req_q.put((rid, prompt, max_tokens))
        return rid, slot

    def submit(self, prompt: str, max_tokens: int = 64, timeout: float = 300.0) -> Dict[str, Any]:
        """Soumet UNE requête (concurrent-safe) et renvoie son résultat."""
        if not self._started:
            raise RuntimeError("start() d'abord")
        rid, slot = self._new_slot(prompt, max_tokens)
        if not slot["event"].wait(timeout):
            with self._lock:
                self._pending.pop(rid, None)
            raise TimeoutError(f"requête {rid} expirée après {timeout}s")
        return slot["result"]

    def submit_batch(self, prompts: List[str], max_tokens: int = 64,
                     timeout: float = 300.0) -> List[Dict[str, Any]]:
        """Soumet N requêtes ; les workers se les volent (work-stealing). Ordre préservé."""
        if not self._started:
            raise RuntimeError("start() d'abord")
        slots = [self._new_slot(p, max_tokens)[1] for p in prompts]
        for s in slots:
            s["event"].wait(timeout)
        return [s["result"] for s in slots]

    def shutdown(self) -> None:
        self._started = False
        for _ in self._procs:
            try:
                self._req_q.put(None)
            except Exception:
                pass
        for p in self._procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
        self._procs = []


def serve_cluster(model_name: str, gpu_ids: Optional[List[int]] = None,
                  host: str = "0.0.0.0", port: int = 5040, dtype: str = "float16") -> None:
    """Lance un serveur HTTP (OpenAI-compatible) qui route en data-parallel vers les workers.

    `vramancer cluster serve <model>` — utilisable aujourd'hui en local multi-process ;
    l'AMD (cross-vendor) et Thunderbolt (cross-nœud) s'ajouteront dans la même archi.
    """
    import torch
    from flask import Flask, request, jsonify
    from werkzeug.serving import make_server

    if gpu_ids is None:
        gpu_ids = list(range(max(1, torch.cuda.device_count())))
    router = ClusterRouter(model_name, gpu_ids=gpu_ids, dtype=dtype)
    print(f"[cluster] démarrage de {len(gpu_ids)} worker(s) (GPU {gpu_ids})…", flush=True)
    router.start()
    print(f"[cluster] prêt — routeur data-parallel sur {len(gpu_ids)} GPU.", flush=True)

    app = Flask(__name__)

    @app.route("/health")
    def health():
        return jsonify({"ok": True, "workers": len(gpu_ids), "gpu_ids": gpu_ids, "model": model_name})

    @app.route("/v1/completions", methods=["POST"])
    @app.route("/api/generate", methods=["POST"])
    def completions():
        body = request.get_json(silent=True) or {}
        prompt = body.get("prompt", "")
        max_tokens = int(body.get("max_tokens", 64))
        if not prompt:
            return jsonify({"error": "prompt required"}), 400
        try:
            r = router.submit(prompt, max_tokens=max_tokens)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        if not r.get("ok"):
            return jsonify({"error": r.get("error")}), 500
        return jsonify({
            "object": "text_completion", "model": model_name,
            "choices": [{"text": r["text"], "index": 0, "finish_reason": "stop"}],
            "usage": {"completion_tokens": r["gen_tokens"]},
            "vramancer": {"gpu_id": r["gpu_id"], "seconds": round(r["dt"], 3)},
        })

    # Dashboard cluster (best effort) — import direct du sous-module (évite la collision
    # de nom: `from dashboard import dashboard_web` renverrait une fonction).
    try:
        from dashboard.dashboard_web import launch_in_thread
        launch_in_thread(port=port + 1)
        print(f"[cluster] dashboard: http://localhost:{port + 1}/dash", flush=True)
    except Exception as e:
        print(f"[cluster] (dashboard indisponible: {e})", flush=True)

    print(f"[cluster] API: http://{host}:{port}/v1/completions  ·  /health", flush=True)
    srv = make_server(host, port, app, threaded=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[cluster] arrêt…")
    finally:
        router.shutdown()
