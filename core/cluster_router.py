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

    while True:
        item = req_q.get()
        if item is None:  # sentinelle d'arrêt
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
        return {"workers": len(self._procs), "gpu_ids": self.gpu_ids}

    def submit_batch(self, prompts: List[str], max_tokens: int = 64) -> List[Dict[str, Any]]:
        """Soumet N requêtes ; les workers se les volent (work-stealing). Renvoie les résultats."""
        if not self._started:
            raise RuntimeError("start() d'abord")
        for i, p in enumerate(prompts):
            self._req_q.put((i, p, max_tokens))
        results = [None] * len(prompts)
        for _ in range(len(prompts)):
            r = self._res_q.get()
            results[r["req_id"]] = r
        return results

    def shutdown(self) -> None:
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
        self._started = False
