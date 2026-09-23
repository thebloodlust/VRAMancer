#!/usr/bin/env python3
"""core/predict.py — prédire avant de télécharger (en-tête par plages HTTP + modèle de coût)."""
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.predict as pr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_llama_server_backend import _write_gguf_tensors  # noqa: E402

GIB = 2 ** 30
DEVS = [{"name": "NVIDIA GeForce RTX 3090", "total_mib": 24576, "free_mib": 24098},
        {"name": "AMD Radeon RX 7900 XT (RADV NAVI31)", "total_mib": 20464, "free_mib": 20415}]


def test_dense_that_fits_matches_measurement():
    """Qwen2.5-32B Q4_K_M sur la 3090 seule : mesuré 36.1-38.3 tok/s."""
    p = pr.Profile(n_layers=64, total=int(18.48 * GIB), experts=0, arch="qwen2")
    pred = pr.predict(p, pr.machine_tiers(DEVS[:1], ram_avail_gib=160))
    assert pred.regime == "tient en VRAM"
    assert 34 < pred.tok_s < 40


def test_moe_spills_experts_after_hot_part():
    p = pr.Profile(n_layers=43, total=int(81 * GIB), experts=int(74.8 * GIB),
                   expert_count=256, expert_used=6, arch="deepseek4")
    pred = pr.predict(p, pr.machine_tiers(DEVS, ram_avail_gib=160))
    names = [n for n, _ in pred.placement]
    assert names[0].endswith("3090") and "RAM" in names
    assert sum(g for _, g in pred.placement) == pytest.approx(81, abs=0.5)
    assert "non calibrée" in pr.fmt_prediction(p, pred)     # DeepSeek-V4 : borne haute


def test_bigger_than_ram_goes_to_disk_instead_of_failing():
    p = pr.Profile(n_layers=80, total=int(400 * GIB), experts=0, arch="llama")
    pred = pr.predict(p, pr.machine_tiers(DEVS[:1], ram_avail_gib=64))
    assert "disque" in pred.regime and pred.placement[-1][0] == "disque (mmap)"
    assert 0 < pred.tok_s < 1


class _RangeHandler(BaseHTTPRequestHandler):
    data = b""

    def log_message(self, *a):
        pass

    def do_GET(self):
        a, b = self.headers["Range"].removeprefix("bytes=").split("-")
        a, b = int(a), min(int(b), len(self.data) - 1)
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {a}-{b}/{len(self.data)}")
        self.send_header("Content-Length", str(b - a + 1))
        self.end_headers()
        self.wfile.write(self.data[a:b + 1])


def test_remote_profile_reads_only_the_header(tmp_path):
    m = tmp_path / "m.gguf"
    _write_gguf_tensors(m, [("blk.0.ffn_up_exps.weight", 12, 3 << 20), ("blk.0.attn_q.weight", 12, 1 << 20)],
                        kv={"general.architecture": "qwen3moe", "qwen3moe.block_count": 1,
                            "qwen3moe.expert_count": 128, "qwen3moe.expert_used_count": 8})
    _RangeHandler.data = m.read_bytes()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        p = pr.remote_profile(f"http://127.0.0.1:{httpd.server_address[1]}/m.gguf")
    finally:
        httpd.shutdown()
    assert (p.total, p.experts, p.expert_count, p.expert_used) == (4 << 20, 3 << 20, 128, 8)
    assert p.fetched < len(_RangeHandler.data)             # pas tout le fichier


def test_hf_url():
    assert pr.hf_url("org/repo/sub/f.gguf") == "https://huggingface.co/org/repo/resolve/main/sub/f.gguf"
