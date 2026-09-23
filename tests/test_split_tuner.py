#!/usr/bin/env python3
"""core/split_tuner.py — réglage mesuré de la répartition entre GPU.

Tout est simulé (pas de GPU requis) : on remplace la mesure `llama-bench` par un
modèle de coût qui reproduit ce qui a été mesuré le 2026-09-23 sur 3090 + 7900 XT
(le temps par token est la SOMME des temps par carte → remplir la rapide d'abord).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.split_tuner as st

GIB = 2 ** 30
DEVS = [
    {"id": "Vulkan0", "name": "NVIDIA GeForce RTX 3090", "total_mib": 24576, "free_mib": 24098},
    {"id": "Vulkan1", "name": "AMD Radeon RX 7900 XT", "total_mib": 20464, "free_mib": 20415},
]


def _fake_bench(fast_idx=0, per_gb_fast=0.30, per_gb_slow=1.00, model_gb=27.3, cap=(22.9, 19.3)):
    """Temps/token ∝ somme des Go traités par chaque carte ; None si une carte déborde."""
    def bench(_bin, _model, split, _ctx, _env, **_kw):
        gb = [s * model_gb for s in split]
        if any(g > c for g, c in zip(gb, cap)):
            return None
        cost = [per_gb_fast if i == fast_idx else per_gb_slow for i in range(len(split))]
        return round(1000.0 / sum(g * c for g, c in zip(gb, cost)), 2)
    return bench


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(st, "CACHE_FILE", tmp_path / "splits.json")
    model = tmp_path / "m.gguf"
    with open(model, "wb") as f:
        f.seek(int(27.3 * GIB) - 1)
        f.write(b"\0")                      # fichier creux de 27.3 GiB
    binary = tmp_path / "llama-server"
    binary.write_text("")
    (tmp_path / "llama-bench").write_text("")
    import core.llama_server_backend as lsb
    monkeypatch.setattr(lsb, "backend_devices", lambda b: DEVS)
    monkeypatch.setattr(lsb, "_runtime_env", lambda b: {})
    return str(model), binary


def test_candidates_start_with_vram_proportional():
    c = st.candidate_splits(DEVS, int(27.3 * GIB))
    assert c[0] == [round(24576 / 45040, 4), round(20464 / 45040, 4)]


def test_fill_cap_respects_free_vram_and_reserve():
    cap = st.fill_cap(DEVS, 0, int(27.3 * GIB), reserve_mib=1024)
    assert 0.80 < cap < 0.85                 # (24098-1024)/27955 ≈ 0.825


def test_tuner_finds_fast_card_and_beats_proportional(isolated, monkeypatch):
    model, binary = isolated
    monkeypatch.setattr(st, "_bench", _fake_bench(fast_idx=0))
    res = st.tune(model, binary, n_ctx=16384, report=lambda *_: None)
    assert res is not None
    assert res["split"][0] > 0.75            # la 3090 (rapide) prend le gros
    assert res["tok_s"] > res["baseline_tok_s"]


def test_tuner_discovers_when_the_OTHER_card_is_faster(isolated, monkeypatch):
    """La carte rapide ne se devine pas : si c'est l'AMD, le tuner doit le trouver."""
    model, binary = isolated
    monkeypatch.setattr(st, "_bench", _fake_bench(fast_idx=1))
    res = st.tune(model, binary, n_ctx=16384, report=lambda *_: None)
    assert res["split"][1] > res["split"][0]


def test_tuner_never_keeps_a_split_that_does_not_fit(isolated, monkeypatch):
    model, binary = isolated
    monkeypatch.setattr(st, "_bench", _fake_bench(fast_idx=0, cap=(20.0, 19.3)))
    res = st.tune(model, binary, n_ctx=16384, report=lambda *_: None)
    assert res["split"][0] * 27.3 <= 20.0 + 1e-6


def test_result_is_cached_and_reused(isolated, monkeypatch):
    model, binary = isolated
    monkeypatch.setattr(st, "_bench", _fake_bench())
    res = st.tune(model, binary, n_ctx=16384, report=lambda *_: None)
    assert st.cached_split(model, DEVS, 16384) == res["split"]
    assert st.cached_split(model, DEVS, 32768) is None     # autre contexte = autre clé


def test_no_split_for_single_gpu(isolated):
    model, _ = isolated
    assert st.cached_split(model, DEVS[:1], 16384) is None
