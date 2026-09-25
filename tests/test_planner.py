#!/usr/bin/env python3
"""core/planner.py — placement des poids par étages, calibré par la mesure.

Toutes les mesures llama-bench sont simulées par un modèle de coût linéaire qui reproduit
les ordres de grandeur mesurés le 2026-09-23 (3090 + 7900 XT + EPYC) : aucun GPU requis.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.planner as pl

GIB = 2 ** 30
DEVS = [
    {"id": "Vulkan0", "name": "NVIDIA GeForce RTX 3090", "total_mib": 24576, "free_mib": 24098},
    {"id": "Vulkan1", "name": "AMD Radeon RX 7900 XT", "total_mib": 20464, "free_mib": 20415},
]


# ── expert_args ────────────────────────────────────────────────────────────

def test_expert_args_places_top_layers_on_primary_then_secondary_then_ram():
    a = pl.expert_args(43, 9, [("Vulkan1", 10)], 2)
    assert a[:4] == ["-ngl", "99", "-ts", "99/1"]      # 2e GPU gardé déclaré (piège -sm none)
    rules = a[a.index("-ot") + 1].split(";")
    assert rules[0].startswith(r"blk\.(24|25|26|27|28|29|30|31|32|33)\.") and rules[0].endswith("=Vulkan1")
    assert rules[1].startswith(r"blk\.(0|1|") and rules[1].endswith(r"|23)\.ffn_(gate|up|down)_exps\.weight=CPU")


def test_expert_args_all_on_primary_has_no_override():
    assert pl.expert_args(40, 40, [], 1) == ["-ngl", "99"]


def test_expert_args_single_gpu_everything_else_in_ram():
    a = pl.expert_args(10, 3, [], 1)
    assert "-ts" not in a
    assert a[-1] == r"blk\.(0|1|2|3|4|5|6)\.ffn_(gate|up|down)_exps\.weight=CPU"


def test_shards_detects_split_gguf(tmp_path):
    assert pl._shards("/m/x-00001-of-00003.gguf") == [
        "/m/x-00001-of-00003.gguf", "/m/x-00002-of-00003.gguf", "/m/x-00003-of-00003.gguf"]
    assert pl._shards("/m/single.gguf") == ["/m/single.gguf"]


# ── régime MoE par étages ──────────────────────────────────────────────────

def _moe_profile():
    per_layer = {i: int(1.74 * GIB) for i in range(43)}
    return pl.ModelProfile(path="/m/ds-00001-of-00002.gguf", shards=["/m/ds-00001-of-00002.gguf"],
                           n_layers=43, total_bytes=int(6.2 * GIB) + sum(per_layer.values()),
                           expert_bytes_per_layer=per_layer, expert_count=256, expert_used=6)


def _moe_bench(c_primary=0.2e-3, c_secondary=1.5e-3, c_ram=2.0e-3, base=10e-3):
    """t = base + Σ couches·coût de l'étage ; lit la répartition dans les arguments."""
    def bench(binary, model, args, env, **kw):
        rules = args[args.index("-ot") + 1].split(";") if "-ot" in args else []
        n = {"Vulkan1": 0, "CPU": 0}
        for r in rules:
            layers = r.split("(")[1].split(")")[0].split("|")
            n[r.rsplit("=", 1)[1]] += len(layers)
        n_primary = 43 - n["Vulkan1"] - n["CPU"]
        t = base + n_primary * c_primary + n["Vulkan1"] * c_secondary + n["CPU"] * c_ram
        return pl.Measure(pp=30.0, tg=1 / t)
    return bench


def test_moe_tiers_uses_secondary_gpu_when_it_beats_ram(monkeypatch):
    monkeypatch.setattr(pl, "bench", _moe_bench(c_secondary=1.5e-3, c_ram=2.0e-3))
    p = pl.plan_moe_tiers(_moe_profile(), DEVS, "/b/llama-server", {}, report=lambda *a: None)
    assert p is not None and p.detail["experts_secondaires"]["Vulkan1"] > 0
    assert p.detail["experts_principal"] > 0


def test_moe_tiers_rejects_secondary_gpu_slower_than_ram(monkeypatch):
    """C'est au planificateur de dire si le 2e GPU vaut le coup, pour CE modèle."""
    monkeypatch.setattr(pl, "bench", _moe_bench(c_secondary=2.5e-3, c_ram=2.0e-3))
    p = pl.plan_moe_tiers(_moe_profile(), DEVS, "/b/llama-server", {}, report=lambda *a: None)
    assert p.detail["experts_secondaires"]["Vulkan1"] == 0


def test_moe_tiers_backs_off_instead_of_crashing(monkeypatch):
    """Garantie « ne jamais planter » : une config qui ne tient pas fait reculer d'un cran."""
    base = _moe_bench()

    def bench(binary, model, args, env, **kw):
        rules = args[args.index("-ot") + 1].split(";") if "-ot" in args else []
        on_cpu_or_sec = sum(len(r.split("(")[1].split(")")[0].split("|")) for r in rules)
        if 43 - on_cpu_or_sec > 7:              # plus de 7 couches d'experts sur le principal : OOM
            return pl.Measure(None, None)
        return base(binary, model, args, env, **kw)

    monkeypatch.setattr(pl, "bench", bench)
    p = pl.plan_moe_tiers(_moe_profile(), DEVS, "/b/llama-server", {}, report=lambda *a: None)
    assert p is not None and p.detail["experts_principal"] <= 7


# ── régime répartition des couches ─────────────────────────────────────────

def test_layer_split_fills_cheapest_gpu_first(monkeypatch):
    """t = Σ f_d·T_d : le planificateur doit retrouver « la 3090 d'abord »."""
    T = [0.0087, 0.0136]                          # coûts mesurés sur Qwen3.6 Q6_K (s/token)

    def bench(binary, model, args, env, **kw):
        f = [float(x) for x in args[args.index("-ts") + 1].split("/")]
        return pl.Measure(pp=2000.0, tg=1 / (f[0] * T[0] + f[1] * T[1]))

    monkeypatch.setattr(pl, "bench", bench)
    prof = pl.ModelProfile(path="/m/q.gguf", shards=["/m/q.gguf"], n_layers=40,
                           total_bytes=int(27.3 * GIB), expert_bytes_per_layer={})
    p = pl.plan_layer_split(prof, DEVS, "/b/llama-server", {}, report=lambda *a: None)
    assert p.detail["split"][0] > 0.75           # la 3090 remplie, comme tune-split
    assert p.predicted_tg > 1 / (0.545 * T[0] + 0.455 * T[1])   # mieux que le prorata


def test_cached_plan_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(pl, "PLAN_CACHE", tmp_path / "plans.json")
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x" * 10)
    pl._save(pl.Plan(model=str(model), regime="layer-split", args=["-ngl", "99", "-ts", "0.83/0.17"],
                     predicted_tg=106.8, measured=pl.Measure(900.0, 107.3)))
    assert pl.cached_plan_args(str(model)) == ["-ngl", "99", "-ts", "0.83/0.17"]
    model.write_bytes(b"x" * 11)                        # autre fichier (taille) → autre clé
    assert pl.cached_plan_args(str(model)) is None


def test_model_size_counts_every_shard(tmp_path):
    from core.llama_server_backend import _model_size_gib
    for k in (1, 2):
        (tmp_path / f"m-0000{k}-of-00002.gguf").write_bytes(b"x" * 2 ** 20)
    assert _model_size_gib(tmp_path / "m-00001-of-00002.gguf") == pytest.approx(2 / 1024)
