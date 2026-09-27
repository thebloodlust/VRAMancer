#!/usr/bin/env python3
"""Apple Silicon : choix du backend (MLX / llama.cpp Metal) et prédiction en mémoire unifiée.

Pas de Mac dans la VM : on simule la plateforme. La mesure réelle reste à faire (Mac M5).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.backends as be
import core.backends_mlx as bm
import core.predict as pr

GIB = 2 ** 30


def _apple(monkeypatch, mlx=True, binding=False):
    monkeypatch.setattr(bm, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(bm, "mlx_available", lambda: mlx)
    monkeypatch.setattr(be, "_llamacpp_available", lambda: binding)


def test_mlx_model_on_apple_silicon_goes_to_mlx(monkeypatch):
    _apple(monkeypatch)
    b = be.select_backend("mlx-community/Qwen2.5-7B-Instruct-4bit")
    assert type(b).__name__ == "MlxServerAdapter"


def test_gguf_without_binding_goes_to_llama_server_metal(monkeypatch):
    """Sans llama-cpp-python, un GGUF ne doit pas retomber sur HuggingFace (lent)."""
    _apple(monkeypatch)
    b = be.select_backend("/models/qwen.gguf")
    assert type(b).__name__ == "LlamaServerAdapter"


def test_explicit_mlx_backend(monkeypatch):
    assert type(be.select_backend("mlx-community/x", backend="mlx")).__name__ == "MlxServerAdapter"


def test_unified_memory_not_counted_twice():
    """Mac M5 32 Go : ce que le GPU Metal prend est retiré de l'étage RAM."""
    devs = [{"name": "Apple M5", "total_mib": 24576, "free_mib": 24576}]
    tiers = pr.machine_tiers(devs, ram_avail_gib=30)
    gpu, ram = tiers[0], tiers[1]
    assert gpu.name == "Apple M5" and gpu.gibs < 153          # nominal × 0.7
    assert ram.capacity == 30 - 4.0 - gpu.capacity


def test_longest_apple_name_wins():
    devs = [{"name": "Apple M4 Max", "total_mib": 49152, "free_mib": 49152}]
    assert pr.machine_tiers(devs, 64)[0].gibs == pr.APPLE_PARAMS["Apple M4 Max"][0]
