"""Tests for core.env_flags — typed accessor facade for VRM_* env vars."""
import os
import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Remove all VRM_* vars before each test to avoid cross-test contamination."""
    for key in list(os.environ):
        if key.startswith("VRM_"):
            monkeypatch.delenv(key, raising=False)
    yield


# ── bool flags ───────────────────────────────────────────────────────────────

def test_vram_lending_default_true():
    from core.env_flags import flags
    assert flags.VRAM_LENDING is True


def test_vram_lending_disabled():
    os.environ["VRM_VRAM_LENDING"] = "0"
    from core.env_flags import flags
    assert flags.VRAM_LENDING is False


def test_force_multi_gpu_default_false():
    from core.env_flags import flags
    assert flags.FORCE_MULTI_GPU is False


def test_force_multi_gpu_enabled():
    os.environ["VRM_FORCE_MULTI_GPU"] = "1"
    from core.env_flags import flags
    assert flags.FORCE_MULTI_GPU is True


def test_spec_adaptive_default_true():
    from core.env_flags import flags
    assert flags.SPEC_ADAPTIVE is True


def test_spec_adaptive_disabled():
    os.environ["VRM_SPEC_ADAPTIVE"] = "0"
    from core.env_flags import flags
    assert flags.SPEC_ADAPTIVE is False


def test_disable_turbo_default_false():
    from core.env_flags import flags
    assert flags.DISABLE_TURBO is False


def test_cuda_graph_default_false():
    from core.env_flags import flags
    assert flags.CUDA_GRAPH is False


# ── int flags ─────────────────────────────────────────────────────────────────

def test_spec_gamma_default():
    from core.env_flags import flags
    assert flags.SPEC_GAMMA == 5


def test_spec_gamma_override():
    os.environ["VRM_SPEC_GAMMA"] = "8"
    from core.env_flags import flags
    assert flags.SPEC_GAMMA == 8


def test_max_batch_size_default():
    from core.env_flags import flags
    assert flags.MAX_BATCH_SIZE == 32


def test_cuda_graph_cache_default():
    from core.env_flags import flags
    assert flags.CUDA_GRAPH_CACHE == 4


def test_kv_compression_bits_default():
    from core.env_flags import flags
    assert flags.KV_COMPRESSION_BITS == 3


# ── float flags ───────────────────────────────────────────────────────────────

def test_lend_ratio_default():
    from core.env_flags import flags
    assert abs(flags.LEND_RATIO - 0.70) < 1e-9


def test_reclaim_threshold_default():
    from core.env_flags import flags
    assert abs(flags.RECLAIM_THRESHOLD - 0.80) < 1e-9


def test_generate_timeout_default():
    from core.env_flags import flags
    assert flags.GENERATE_TIMEOUT == 300.0


# ── str flags ─────────────────────────────────────────────────────────────────

def test_parallel_mode_default():
    from core.env_flags import flags
    assert flags.PARALLEL_MODE == "pp"


def test_quantization_default_empty():
    from core.env_flags import flags
    assert flags.QUANTIZATION == ""


def test_kv_compression_default_empty():
    from core.env_flags import flags
    assert flags.KV_COMPRESSION == ""


def test_draft_model_default_none():
    from core.env_flags import flags
    assert flags.DRAFT_MODEL is None


def test_draft_model_set():
    os.environ["VRM_DRAFT_MODEL"] = "Qwen/Qwen2.5-0.5B"
    from core.env_flags import flags
    assert flags.DRAFT_MODEL == "Qwen/Qwen2.5-0.5B"


# ── live env mutation ─────────────────────────────────────────────────────────

def test_flags_live_read(monkeypatch):
    """Flags re-read os.environ on each access (no caching)."""
    from core.env_flags import flags
    monkeypatch.setenv("VRM_MAX_BATCH_SIZE", "16")
    assert flags.MAX_BATCH_SIZE == 16
    monkeypatch.setenv("VRM_MAX_BATCH_SIZE", "64")
    assert flags.MAX_BATCH_SIZE == 64
