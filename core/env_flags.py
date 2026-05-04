"""Centralised environment flags for VRAMancer.

Every ``VRM_*`` environment variable read by core/ is declared here with its
default value and a short description.  Modules should import the typed
accessor instead of scattering ``os.environ.get("VRM_…")`` everywhere.

Usage::

    from core.env_flags import flags
    if flags.MINIMAL_TEST:
        ...
    port = flags.API_PORT

The module is intentionally *stateless* — each attribute read hits
``os.environ`` so that runtime changes (e.g. test fixtures doing
``os.environ["VRM_X"] = "1"``) are immediately visible.
"""
from __future__ import annotations

import os
from typing import Optional


def _bool(key: str, default: str = "") -> bool:
    """Return True when the env var is a truthy string."""
    return os.environ.get(key, default).lower() in ("1", "true", "yes")


def _int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


def _float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


def _str(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _opt_str(key: str) -> Optional[str]:
    return os.environ.get(key) or None


# ---------------------------------------------------------------------------
# Typed flag facade — each property is a live read of os.environ
# ---------------------------------------------------------------------------

class _Flags:
    """Live-read facade over ``os.environ`` for all VRM_* flags."""

    # ── Modes ──────────────────────────────────────────────────────────
    @property
    def MINIMAL_TEST(self) -> bool:
        """Lightweight test mode — stubs heavy backends."""
        return _bool("VRM_MINIMAL_TEST")

    @property
    def TEST_MODE(self) -> bool:
        """Generic test-mode flag (relaxes some runtime checks)."""
        return _bool("VRM_TEST_MODE")

    @property
    def PRODUCTION(self) -> bool:
        """Production hardening — enforces tokens, HSTS, startup checks."""
        return _bool("VRM_PRODUCTION")

    @property
    def DEBUG(self) -> bool:
        return _bool("VRM_DEBUG")

    @property
    def STRICT_IMPORT(self) -> bool:
        """Raise on missing optional dependency instead of falling back."""
        return _bool("VRM_STRICT_IMPORT")

    @property
    def READ_ONLY(self) -> bool:
        """Block mutating API endpoints."""
        return _bool("VRM_READ_ONLY")

    # ── Inference ──────────────────────────────────────────────────────
    @property
    def BACKEND(self) -> str:
        return _str("VRM_BACKEND", "auto")

    @property
    def MODEL(self) -> str:
        return _str("VRM_MODEL", "gpt2")

    @property
    def QUANTIZATION(self) -> str:
        return _str("VRM_QUANTIZATION", "").lower()

    @property
    def PARALLEL_MODE(self) -> str:
        """'pp' (pipeline parallelism) or 'tp' (tensor parallelism)."""
        return _str("VRM_PARALLEL_MODE", "pp").lower()

    @property
    def FORCE_MULTI_GPU(self) -> bool:
        return _bool("VRM_FORCE_MULTI_GPU")

    @property
    def MAX_BATCH_SIZE(self) -> int:
        return _int("VRM_MAX_BATCH_SIZE", 32)

    @property
    def GENERATE_TIMEOUT(self) -> float:
        return _float("VRM_GENERATE_TIMEOUT", 300.0)

    @property
    def MAX_PROMPT_LENGTH(self) -> int:
        return _int("VRM_MAX_PROMPT_LENGTH", 100_000)

    @property
    def TRUST_REMOTE_CODE(self) -> bool:
        return _bool("VRM_TRUST_REMOTE_CODE")

    @property
    def CONTINUOUS_BATCHING(self) -> bool:
        return _bool("VRM_CONTINUOUS_BATCHING")

    @property
    def PREFILL_CHUNK(self) -> int:
        return _int("VRM_PREFILL_CHUNK", 512)

    # ── Speculative decoding ───────────────────────────────────────────
    @property
    def DRAFT_MODEL(self) -> Optional[str]:
        return _opt_str("VRM_DRAFT_MODEL")

    @property
    def SPEC_GAMMA(self) -> int:
        return _int("VRM_SPEC_GAMMA", 5)

    @property
    def SPEC_ADAPTIVE(self) -> bool:
        return _str("VRM_SPEC_ADAPTIVE", "1") != "0"

    @property
    def SPEC_WINDOW(self) -> int:
        return _int("VRM_SPEC_WINDOW", 32)

    # ── TurboEngine / CUDA graphs ─────────────────────────────────────
    @property
    def DISABLE_TURBO(self) -> bool:
        return _bool("VRM_DISABLE_TURBO")

    @property
    def TURBO_MAX_SEQ(self) -> int:
        return _int("VRM_TURBO_MAX_SEQ", 2048)

    @property
    def CUDA_GRAPH(self) -> bool:
        return _bool("VRM_CUDA_GRAPH")

    @property
    def CUDA_GRAPH_CACHE(self) -> int:
        return _int("VRM_CUDA_GRAPH_CACHE", 4)

    @property
    def CUDA_GRAPH_WARMUP(self) -> int:
        return _int("VRM_CUDA_GRAPH_WARMUP", 3)

    # ── KV cache / compression ─────────────────────────────────────────
    @property
    def KV_COMPRESSION(self) -> str:
        return _str("VRM_KV_COMPRESSION", "").lower()

    @property
    def KV_COMPRESSION_BITS(self) -> int:
        return _int("VRM_KV_COMPRESSION_BITS", 3)

    @property
    def KV_CACHE_RESIDUAL(self) -> int:
        return _int("VRM_KV_CACHE_RESIDUAL", 128)

    @property
    def SPARSE_V_RATIO(self) -> float:
        return _float("VRM_SPARSE_V_RATIO", 1.0)

    # ── Transfer / memory ──────────────────────────────────────────────
    @property
    def TRANSFER_METHOD(self) -> str:
        return _str("VRM_TRANSFER_METHOD", "").lower()

    @property
    def TRANSFER_P2P(self) -> bool:
        """False only when explicitly disabled."""
        val = _str("VRM_TRANSFER_P2P", "").lower()
        return val not in ("0", "false", "no") if val else True

    @property
    def VRAM_LENDING(self) -> bool:
        return _str("VRM_VRAM_LENDING", "1").lower() not in ("0", "false", "no")

    @property
    def LEND_RATIO(self) -> float:
        return _float("VRM_LEND_RATIO", 0.70)

    @property
    def RECLAIM_THRESHOLD(self) -> float:
        return _float("VRM_RECLAIM_THRESHOLD", 0.80)

    @property
    def LENDING_INTERVAL(self) -> float:
        return _float("VRM_LENDING_INTERVAL", 2.0)

    @property
    def REBALANCE_INTERVAL(self) -> float:
        return _float("VRM_REBALANCE_INTERVAL", 5.0)

    # ── Networking / cluster ───────────────────────────────────────────
    @property
    def API_HOST(self) -> str:
        return _str("VRM_API_HOST", "0.0.0.0")

    @property
    def API_PORT(self) -> int:
        return _int("VRM_API_PORT", 5030)

    @property
    def API_TOKEN(self) -> Optional[str]:
        return _opt_str("VRM_API_TOKEN")

    @property
    def METRICS_PORT(self) -> int:
        return _int("VRM_METRICS_PORT", 9108)

    @property
    def METRICS_BIND(self) -> str:
        return _str("VRM_METRICS_BIND", "127.0.0.1")

    @property
    def NODE_ID(self) -> str:
        return _str("VRM_NODE_ID", "local")

    @property
    def PEER_IPS(self) -> str:
        return _str("VRM_PEER_IPS", "")

    @property
    def CLUSTER_SECRET(self) -> Optional[str]:
        return _opt_str("VRM_CLUSTER_SECRET")

    @property
    def SAME_RACK_NODES(self) -> str:
        return _str("VRM_SAME_RACK_NODES", "")

    # ── AITP / VTP ─────────────────────────────────────────────────────
    @property
    def AITP_PORT(self) -> int:
        return _int("VRM_AITP_PORT", 55555)

    @property
    def AITP_MAX_QUEUE(self) -> int:
        return _int("VRM_AITP_MAX_QUEUE", 64)

    @property
    def AITP_STAGING_MB(self) -> int:
        return _int("VRM_AITP_STAGING_MB", 256)

    @property
    def VTP_ENABLED(self) -> bool:
        return _bool("VRM_VTP_ENABLED")

    @property
    def VTP_PORT(self) -> int:
        return _int("VRM_VTP_PORT", 55556)

    @property
    def VTP_CHUNK_MB(self) -> int:
        return _int("VRM_VTP_CHUNK_MB", 64)

    @property
    def VTP_CREDITS(self) -> int:
        return _int("VRM_VTP_CREDITS", 4)

    @property
    def VTP_MAX_INFLIGHT(self) -> int:
        return _int("VRM_VTP_MAX_INFLIGHT", 8)

    @property
    def TRANSPORT_TIMEOUT(self) -> float:
        return _float("VRM_TRANSPORT_TIMEOUT", 30.0)

    @property
    def TRANSPORT_INSECURE(self) -> bool:
        return _bool("VRM_TRANSPORT_INSECURE")

    @property
    def DDNS_HOST(self) -> Optional[str]:
        return _opt_str("VRM_DDNS_HOST")

    # ── WebGPU / WebNPU ────────────────────────────────────────────────
    @property
    def WEBGPU_WS_PORT(self) -> int:
        return _int("VRM_WEBGPU_WS_PORT", 9090)

    @property
    def WEBGPU_HTTP_PORT(self) -> int:
        return _int("VRM_WEBGPU_HTTP_PORT", 8443)

    @property
    def WEBGPU_TIMEOUT(self) -> float:
        return _float("VRM_WEBGPU_TIMEOUT", 30.0)

    @property
    def WEBGPU_NO_SSL(self) -> bool:
        return _bool("VRM_WEBGPU_NO_SSL")

    # ── Security ───────────────────────────────────────────────────────
    @property
    def AUTH_SECRET(self) -> Optional[str]:
        return _opt_str("VRM_AUTH_SECRET")

    @property
    def CORS_ORIGINS(self) -> str:
        return _str("VRM_CORS_ORIGINS", "http://localhost,http://127.0.0.1")

    @property
    def RATE_MAX(self) -> int:
        return _int("VRM_RATE_MAX", 200)

    @property
    def MAX_BODY(self) -> int:
        return _int("VRM_MAX_BODY", 5_242_880)

    @property
    def DISABLE_RATE_LIMIT(self) -> bool:
        return _bool("VRM_DISABLE_RATE_LIMIT")

    @property
    def DISABLE_SECRET_ROTATION(self) -> bool:
        return _bool("VRM_DISABLE_SECRET_ROTATION")

    # ── Persistence / observability ────────────────────────────────────
    @property
    def SQLITE_PATH(self) -> Optional[str]:
        return _opt_str("VRM_SQLITE_PATH")

    @property
    def DATA_DIR(self) -> str:
        return _str("VRM_DATA_DIR", ".")

    @property
    def LOG_JSON(self) -> bool:
        return _bool("VRM_LOG_JSON")

    @property
    def LOG_FILE(self) -> Optional[str]:
        return _opt_str("VRM_LOG_FILE")

    @property
    def TRACING(self) -> bool:
        return _bool("VRM_TRACING")

    # ── GPU / heterogeneous ────────────────────────────────────────────
    @property
    def HETERO_STRATEGY(self) -> str:
        return _str("VRM_HETERO_STRATEGY", "").lower()

    @property
    def GPU_ORDER(self) -> str:
        return _str("VRM_GPU_ORDER", "")

    @property
    def PYNVML_TIMEOUT(self) -> int:
        return _int("VRM_PYNVML_TIMEOUT", 5)

    # ── Tokenizer ──────────────────────────────────────────────────────
    @property
    def TOKENIZER_WORKERS(self) -> int:
        return _int("VRM_TOKENIZER_WORKERS", 4)

    @property
    def FORCE_BASIC_TOKENIZER(self) -> bool:
        return _bool("VRM_FORCE_BASIC_TOKENIZER")

    # ── Misc ───────────────────────────────────────────────────────────
    @property
    def DISABLE_ONNX(self) -> bool:
        return _bool("VRM_DISABLE_ONNX")

    @property
    def DISABLE_SOCKETIO(self) -> bool:
        return _bool("VRM_DISABLE_SOCKETIO")

    @property
    def LLAMA_SERVER_PORT(self) -> int:
        return _int("VRM_LLAMA_SERVER_PORT", 8080)

    # ── Test-only flags ────────────────────────────────────────────────
    @property
    def TEST_RELAX_SECURITY(self) -> bool:
        return _bool("VRM_TEST_RELAX_SECURITY")

    @property
    def TEST_ALL_OPEN(self) -> bool:
        return _bool("VRM_TEST_ALL_OPEN")

    @property
    def TEST_BYPASS_HA(self) -> bool:
        return _bool("VRM_TEST_BYPASS_HA")

    @property
    def BACKEND_ALLOW_STUB(self) -> bool:
        return _bool("VRM_BACKEND_ALLOW_STUB")

    def __repr__(self) -> str:
        active = []
        for attr in sorted(dir(self)):
            if attr.startswith("_"):
                continue
            val = getattr(self, attr)
            if val and val is not True:
                active.append(f"{attr}={val!r}")
            elif val is True:
                active.append(attr)
        return f"VRM Flags({', '.join(active) or 'defaults'})"


flags = _Flags()
