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
    def CLUSTER_AUTO_DISCOVER(self) -> bool:
        """Auto-start cluster discovery (UDP broadcast) at PipelineRegistry init."""
        return _bool("VRM_CLUSTER_AUTO_DISCOVER")

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
    def FEATURE_AITP(self) -> bool:
        """Opt-in AITP/VTP network stack (opens listening sockets at start_vtp_server)."""
        return _bool("VRM_FEATURE_AITP")

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


# ---------------------------------------------------------------------------
# Registry of every VRM_* env var read across the codebase.
#
# Keep alphabetical. The set is the single source of truth for:
#   - dump_flags()      → Markdown documentation generator
#   - unknown_env_flags() → CI helper to catch typos / undocumented flags
#
# When you add a new ``VRM_*`` env var anywhere in core/, also add it here
# (and ideally a typed property on _Flags above).
# ---------------------------------------------------------------------------

_KNOWN_FLAGS: dict[str, tuple[str, str]] = {
    # name : (category, description)

    # ---- modes / test -----------------------------------------------------
    "VRM_MINIMAL_TEST":         ("test", "Stub-mode for CI without GPU/torch."),
    "VRM_TEST_MODE":            ("test", "Generic test mode, relaxes runtime checks."),
    "VRM_TEST_RELAX_SECURITY":  ("test", "Skip strict security validators in tests."),
    "VRM_TEST_ALL_OPEN":        ("test", "Auth-bypass for ALL endpoints (tests only)."),
    "VRM_TEST_BYPASS_HA":       ("test", "Skip HA peer sync inside tests."),
    "VRM_DISABLE_RATE_LIMIT":   ("test", "Disable rate limiting (CI/dev)."),
    "VRM_BACKEND_ALLOW_STUB":   ("test", "Return stub when backend dep missing."),
    "VRM_STRICT_IMPORT":        ("test", "Crash on missing optional dep instead of degrade."),
    "VRM_PRODUCTION":           ("mode", "Production hardening (token+secret required)."),
    "VRM_READ_ONLY":            ("mode", "Block all mutating endpoints."),
    "VRM_DEBUG":                ("mode", "Generic debug toggle."),

    # ---- security ---------------------------------------------------------
    "VRM_API_TOKEN":            ("security", "Bearer token required by API auth."),
    "VRM_AUTH_SECRET":          ("security", "JWT signing secret (production-required)."),
    "VRM_AUTH_EXP":             ("security", "JWT access-token lifetime (s)."),
    "VRM_AUTH_REFRESH_EXP":     ("security", "JWT refresh-token lifetime (s)."),
    "VRM_DEFAULT_ADMIN_PASS":   ("security", "Override dev default admin password."),
    "VRM_DISABLE_SECRET_ROTATION": ("security", "Disable periodic JWT secret rotation."),
    "VRM_RATE_MAX":             ("security", "Per-IP+path rate limit per minute."),
    "VRM_CORS_ORIGINS":         ("security", "Allowed CORS origins (CSV)."),
    "VRM_TRANSPORT_INSECURE":   ("security", "Allow plaintext transport (dev only)."),
    "VRM_HA_SECRET":            ("security", "HA cluster shared secret."),
    "VRM_CLUSTER_SECRET":       ("security", "Cluster discovery shared secret."),

    # ---- API / server -----------------------------------------------------
    "VRM_API_HOST":             ("api", "Flask bind host."),
    "VRM_API_PORT":             ("api", "Flask bind port."),
    "VRM_API_BASE":             ("api", "Public base URL (clients)."),
    "VRM_API_DEBUG":            ("api", "Flask debug flag."),
    "VRM_WORKERS":              ("api", "Gunicorn worker count."),
    "VRM_THREADS":              ("api", "Gunicorn threads/worker."),
    "VRM_GUNICORN_TIMEOUT":     ("api", "Gunicorn worker timeout (s)."),
    "VRM_GENERATE_TIMEOUT":     ("api", "Continuous-batcher request timeout (s)."),
    "VRM_INFERENCE_TIMEOUT":    ("api", "Generic per-inference timeout (s)."),
    "VRM_SSE_TIMEOUT":          ("api", "SSE keepalive timeout (s)."),
    "VRM_MAX_PROMPT_LENGTH":    ("api", "Max prompt size (chars)."),
    "VRM_MAX_BODY":             ("api", "Max request body (bytes)."),
    "VRM_MAX_BATCH_SIZE":       ("api", "Continuous-batcher waiting queue cap."),
    "VRM_MAX_CONCURRENT":       ("api", "Concurrent in-flight cap."),
    "VRM_MAX_QUEUE_SIZE":       ("api", "Request queue cap (backpressure)."),
    "VRM_DASHBOARD_MINIMAL":    ("api", "Minimal dashboard renderer."),
    "VRM_DISABLE_SOCKETIO":     ("api", "Disable Socket.IO heartbeat."),
    "VRM_CB_FAILURE_THRESHOLD": ("api", "Circuit breaker fail count → open."),
    "VRM_CB_RECOVERY_TIMEOUT":  ("api", "Circuit breaker HALF_OPEN interval (s)."),
    "VRM_PYNVML_TIMEOUT":       ("api", "pynvml call timeout (s)."),
    "VRM_SHARED_QUEUE":         ("api", "Shared queue backend URL."),

    # ---- backend / model --------------------------------------------------
    "VRM_BACKEND":              ("backend", "Forced backend (huggingface|vllm|llama_cpp|...)."),
    "VRM_MODEL":                ("backend", "Default model id."),
    "VRM_QUANTIZATION":         ("backend", "Quantization mode (nvfp4|nf4|int8|gptq|awq|empty)."),
    "VRM_TRUST_REMOTE_CODE":    ("backend", "Pass trust_remote_code=True to HF."),
    "VRM_DRAFT_MODEL":          ("backend", "Speculative decoding draft model id."),
    "VRM_FORCE_BASIC_TOKENIZER": ("backend", "Use BasicTokenizer fallback."),
    "VRM_TOKENIZER_WORKERS":    ("backend", "Tokenizer thread pool size."),
    "VRM_DISABLE_ONNX":         ("backend", "Disable ONNX export pathway."),
    "VRM_DISABLE_TURBO":        ("backend", "Disable TurboEngine CUDA Graph decode."),
    "VRM_TURBO_MAX_SEQ":        ("backend", "TurboEngine static KV max seq."),
    "VRM_TURBOQUANT_WORKER":    ("backend", "Run TurboQuant on GPU dispatch worker."),
    "VRM_FORCE_MULTI_GPU":      ("backend", "Force multi-GPU split."),
    "VRM_GPU_ORDER":            ("backend", "Mirror of CUDA_DEVICE_ORDER."),
    "VRM_VLLM_TARGET_GPU":      ("backend", "vLLM compute device when lending."),
    "VRM_HETERO_STRATEGY":      ("backend", "Placement: profiled|vram|balanced."),
    "VRM_PARALLEL_MODE":        ("backend", "pp (pipeline) | tp (tensor)."),
    "VRM_SPLIT_RATIOS":         ("backend", "Manual VRAM split ratios (CSV)."),
    "VRM_PREFILL_CHUNK":        ("backend", "Chunked prefill chunk size."),
    "VRM_CONTINUOUS_BATCHING":  ("backend", "Enable continuous batcher."),
    "VRM_LLAMA_SERVER_PORT":    ("backend", "llama.cpp server port."),
    "VRM_CUDA_GRAPH":           ("backend", "Persistent CUDA Graph decode."),
    "VRM_CUDA_GRAPH_CACHE":     ("backend", "CUDA Graph cache count or path."),
    "VRM_CUDA_GRAPH_WARMUP":    ("backend", "Warmup iters before graph capture."),
    "VRM_SPEC_GAMMA":           ("backend", "Speculative draft length γ."),
    "VRM_SPEC_WINDOW":          ("backend", "Speculative rolling window."),
    "VRM_SPEC_ADAPTIVE":        ("backend", "Adaptive γ on accept rate."),
    "VRM_DEBUG_SAMPLING":       ("backend", "Per-branch sampling counters."),

    # ---- transfer ---------------------------------------------------------
    "VRM_TRANSFER_METHOD":      ("transfer", "Force strategy (0..4|auto)."),
    "VRM_TRANSFER_P2P":         ("transfer", "Allow P2P CUDA transfers."),
    "VRM_TRANSFER_OVERLAP":     ("transfer", "CUDA stream overlap for P2P."),
    "VRM_TRANSFER_ASYNC":       ("transfer", "Async double-buffered cross-vendor."),
    "VRM_FASTPATH_IF":          ("transfer", "Fastpath network interface."),
    "VRM_FASTPATH_BENCH_TTL":   ("transfer", "Fastpath bench cache TTL (s)."),
    "VRM_TRANSPORT_TIMEOUT":    ("transfer", "Generic transport timeout (s)."),

    # ---- KV / compression -------------------------------------------------
    "VRM_KV_COMPRESSION":       ("kv", "KV cache codec (turboquant|fp8)."),
    "VRM_KV_COMPRESSION_BITS":  ("kv", "Bits per polar angle."),
    "VRM_SPARSE_V_RATIO":       ("kv", "Top-k fraction for V decompress."),
    "VRM_KV_CACHE_RESIDUAL":    ("kv", "Residual layer count for reconstruction."),
    "VRM_KV_DRAM_LIMIT_GB":     ("kv", "DRAM cap for KV offload (GB)."),
    "VRM_KV_LEND":              ("kv", "Allow KV cache to use lending pool."),
    "VRM_KV_LEND_ATTENTION":    ("kv", "Lend KV during attention compute."),
    "VRM_KV_OFFLOAD_ENGRAM":    ("kv", "Offload cold KV pages to NVMe."),

    # ---- VRAM lending -----------------------------------------------------
    "VRM_VRAM_LENDING":         ("lending", "Enable lending pool."),
    "VRM_LEND_RATIO":           ("lending", "Max lendable free-VRAM ratio."),
    "VRM_LEND_RATIO_GPU":       ("lending", "Per-GPU override (idx:ratio CSV)."),
    "VRM_RECLAIM_THRESHOLD":    ("lending", "Util triggering reclaim."),
    "VRM_LENDING_INTERVAL":     ("lending", "Lending monitor poll (s)."),
    "VRM_REBALANCE_INTERVAL":   ("lending", "Block rebalancer interval (s)."),
    "VRM_EP_STAGING_MODE":      ("lending", "Expert pinning: stream_every|warmup|mirror_only."),

    # ---- cluster / HA -----------------------------------------------------
    "VRM_NODE_ID":              ("cluster", "Override node identifier."),
    "VRM_CLUSTER_AUTO_DISCOVER": ("cluster", "Auto-start mDNS+UDP discovery."),
    "VRM_CLUSTER_L1_THRESHOLD": ("cluster", "L1 cluster threshold."),
    "VRM_PEER_IPS":             ("cluster", "Static peer IPs (CSV)."),
    "VRM_SAME_RACK_NODES":      ("cluster", "Same-rack node IDs (CSV)."),
    "VRM_HA_PEERS":             ("cluster", "HA peer URLs (CSV)."),
    "VRM_HA_JOURNAL":           ("cluster", "HA journal path."),
    "VRM_HA_JOURNAL_MAX":       ("cluster", "HA journal max entries."),
    "VRM_DDNS_HOST":            ("cluster", "Dynamic DNS hostname."),
    "VRM_MEMBERSHIP_LOG":       ("cluster", "Cluster membership JSONL."),
    "VRM_MC_ADDR":              ("cluster", "Cluster multicast IPv4."),
    "VRM_MC_PORT":              ("cluster", "Cluster multicast UDP port."),
    "VRM_EDGE_MAX_AGE":         ("cluster", "Edge node TTL (s)."),
    "VRM_SENSING_HEARTBEAT":    ("cluster", "AITP sensing heartbeat (s)."),
    "VRM_SENSING_PEER_TTL":     ("cluster", "Peer TTL before evict (s)."),
    "VRM_WOI_MAC_":             ("cluster", "Wake-on-LAN MAC env prefix."),

    # ---- AITP -------------------------------------------------------------
    "VRM_FEATURE_AITP":         ("aitp", "Enable AITP networking stack."),
    "VRM_AITP_PORT":            ("aitp", "AITP UDP port."),
    "VRM_AITP_MAX_QUEUE":       ("aitp", "AITP recv queue size."),
    "VRM_AITP_STAGING_MB":      ("aitp", "AITP staging buffer (MB)."),
    "VRM_ANYCAST_GROUP":        ("aitp", "IPv6 anycast multicast group."),
    "VRM_ANYCAST_STRATEGY":     ("aitp", "weighted|least_latency|round_robin."),
    "VRM_ANYCAST_MIN_STRENGTH": ("aitp", "Min synapse strength for routing."),
    "VRM_RAID_DATA_SHARDS":     ("aitp", "RAID-RS data shards."),
    "VRM_RAID_PARITY_SHARDS":   ("aitp", "RAID-RS parity shards."),
    "VRM_RAID_PARALLEL":        ("aitp", "Parallel shard send."),
    "VRM_RAID_TIMEOUT":         ("aitp", "RAID send timeout (s)."),

    # ---- VTP --------------------------------------------------------------
    "VRM_VTP_ENABLED":          ("vtp", "Enable VRAMancer Tensor Protocol."),
    "VRM_VTP_PORT":             ("vtp", "VTP TCP port."),
    "VRM_VTP_WORKER_PORT":      ("vtp", "VTP worker port."),
    "VRM_VTP_CHUNK_MB":         ("vtp", "VTP chunk size (MB)."),
    "VRM_VTP_CREDITS":          ("vtp", "VTP flow control credits."),
    "VRM_VTP_MAX_INFLIGHT":     ("vtp", "VTP max in-flight tensors."),

    # ---- WebGPU (experimental) -------------------------------------------
    "VRM_WEBGPU_HTTP_PORT":     ("webgpu", "WebGPU dashboard HTTP port."),
    "VRM_WEBGPU_WS_PORT":       ("webgpu", "WebGPU WebSocket port."),
    "VRM_WEBGPU_NO_SSL":        ("webgpu", "Disable SSL on WebGPU WS."),
    "VRM_WEBGPU_TIMEOUT":       ("webgpu", "WebGPU task timeout (s)."),

    # ---- observability ----------------------------------------------------
    "VRM_LOG_JSON":             ("observability", "Structured JSON logs."),
    "VRM_LOG_FILE":             ("observability", "Log file path (empty = stderr)."),
    "VRM_TRACING":              ("observability", "Enable OpenTelemetry."),
    "VRM_TRACING_ATTRS":        ("observability", "Extra OTEL resource attrs."),
    "VRM_METRICS_BIND":         ("observability", "Prometheus bind address."),
    "VRM_METRICS_PORT":         ("observability", "Prometheus port."),

    # ---- persistence / data ----------------------------------------------
    "VRM_SQLITE_PATH":          ("misc", "SQLite persistence path."),
    "VRM_DATA_DIR":             ("misc", "VRAMancer data directory."),
    "VRM_CACHE_DIR":            ("misc", "Model/cache directory."),
    "VRM_AUTOSAVE_INTERVAL":    ("misc", "Autosave interval (s)."),
    "VRM_AUTOSAVE_MEMORY":      ("misc", "Hierarchical memory autosave."),
}


_CATEGORY_ORDER = [
    "mode", "test", "security", "api", "backend", "transfer",
    "kv", "lending", "cluster", "aitp", "vtp", "webgpu",
    "observability", "misc",
]


def known_flag(name: str) -> bool:
    """Return True if ``name`` is registered in :data:`_KNOWN_FLAGS`."""
    return name in _KNOWN_FLAGS


def unknown_env_flags() -> list[str]:
    """Return ``VRM_*`` env vars that are set in the process but NOT registered.

    Useful in CI to detect typos / undocumented flags. Empty list = all good.
    """
    return sorted(
        n for n in os.environ
        if n.startswith("VRM_") and n not in _KNOWN_FLAGS
        and not any(n.startswith(p) for p in ("VRM_WOI_MAC_",))
    )


def dump_flags() -> str:
    """Return a Markdown table grouping every registered flag by category."""
    by_cat: dict[str, list[tuple[str, str]]] = {}
    for name, (cat, desc) in _KNOWN_FLAGS.items():
        by_cat.setdefault(cat, []).append((name, desc))

    lines = ["# VRAMancer environment flags",
             "",
             f"_{len(_KNOWN_FLAGS)} flags registered._",
             ""]
    seen_cats = set(by_cat.keys())
    ordered = [c for c in _CATEGORY_ORDER if c in seen_cats]
    ordered += sorted(seen_cats - set(_CATEGORY_ORDER))
    for cat in ordered:
        rows = sorted(by_cat[cat])
        lines.append(f"## {cat}")
        lines.append("")
        lines.append("| Flag | Description |")
        lines.append("|------|-------------|")
        for name, desc in rows:
            lines.append(f"| `{name}` | {desc.replace('|', '\\|')} |")
        lines.append("")
    return "\n".join(lines)


def dump_active() -> dict[str, str]:
    """Return registered ``VRM_*`` env vars whose value is currently set."""
    return {n: os.environ[n] for n in _KNOWN_FLAGS if n in os.environ}


__all__ = [
    "flags", "_Flags",
    "known_flag", "unknown_env_flags", "dump_flags", "dump_active",
]


if __name__ == "__main__":  # pragma: no cover - manual CLI use
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "active":
        import json
        print(json.dumps(dump_active(), indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "unknown":
        for n in unknown_env_flags():
            print(n)
    else:
        print(dump_flags())
