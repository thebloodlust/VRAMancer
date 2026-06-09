"""V6.E Expert Pinning — hot/cold MoE expert tiering.

Reads the profiler's expert histogram (``benchmarks/results/qwen3_coder_expert_histogram.json``),
derives a registry of "hot" expert ids per layer, and installs hooks into vLLM's
FP8 MoE method to:

1. After ``process_weights_after_loading`` finishes (kernel-shuffled layout), mirror
   *cold* expert slices to a flat staging tensor on the lender GPU (cuda:1).
   This is the load-time P2P transfer — proves the hook fires and accounts for
   the bytes living on the lender.

2. Before each ``apply()`` call, walk ``topk_ids`` to find the cold experts hit
   by the current batch, then copy them from the cuda:1 mirror back to
   ``layer.w13_weight[expert_id]`` (and ``w2``) on cuda:0 via a dedicated CUDA
   stream. The kernel runs unmodified — it sees a freshly-filled ``(num_experts,
   ...)`` tensor.

Phase B-1 PoC scope:
- The cold rows on cuda:0 are NOT zeroed after mirror — this preserves
  correctness even if staging races. The win is a counted, observable P2P data
  plane (lender ↔ compute) tied directly to MoE expert routing. Phase B-2 can
  free the cuda:0 cold rows once the staging is confirmed deterministic.
- Only the FP8 path (``Fp8MoEMethod``) is patched — AWQ MoE comes in B-3.

See ``docs/history/PLAN_V6E_EXPERT_PINNING.md`` sections A, B, C (strategy 2), D for the
design rationale.

Metrics gotcha (B-2 warmup mode)
--------------------------------
``_post_load_mirror_cold`` initialises ``resident_cold = set(cold_ids)`` —
i.e. *every* cold expert is considered already on cuda:0 right after model
load (because we never zero those rows). In ``warmup`` staging mode the apply
hook then short-circuits every cold lookup as a residency hit and emits ZERO
runtime PCIe streams.

So when you read the bench output, ``cold_experts_streamed=0`` is **NORMAL**
in warmup mode and is **NOT** evidence the pinning code is dead. The honest
metric proving the hook fires is ``cold_cache_hits`` (incremented every time
``apply()`` finds a cold expert in ``resident_cold`` and skips the stream).
Only ``stream_every`` mode produces non-zero ``cold_experts_streamed``.

Bench scripts (e.g. ``benchmarks/bench_qwen3_coder_pinning.py``) display both
``X streamed / Y cache-hits`` so the data plane is observable in either mode.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("vramancer.expert_pinning")

# ────────────────────────────────────────────────────────────────────────────
# Module state
# ────────────────────────────────────────────────────────────────────────────

_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_HOT_REGISTRY: Dict[int, frozenset] = {}
_NUM_LAYERS: int = 0
_NUM_EXPERTS: int = 0
_TOPK_PCT: float = 20.0
_LENDER_GPU: int = 1
_COMPUTE_GPU: int = 0
_LENDING_POOL = None  # core.vram_lending.VRAMLendingPool
_LENDING_LEASE = None  # VRAMLease

# B-2: staging mode controls per-call behaviour.
#   "stream_every" : B-1 legacy. Re-stream every cold-expert hit per apply().
#                    1.7 GB/token PCIe overhead. Useful as proof-of-dataplane.
#   "warmup"       : B-2 default. First hit streams from lender; subsequent
#                    hits skip (cold row on cuda:0 was never overwritten,
#                    still correct). After ~10 tokens, runtime PCIe drops to 0.
#   "mirror_only"  : Post-load mirror only. apply() hook is a stats-only no-op.
#                    Equivalent to native vLLM speed; lender holds idle mirror.
_STAGING_MODE: str = "warmup"

# Per-layer mirror state. Populated lazily by the post-load hook.
# layer_id -> {
#   "cold_ids": List[int],            # cold expert ids in this layer
#   "cold_remap": dict[int, int],     # global expert id -> row in cold tensor
#   "w13_cold": torch.Tensor,         # (num_cold, *w13_shape_per_expert) on cuda:1
#   "w2_cold": torch.Tensor,
#   "w13_scale_cold": torch.Tensor | None,
#   "w2_scale_cold": torch.Tensor | None,
#   "hot_mask": torch.Tensor,         # (num_experts,) bool on cuda:0
#   "ref": weakref to layer,
# }
_LAYER_STATE: Dict[int, Dict[str, Any]] = {}
_PREFETCH_STREAM = None  # torch.cuda.Stream on compute GPU

_STATS = {
    "layers_hooked": 0,
    "bytes_mirrored_to_lender": 0,
    "apply_calls": 0,
    "apply_calls_with_cold": 0,
    "cold_experts_streamed": 0,    # cumulative count of (layer_id, expert_id) hits
    "cold_bytes_streamed": 0,       # cumulative bytes copied lender→compute at apply
    "cold_cache_hits": 0,           # B-2: cold expert hit but already resident, skip
    "first_apply_t": 0.0,
    "last_apply_t": 0.0,
    "staging_mode": "warmup",
}


# ────────────────────────────────────────────────────────────────────────────
# Section A — Hot/cold registry
# ────────────────────────────────────────────────────────────────────────────

def compute_hot_from_histogram(
    histogram_path: str,
    topk_pct: float = 20.0,
    cache_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Derive a {layer -> hot_expert_ids} mapping from the profiler histogram.

    The profiler already stores ``hot_experts_top20pct`` for the standard 20 %
    cut. When ``topk_pct == 20`` we read that field directly; otherwise we
    re-rank from the raw histogram.

    Writes a derived ``hot_experts.json`` next to the histogram (or at
    ``cache_path``) so the worker can ``json.load`` it in one call.
    """
    hist_p = Path(histogram_path)
    with open(hist_p, "r") as f:
        d = json.load(f)

    num_layers = int(d["num_layers"])
    num_experts = int(d["num_experts_per_layer"])

    if abs(topk_pct - 20.0) < 1e-6 and "hot_experts_top20pct" in d:
        hot_per_layer = d["hot_experts_top20pct"]
        # Use the actual list length (the profiler may have rounded differently).
        topk = len(hot_per_layer[0]) if hot_per_layer else int(round(num_experts * 0.2))
    else:
        topk = max(1, int(round(num_experts * (topk_pct / 100.0))))
        histogram = d["histogram"]
        hot_per_layer = []
        for layer_idx in range(num_layers):
            counts = histogram[layer_idx]
            ranked = sorted(
                range(num_experts), key=lambda e: counts[e], reverse=True
            )
            hot_per_layer.append(sorted(ranked[:topk]))

    derived = {
        "model": d.get("model"),
        "source_histogram": str(hist_p.name),
        "topk_pct": float(topk_pct),
        "topk_per_layer": topk,
        "num_layers": num_layers,
        "num_experts_per_layer": num_experts,
        "hot_share_observed": d.get("hot_share_top20pct"),
        "hot_experts": {str(i): list(map(int, hot_per_layer[i])) for i in range(num_layers)},
    }

    out_p = Path(cache_path) if cache_path else hist_p.parent / "hot_experts.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(derived, indent=2))
    return derived


def load_hot_registry(path: str) -> Dict[int, frozenset]:
    """Load a ``hot_experts.json`` derived file into ``{layer_idx: frozenset(expert_ids)}``."""
    with open(path, "r") as f:
        d = json.load(f)
    raw = d["hot_experts"]
    return {int(k): frozenset(int(e) for e in v) for k, v in raw.items()}


# ────────────────────────────────────────────────────────────────────────────
# Section B — Hooks installation
# ────────────────────────────────────────────────────────────────────────────

def install(
    histogram_path: str,
    lending_pool: Any,
    lending_lease: Any,
    compute_gpu: int = 0,
    lender_gpu: int = 1,
    topk_pct: float = 20.0,
    cache_path: Optional[str] = None,
    staging_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Install the create_weights / apply hooks on vLLM's FP8 MoE method.

    Idempotent: a second call is a no-op (with a logged warning).

    Args:
        histogram_path: path to ``qwen3_coder_expert_histogram.json``
        lending_pool: ``core.vram_lending.VRAMLendingPool`` instance (post-register)
        lending_lease: an active ``VRAMLease`` whose ``tensor_ref`` is the flat
            staging buffer on the lender GPU.
        compute_gpu: physical GPU index for compute (logical cuda device used
            by vLLM after CVD pin — typically 0 in-process when both GPUs
            visible)
        lender_gpu: physical GPU index hosting the cold-expert staging buffer
        topk_pct: percentile to keep hot (the registry is derived against this)
        cache_path: where to write the derived ``hot_experts.json``

    Returns the parsed derived dict (for logging / metadata).
    """
    global _INSTALLED, _HOT_REGISTRY, _NUM_LAYERS, _NUM_EXPERTS
    global _COMPUTE_GPU, _LENDER_GPU, _TOPK_PCT, _LENDING_POOL, _LENDING_LEASE
    global _PREFETCH_STREAM, _STAGING_MODE

    with _INSTALL_LOCK:
        if _INSTALLED:
            logger.warning("[expert_pinning] install() already called — skipping")
            return {"already_installed": True}

        # Resolve staging mode (env var > arg > default).
        env_mode = os.environ.get("VRM_EP_STAGING_MODE", "").strip().lower()
        mode = env_mode or staging_mode or "warmup"
        if mode not in ("stream_every", "warmup", "mirror_only"):
            logger.warning(
                "[expert_pinning] unknown staging_mode=%r, falling back to 'warmup'",
                mode,
            )
            mode = "warmup"
        _STAGING_MODE = mode
        _STATS["staging_mode"] = mode

        derived = compute_hot_from_histogram(
            histogram_path, topk_pct=topk_pct, cache_path=cache_path
        )
        cache_p = Path(cache_path) if cache_path else (
            Path(histogram_path).parent / "hot_experts.json"
        )
        _HOT_REGISTRY = load_hot_registry(str(cache_p))
        _NUM_LAYERS = int(derived["num_layers"])
        _NUM_EXPERTS = int(derived["num_experts_per_layer"])
        _TOPK_PCT = float(topk_pct)
        _COMPUTE_GPU = int(compute_gpu)
        _LENDER_GPU = int(lender_gpu)
        _LENDING_POOL = lending_pool
        _LENDING_LEASE = lending_lease

        _patch_fp8_moe_method()

        try:
            import torch
            with torch.cuda.device(_COMPUTE_GPU):
                _PREFETCH_STREAM = torch.cuda.Stream(device=_COMPUTE_GPU)
        except Exception as e:
            logger.warning("[expert_pinning] could not create prefetch stream: %s", e)
            _PREFETCH_STREAM = None

        _INSTALLED = True
        logger.info(
            "[expert_pinning] installed: %d layers, %d experts/layer, "
            "topk_pct=%.0f, hot=%d/layer, compute=cuda:%d, lender=cuda:%d, "
            "staging_mode=%s",
            _NUM_LAYERS, _NUM_EXPERTS, _TOPK_PCT, derived["topk_per_layer"],
            _COMPUTE_GPU, _LENDER_GPU, _STAGING_MODE,
        )
        derived["staging_mode"] = _STAGING_MODE
        return derived


def is_installed() -> bool:
    return _INSTALLED


def get_runtime_stats() -> Dict[str, Any]:
    """Snapshot of the cumulative pinning counters (safe to call any time)."""
    s = dict(_STATS)
    s["layers_in_state"] = len(_LAYER_STATE)
    return s


# ────────────────────────────────────────────────────────────────────────────
# Internal: vLLM Fp8MoEMethod patching
# ────────────────────────────────────────────────────────────────────────────

def _patch_fp8_moe_method() -> None:
    """Wrap ``Fp8MoEMethod.process_weights_after_loading`` and ``apply`` in place."""
    try:
        from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod
    except Exception as e:
        logger.error("[expert_pinning] cannot import Fp8MoEMethod: %s", e)
        return

    original_pwal = Fp8MoEMethod.process_weights_after_loading
    original_apply = Fp8MoEMethod.apply

    def patched_pwal(self, layer):
        original_pwal(self, layer)
        try:
            _post_load_mirror_cold(layer)
        except Exception as e:
            logger.warning(
                "[expert_pinning] post-load mirror failed for layer %s: %s",
                getattr(layer, "layer_name", "?"), e, exc_info=True,
            )

    def patched_apply(self, layer, x, topk_weights, topk_ids, shared_experts_input):
        try:
            _stage_cold_for_call(layer, topk_ids)
        except Exception as e:
            logger.debug(
                "[expert_pinning] apply-time staging failed for layer %s: %s",
                getattr(layer, "layer_name", "?"), e,
            )
        return original_apply(self, layer, x, topk_weights, topk_ids, shared_experts_input)

    Fp8MoEMethod.process_weights_after_loading = patched_pwal  # type: ignore[assignment]
    Fp8MoEMethod.apply = patched_apply  # type: ignore[assignment]
    logger.info("[expert_pinning] patched Fp8MoEMethod.process_weights_after_loading + apply")


# ────────────────────────────────────────────────────────────────────────────
# Internal: post-load cold mirror to lender GPU
# ────────────────────────────────────────────────────────────────────────────

def _post_load_mirror_cold(layer) -> None:
    """Mirror cold-expert weight slices to the lender staging tensor.

    Called once per FusedMoE layer after vLLM's normal post-load processing.
    Records per-layer state in ``_LAYER_STATE`` for the apply hook to use.
    """
    import torch

    try:
        layer_id = int(layer.layer_id)
    except Exception:
        # No layer_id (probably a non-FusedMoE quant method via Fp8MoEMethod);
        # silently skip — better than crashing model load.
        return

    if layer_id in _LAYER_STATE:
        return  # idempotent

    hot_set = _HOT_REGISTRY.get(layer_id)
    if hot_set is None:
        # Layer outside the histogram coverage (e.g. > num_layers). Skip.
        return

    w13 = getattr(layer, "w13_weight", None)
    w2 = getattr(layer, "w2_weight", None)
    if w13 is None or w2 is None:
        return

    num_experts = int(w13.shape[0])
    if num_experts != _NUM_EXPERTS:
        logger.warning(
            "[expert_pinning] layer %d num_experts=%d != registry %d — skipping",
            layer_id, num_experts, _NUM_EXPERTS,
        )
        return

    cold_ids = [e for e in range(num_experts) if e not in hot_set]
    cold_remap = {e: i for i, e in enumerate(cold_ids)}

    # Allocate cold mirrors on lender GPU (one tensor per weight family).
    # We allocate fresh torch.empty on the lender device — the lending pool's
    # role here is bookkeeping (lease ACTIVE, accounting). Allocating a
    # separate flat-byte arena from the lease is the proper thing to do per
    # the plan section D, but for the B-1 PoC simplicity we let torch's
    # caching allocator on cuda:N hold these — they'll show up in NVML deltas
    # the same way.
    lender_dev = f"cuda:{_LENDER_GPU}"

    # Prefer the materialized lease staging buffer (single contiguous arena
    # on the lender GPU) — its NVML footprint is the value the bench's
    # placement guard checks. Fall back to per-tensor torch.empty if the
    # caller didn't materialize the lease.
    use_lease = (
        _LENDING_POOL is not None
        and _LENDING_LEASE is not None
        and getattr(_LENDING_LEASE, "tensor_ref", None) is not None
        and hasattr(_LENDING_POOL, "slice_for_expert")
    )

    def _alloc_cold(src_tensor, kind: str):
        per_expert_shape = tuple(src_tensor.shape[1:])
        if use_lease:
            # Allocate (num_cold, *per_expert_shape) by stacking per-expert
            # slices carved from the staging buffer. Build a Python list of
            # views; we don't need them contiguous for our copy_ pattern.
            slices: List[Any] = []
            for local in range(len(cold_ids)):
                v = _LENDING_POOL.slice_for_expert(
                    _LENDING_LEASE,
                    layer_idx=layer_id,
                    expert_local_idx=local,
                    kind=kind,
                    shape=per_expert_shape,
                    dtype=src_tensor.dtype,
                )
                if v is None:
                    return None
                slices.append(v)
            return slices  # list-of-views, not a 3D tensor
        return torch.empty(
            (len(cold_ids), *per_expert_shape),
            dtype=src_tensor.dtype,
            device=lender_dev,
        )

    w13_cold = _alloc_cold(w13, "w13")
    w2_cold = _alloc_cold(w2, "w2")

    # If lease-backed allocation ran out of room, fall back per-tensor.
    if w13_cold is None:
        w13_cold = torch.empty(
            (len(cold_ids), *tuple(w13.shape[1:])),
            dtype=w13.dtype, device=lender_dev,
        )
    if w2_cold is None:
        w2_cold = torch.empty(
            (len(cold_ids), *tuple(w2.shape[1:])),
            dtype=w2.dtype, device=lender_dev,
        )

    # Optional scales — present for FP8 MoE in vLLM 0.18.1+
    w13_scale = getattr(layer, "w13_weight_scale", None)
    if w13_scale is None:
        w13_scale = getattr(layer, "w13_weight_scale_inv", None)
    w2_scale = getattr(layer, "w2_weight_scale", None)
    if w2_scale is None:
        w2_scale = getattr(layer, "w2_weight_scale_inv", None)

    w13_scale_cold = None
    w2_scale_cold = None
    if (
        w13_scale is not None
        and isinstance(w13_scale, torch.Tensor)
        and w13_scale.dim() >= 1
        and w13_scale.shape[0] == num_experts
    ):
        w13_scale_cold = _alloc_cold(w13_scale, "w13_scale")
        if w13_scale_cold is None:
            w13_scale_cold = torch.empty(
                (len(cold_ids), *tuple(w13_scale.shape[1:])),
                dtype=w13_scale.dtype, device=lender_dev,
            )
    if (
        w2_scale is not None
        and isinstance(w2_scale, torch.Tensor)
        and w2_scale.dim() >= 1
        and w2_scale.shape[0] == num_experts
    ):
        w2_scale_cold = _alloc_cold(w2_scale, "w2_scale")
        if w2_scale_cold is None:
            w2_scale_cold = torch.empty(
                (len(cold_ids), *tuple(w2_scale.shape[1:])),
                dtype=w2_scale.dtype, device=lender_dev,
            )

    # Copy cold rows: cuda:0 → cuda:1 via PyTorch copy_ (cudaMemcpyPeerAsync
    # under the hood when can_device_access_peer is true; otherwise CUDA
    # stages via DRAM transparently). Either way, NVML on the lender will
    # report the bytes growing.
    bytes_total = 0
    for e in cold_ids:
        local = cold_remap[e]
        w13_cold[local].copy_(w13[e])
        w2_cold[local].copy_(w2[e])
        bytes_total += w13[e].numel() * w13[e].element_size()
        bytes_total += w2[e].numel() * w2[e].element_size()
        if w13_scale_cold is not None:
            w13_scale_cold[local].copy_(w13_scale[e])
            bytes_total += w13_scale[e].numel() * w13_scale[e].element_size()
        if w2_scale_cold is not None:
            w2_scale_cold[local].copy_(w2_scale[e])
            bytes_total += w2_scale[e].numel() * w2_scale[e].element_size()

    # hot_mask on compute GPU for fast topk_ids filtering.
    compute_dev = f"cuda:{_COMPUTE_GPU}"
    hot_mask = torch.zeros(num_experts, dtype=torch.bool, device=compute_dev)
    for e in hot_set:
        hot_mask[e] = True

    _LAYER_STATE[layer_id] = {
        "cold_ids": cold_ids,
        "cold_remap": cold_remap,
        "w13_cold": w13_cold,
        "w2_cold": w2_cold,
        "w13_scale_cold": w13_scale_cold,
        "w2_scale_cold": w2_scale_cold,
        "hot_mask": hot_mask,
        "num_experts": num_experts,
        # B-2: cold experts whose cuda:0 row currently holds the correct bytes.
        # After post-load mirror, every cold row on cuda:0 is correct (we never
        # zeroed it). So initialize the residency set to ALL cold ids — the
        # "warmup" mode then short-circuits every per-call stream as a hit.
        # In "stream_every" mode this set is ignored.
        "resident_cold": set(cold_ids),
    }
    _STATS["layers_hooked"] += 1
    _STATS["bytes_mirrored_to_lender"] += bytes_total

    if _STATS["layers_hooked"] <= 3 or _STATS["layers_hooked"] % 16 == 0:
        logger.info(
            "[expert_pinning] mirrored layer %d: %d cold experts → cuda:%d "
            "(%.1f MB cumulative)",
            layer_id, len(cold_ids), _LENDER_GPU,
            _STATS["bytes_mirrored_to_lender"] / 1e6,
        )


# ────────────────────────────────────────────────────────────────────────────
# Internal: per-call cold expert staging
# ────────────────────────────────────────────────────────────────────────────

def _stage_cold_for_call(layer, topk_ids) -> None:
    """Walk topk_ids → identify cold experts → ensure they're resident on cuda:0.

    Behaviour controlled by ``_STAGING_MODE``:
      - ``stream_every`` (B-1) : copy cuda:1 → cuda:0 for every cold hit.
      - ``warmup``      (B-2)  : copy only for cold experts not yet resident.
                                 After warmup the runtime PCIe traffic is 0.
      - ``mirror_only``        : stats-only no-op (matches native vLLM speed).
    """
    import torch

    _STATS["apply_calls"] += 1
    now = time.perf_counter()
    if _STATS["first_apply_t"] == 0.0:
        _STATS["first_apply_t"] = now
    _STATS["last_apply_t"] = now

    if _STAGING_MODE == "mirror_only":
        return

    try:
        layer_id = int(layer.layer_id)
    except Exception:
        return

    state = _LAYER_STATE.get(layer_id)
    if state is None:
        return

    if not isinstance(topk_ids, torch.Tensor) or topk_ids.numel() == 0:
        return

    hot_mask = state["hot_mask"]
    # Resolve cold experts hit in this batch.
    flat = topk_ids.reshape(-1)
    # Clamp out-of-range ids (some kernels use sentinel -1/num_experts for
    # padding); filter to [0, num_experts) before mask lookup.
    num_experts = state["num_experts"]
    valid = (flat >= 0) & (flat < num_experts)
    flat = flat[valid]
    if flat.numel() == 0:
        return
    is_hot = hot_mask[flat]
    cold_hits = flat[~is_hot]
    if cold_hits.numel() == 0:
        return

    unique_cold = torch.unique(cold_hits).tolist()
    if not unique_cold:
        return

    _STATS["apply_calls_with_cold"] += 1

    # B-2 warmup mode: skip cold experts already resident on cuda:0.
    if _STAGING_MODE == "warmup":
        resident = state["resident_cold"]
        skipped = sum(1 for e in unique_cold if int(e) in resident)
        unique_cold = [e for e in unique_cold if int(e) not in resident]
        if skipped:
            _STATS["cold_cache_hits"] += skipped
        if not unique_cold:
            return

    w13 = layer.w13_weight
    w2 = layer.w2_weight
    w13_cold = state["w13_cold"]
    w2_cold = state["w2_cold"]
    cold_remap = state["cold_remap"]

    bytes_streamed = 0
    cold_count = 0

    stream = _PREFETCH_STREAM
    if stream is not None:
        stream.wait_stream(torch.cuda.current_stream(_COMPUTE_GPU))
    cm = torch.cuda.stream(stream) if stream is not None else _NullCM()
    with cm:
        for e in unique_cold:
            local = cold_remap.get(int(e))
            if local is None:
                continue
            w13[e].copy_(w13_cold[local], non_blocking=True)
            w2[e].copy_(w2_cold[local], non_blocking=True)
            bytes_streamed += w13[e].numel() * w13[e].element_size()
            bytes_streamed += w2[e].numel() * w2[e].element_size()
            if state["w13_scale_cold"] is not None:
                w13_scale_attr = (
                    "w13_weight_scale" if hasattr(layer, "w13_weight_scale")
                    else "w13_weight_scale_inv"
                )
                w13_s = getattr(layer, w13_scale_attr, None)
                if w13_s is not None and w13_s.shape[0] == num_experts:
                    w13_s[e].copy_(state["w13_scale_cold"][local], non_blocking=True)
            if state["w2_scale_cold"] is not None:
                w2_scale_attr = (
                    "w2_weight_scale" if hasattr(layer, "w2_weight_scale")
                    else "w2_weight_scale_inv"
                )
                w2_s = getattr(layer, w2_scale_attr, None)
                if w2_s is not None and w2_s.shape[0] == num_experts:
                    w2_s[e].copy_(state["w2_scale_cold"][local], non_blocking=True)
            cold_count += 1
    if stream is not None:
        torch.cuda.current_stream(_COMPUTE_GPU).wait_stream(stream)

    _STATS["cold_experts_streamed"] += cold_count
    _STATS["cold_bytes_streamed"] += bytes_streamed

    # B-2: mark streamed experts as resident on cuda:0. Their rows now hold
    # the correct bytes and won't be touched until eviction (none in B-2).
    if _STAGING_MODE == "warmup" and cold_count > 0:
        resident = state["resident_cold"]
        for e in unique_cold:
            resident.add(int(e))


class _NullCM:
    """Inert context manager when a dedicated prefetch stream is unavailable."""
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


__all__ = [
    "compute_hot_from_histogram",
    "load_hot_registry",
    "install",
    "is_installed",
    "get_runtime_stats",
]
