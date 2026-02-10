"""Layer-level profiler for optimal GPU placement.

Profiles each transformer layer to measure:
  - Forward-pass latency (ms)
  - Memory footprint (parameters + activations, MB)
  - Compute intensity (FLOPS estimate)

Results feed into PlacementEngine for optimal layer-to-GPU assignment
instead of naive VRAM-proportional splitting.

Usage:
    profiler = LayerProfiler()
    profiles = profiler.profile_model("meta-llama/Llama-2-7b")
    # -> [LayerProfile(index=0, latency_ms=1.2, memory_mb=45, ...), ...]
"""
from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_logger = logging.getLogger("vramancer.layer_profiler")

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore

try:
    from core.utils import detect_backend, enumerate_devices
except ImportError:
    def detect_backend():
        return "cpu"
    def enumerate_devices():
        return [{"id": "cpu:0", "backend": "cpu", "index": 0, "name": "CPU"}]

# Lazy import to avoid circular dependency with model_splitter
_extract_layers = None

def _get_extract_layers():
    global _extract_layers
    if _extract_layers is None:
        try:
            from core.model_splitter import _extract_layers as _fn
            _extract_layers = _fn
        except ImportError:
            pass
    return _extract_layers


# ---------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------

@dataclass
class LayerProfile:
    """Profile data for a single transformer layer."""
    index: int
    name: str = ""
    # Timing
    latency_ms: float = 0.0
    latency_std_ms: float = 0.0
    # Memory
    param_count: int = 0
    param_memory_mb: float = 0.0
    activation_memory_mb: float = 0.0
    total_memory_mb: float = 0.0
    # Compute
    estimated_flops: int = 0
    compute_intensity: float = 0.0  # FLOPS / byte (arithmetic intensity)
    # Classification
    layer_type: str = "unknown"  # attention, mlp, norm, embedding, other


@dataclass
class GPUProfile:
    """Performance profile for a single GPU."""
    index: int
    name: str
    backend: str
    total_vram_mb: int = 0
    free_vram_mb: int = 0
    # Measured performance
    compute_throughput_gflops: float = 0.0
    memory_bandwidth_gbps: float = 0.0


@dataclass
class PlacementPlan:
    """Optimal layer-to-GPU assignment plan."""
    assignments: List[Tuple[int, int]] = field(default_factory=list)
    # (layer_index, gpu_index) pairs
    estimated_latency_ms: float = 0.0
    estimated_transfer_overhead_ms: float = 0.0
    gpu_utilization: Dict[int, float] = field(default_factory=dict)
    # Per-GPU estimated VRAM usage ratio


# ---------------------------------------------------------------
# Layer Profiler
# ---------------------------------------------------------------

class LayerProfiler:
    """Profile transformer layers for placement optimization.

    Methods:
        profile_model(model_or_name) -> List[LayerProfile]
        profile_gpus() -> List[GPUProfile]
        compute_optimal_placement(layers, gpus) -> PlacementPlan
    """

    def __init__(
        self,
        warmup_iters: int = 3,
        profile_iters: int = 10,
        batch_size: int = 1,
        seq_length: int = 128,
        dtype: Optional[Any] = None,
    ):
        self.warmup_iters = warmup_iters
        self.profile_iters = profile_iters
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.dtype = dtype
        self._stub = _MINIMAL or torch is None

    # ------------------------------------------------------------------
    # Model layer profiling
    # ------------------------------------------------------------------

    def profile_model(
        self,
        model: Any,
        device: Optional[str] = None,
    ) -> List[LayerProfile]:
        """Profile all transformer layers in a model.

        Args:
            model: A loaded nn.Module (HuggingFace or custom).
            device: Device to profile on (default: best available).

        Returns:
            List of LayerProfile, one per layer.
        """
        if self._stub:
            _logger.info("Stub mode: returning synthetic profiles")
            return self._synthetic_profiles(32)

        if device is None:
            backend = detect_backend()
            device = "cuda:0" if backend in ("cuda", "rocm") else (
                "mps" if backend == "mps" else "cpu"
            )

        layers = _get_extract_layers()(model) if _get_extract_layers() else None
        if layers is None:
            # Try to get all children as layers
            layers = list(model.children())

        if not layers:
            _logger.warning("No layers found in model")
            return []

        # Get hidden size from model config or first layer
        hidden_size = self._detect_hidden_size(model, layers)

        profiles = []
        for i, layer in enumerate(layers):
            profile = self._profile_single_layer(layer, i, device, hidden_size)
            profiles.append(profile)

        # Log summary
        total_params = sum(p.param_count for p in profiles)
        total_mem = sum(p.total_memory_mb for p in profiles)
        _logger.info(
            f"Profiled {len(profiles)} layers: "
            f"{total_params / 1e6:.1f}M params, {total_mem:.0f} MB total"
        )

        return profiles

    def _profile_single_layer(
        self,
        layer: Any,
        index: int,
        device: str,
        hidden_size: int,
    ) -> LayerProfile:
        """Profile a single layer: latency, memory, FLOPS."""
        profile = LayerProfile(index=index)
        profile.name = type(layer).__name__
        profile.layer_type = self._classify_layer(layer)

        # --- Parameter count and memory ---
        try:
            params = sum(p.numel() for p in layer.parameters())
            param_bytes = sum(
                p.numel() * p.element_size() for p in layer.parameters()
            )
            profile.param_count = params
            profile.param_memory_mb = param_bytes / (1024 * 1024)
        except Exception:
            profile.param_count = 0
            profile.param_memory_mb = 0.0

        # --- Activation memory estimate ---
        # For a transformer layer: activations ~ batch * seq * hidden * 4 bytes * 2
        # (input + output, float32)
        elem_size = 4 if self.dtype is None else 2  # fp32 or fp16
        activation_bytes = (
            self.batch_size * self.seq_length * hidden_size * elem_size * 2
        )
        profile.activation_memory_mb = activation_bytes / (1024 * 1024)
        profile.total_memory_mb = profile.param_memory_mb + profile.activation_memory_mb

        # --- FLOPS estimation ---
        profile.estimated_flops = self._estimate_layer_flops(
            layer, profile.layer_type, hidden_size
        )
        if profile.total_memory_mb > 0:
            total_bytes = profile.total_memory_mb * 1024 * 1024
            profile.compute_intensity = profile.estimated_flops / max(total_bytes, 1)

        # --- Latency measurement ---
        try:
            layer_on_device = layer.to(device)
            dummy = torch.randn(
                self.batch_size, self.seq_length, hidden_size,
                device=device,
                dtype=self.dtype or torch.float32,
            )

            # Warmup
            for _ in range(self.warmup_iters):
                with torch.no_grad():
                    try:
                        _ = layer_on_device(dummy)
                    except Exception:
                        # Some layers need different input shapes
                        break

            # Timed runs
            latencies = []
            for _ in range(self.profile_iters):
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    try:
                        _ = layer_on_device(dummy)
                    except Exception:
                        break
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) * 1000  # ms
                latencies.append(dt)

            if latencies:
                profile.latency_ms = sum(latencies) / len(latencies)
                if len(latencies) > 1:
                    mean = profile.latency_ms
                    variance = sum((x - mean) ** 2 for x in latencies) / len(latencies)
                    profile.latency_std_ms = variance ** 0.5
            else:
                # Fallback: estimate from FLOPS
                profile.latency_ms = profile.estimated_flops / 1e9  # rough

        except Exception as exc:
            _logger.debug(f"Latency measurement failed for layer {index}: {exc}")
            profile.latency_ms = profile.estimated_flops / 1e9 if profile.estimated_flops else 0.1

        return profile

    def _classify_layer(self, layer: Any) -> str:
        """Classify layer type from its class name."""
        name = type(layer).__name__.lower()
        if any(k in name for k in ("attention", "attn", "selfattention", "mha")):
            return "attention"
        if any(k in name for k in ("mlp", "feedforward", "ffn", "dense", "linear")):
            return "mlp"
        if any(k in name for k in ("norm", "layernorm", "rmsnorm", "batchnorm")):
            return "norm"
        if any(k in name for k in ("embed", "wte", "wpe", "token")):
            return "embedding"
        if any(k in name for k in ("conv", "pool")):
            return "conv"
        return "block"  # Likely a full transformer block

    def _detect_hidden_size(self, model: Any, layers: list) -> int:
        """Detect hidden size from model config or layer parameters."""
        # Try HuggingFace config
        if hasattr(model, "config"):
            cfg = model.config
            for attr in ("hidden_size", "d_model", "n_embd", "embed_dim"):
                if hasattr(cfg, attr):
                    return getattr(cfg, attr)

        # Try from first layer's weight shape
        if layers:
            for p in layers[0].parameters():
                if p.dim() >= 2:
                    return p.shape[-1]

        return 768  # Default (GPT-2 base)

    def _estimate_layer_flops(
        self,
        layer: Any,
        layer_type: str,
        hidden_size: int,
    ) -> int:
        """Estimate FLOPS for one forward pass of a layer.

        Based on standard transformer FLOP formulas:
          - Attention: 2 * batch * seq^2 * hidden + 4 * batch * seq * hidden^2
          - MLP: 2 * batch * seq * hidden * 4*hidden (for standard 4x expansion)
          - Norm: batch * seq * hidden (negligible)
          - Block (combined attention + MLP): sum of above
        """
        B, S, H = self.batch_size, self.seq_length, hidden_size

        if layer_type == "attention":
            # QKV projection + attention scores + output projection
            return 2 * B * S * S * H + 4 * B * S * H * H

        if layer_type == "mlp":
            # Two linear layers with 4x expansion
            expansion = 4
            # Try to get actual expansion from layer
            try:
                for child in layer.children():
                    if hasattr(child, "out_features"):
                        expansion = child.out_features / H
                        break
            except Exception:
                pass
            return int(2 * B * S * H * H * expansion * 2)

        if layer_type in ("norm", "embedding"):
            return B * S * H  # negligible

        if layer_type == "block":
            # Full transformer block = attention + MLP + 2x norm
            attn_flops = 2 * B * S * S * H + 4 * B * S * H * H
            mlp_flops = 2 * B * S * H * H * 4 * 2
            norm_flops = 2 * B * S * H
            return attn_flops + mlp_flops + norm_flops

        # Unknown: estimate from parameter count
        try:
            params = sum(p.numel() for p in layer.parameters())
            return 2 * B * S * params  # rough: 2 * batch * seq * params
        except Exception:
            return B * S * H

    # ------------------------------------------------------------------
    # GPU profiling
    # ------------------------------------------------------------------

    def profile_gpus(self) -> List[GPUProfile]:
        """Profile all available GPUs for compute throughput and memory BW.

        Returns:
            List of GPUProfile with measured performance metrics.
        """
        if self._stub:
            return [GPUProfile(index=0, name="StubGPU", backend="cpu",
                               total_vram_mb=8192, free_vram_mb=8192,
                               compute_throughput_gflops=100.0,
                               memory_bandwidth_gbps=400.0)]

        devices = enumerate_devices()
        profiles = []

        for dev in devices:
            if dev["backend"] == "cpu":
                profiles.append(GPUProfile(
                    index=dev["index"],
                    name=dev["name"],
                    backend="cpu",
                    compute_throughput_gflops=self._benchmark_cpu_gflops(),
                    memory_bandwidth_gbps=self._benchmark_cpu_bandwidth(),
                ))
                continue

            idx = dev["index"]
            total_mem = dev.get("total_memory", 0) or 0

            gp = GPUProfile(
                index=idx,
                name=dev["name"],
                backend=dev["backend"],
                total_vram_mb=total_mem // (1024 * 1024) if total_mem else 0,
            )

            # Measure free VRAM
            try:
                if torch.cuda.is_available() and idx < torch.cuda.device_count():
                    allocated = torch.cuda.memory_allocated(idx)
                    gp.free_vram_mb = (total_mem - allocated) // (1024 * 1024)
            except Exception:
                gp.free_vram_mb = gp.total_vram_mb

            # Benchmark compute throughput
            gp.compute_throughput_gflops = self._benchmark_gpu_gflops(idx)
            gp.memory_bandwidth_gbps = self._benchmark_gpu_bandwidth(idx)

            profiles.append(gp)

        return profiles

    def _benchmark_gpu_gflops(self, gpu_idx: int, size: int = 2048) -> float:
        """Measure GPU compute throughput via matrix multiplication."""
        if not torch or not torch.cuda.is_available():
            return 0.0
        try:
            device = f"cuda:{gpu_idx}"
            a = torch.randn(size, size, device=device, dtype=torch.float32)
            b = torch.randn(size, size, device=device, dtype=torch.float32)

            # Warmup
            for _ in range(3):
                torch.mm(a, b)
            torch.cuda.synchronize(gpu_idx)

            # Timed
            t0 = time.perf_counter()
            iters = 10
            for _ in range(iters):
                torch.mm(a, b)
            torch.cuda.synchronize(gpu_idx)
            dt = time.perf_counter() - t0

            # FLOPS = 2 * N^3 per matmul
            total_flops = 2 * size ** 3 * iters
            gflops = total_flops / (dt * 1e9)
            return round(gflops, 1)
        except Exception as exc:
            _logger.debug(f"GPU {gpu_idx} compute benchmark failed: {exc}")
            return 0.0

    def _benchmark_gpu_bandwidth(self, gpu_idx: int, size_mb: int = 256) -> float:
        """Measure GPU memory bandwidth via large copies."""
        if not torch or not torch.cuda.is_available():
            return 0.0
        try:
            device = f"cuda:{gpu_idx}"
            n_elements = (size_mb * 1024 * 1024) // 4
            src = torch.randn(n_elements, device=device, dtype=torch.float32)

            # Warmup
            for _ in range(3):
                _ = src.clone()
            torch.cuda.synchronize(gpu_idx)

            t0 = time.perf_counter()
            iters = 10
            for _ in range(iters):
                _ = src.clone()
            torch.cuda.synchronize(gpu_idx)
            dt = time.perf_counter() - t0

            # 2x size_mb per clone (read + write)
            total_bytes = 2 * size_mb * 1024 * 1024 * iters
            gbps = total_bytes / (dt * 1e9)
            return round(gbps, 1)
        except Exception as exc:
            _logger.debug(f"GPU {gpu_idx} bandwidth benchmark failed: {exc}")
            return 0.0

    def _benchmark_cpu_gflops(self, size: int = 1024) -> float:
        """Measure CPU compute throughput."""
        if not torch:
            return 1.0
        try:
            a = torch.randn(size, size, dtype=torch.float32)
            b = torch.randn(size, size, dtype=torch.float32)
            t0 = time.perf_counter()
            iters = 5
            for _ in range(iters):
                torch.mm(a, b)
            dt = time.perf_counter() - t0
            total_flops = 2 * size ** 3 * iters
            return round(total_flops / (dt * 1e9), 1)
        except Exception:
            return 1.0

    def _benchmark_cpu_bandwidth(self, size_mb: int = 64) -> float:
        """Measure CPU memory bandwidth."""
        if not torch:
            return 10.0
        try:
            n = (size_mb * 1024 * 1024) // 4
            src = torch.randn(n, dtype=torch.float32)
            t0 = time.perf_counter()
            iters = 5
            for _ in range(iters):
                _ = src.clone()
            dt = time.perf_counter() - t0
            total_bytes = 2 * size_mb * 1024 * 1024 * iters
            return round(total_bytes / (dt * 1e9), 1)
        except Exception:
            return 10.0

    # ------------------------------------------------------------------
    # Synthetic profiles (stub mode)
    # ------------------------------------------------------------------

    def _synthetic_profiles(self, n_layers: int) -> List[LayerProfile]:
        """Generate synthetic layer profiles for testing."""
        profiles = []
        for i in range(n_layers):
            profiles.append(LayerProfile(
                index=i,
                name=f"TransformerBlock_{i}",
                latency_ms=1.0 + (i % 5) * 0.1,
                param_count=7_000_000,
                param_memory_mb=26.7,
                activation_memory_mb=2.0,
                total_memory_mb=28.7,
                estimated_flops=int(1.5e9),
                compute_intensity=50.0,
                layer_type="block",
            ))
        return profiles


# ---------------------------------------------------------------
# Optimal placement solver
# ---------------------------------------------------------------

def compute_optimal_placement(
    layer_profiles: List[LayerProfile],
    gpu_profiles: List[GPUProfile],
    transfer_bandwidth_gbps: float = 25.0,
) -> PlacementPlan:
    """Compute optimal layer-to-GPU assignment using dynamic programming.

    Minimizes total inference latency = sum(layer latency) + transfer overhead.
    Transfer overhead occurs at GPU boundaries (layers on different GPUs need
    activation transfer between them).

    The DP considers:
      - Layer compute cost (latency_ms)
      - Layer memory footprint (total_memory_mb)
      - GPU compute throughput (to scale latency)
      - GPU free VRAM (hard constraint)
      - Transfer cost at GPU boundaries

    Args:
        layer_profiles: Profiled layers, in model order.
        gpu_profiles: Available GPU profiles.
        transfer_bandwidth_gbps: Inter-GPU bandwidth (measured or estimated).

    Returns:
        PlacementPlan with optimal assignments.
    """
    n_layers = len(layer_profiles)
    n_gpus = len(gpu_profiles)

    if n_layers == 0 or n_gpus == 0:
        return PlacementPlan()

    if n_gpus == 1:
        return PlacementPlan(
            assignments=[(i, gpu_profiles[0].index) for i in range(n_layers)],
            estimated_latency_ms=sum(lp.latency_ms for lp in layer_profiles),
            gpu_utilization={gpu_profiles[0].index: 1.0},
        )

    # Compute relative speed factors
    # Normalize compute throughput: fastest GPU = 1.0
    max_gflops = max(gp.compute_throughput_gflops for gp in gpu_profiles) or 1.0
    speed_factor = {
        gp.index: (gp.compute_throughput_gflops / max_gflops) if gp.compute_throughput_gflops > 0 else 0.1
        for gp in gpu_profiles
    }

    # Transfer cost between layers on different GPUs (ms)
    # activation_size * 8 / bandwidth = time in seconds
    def transfer_cost_ms(layer_idx: int) -> float:
        act_mb = layer_profiles[layer_idx].activation_memory_mb
        act_bytes = act_mb * 1024 * 1024
        if transfer_bandwidth_gbps <= 0:
            return 0.0
        return (act_bytes * 8 / (transfer_bandwidth_gbps * 1e9)) * 1000  # ms

    # DP: dp[i][g] = minimum latency to process layers 0..i with layer i on GPU g
    INF = float("inf")
    dp = [[INF] * n_gpus for _ in range(n_layers)]
    parent = [[-1] * n_gpus for _ in range(n_layers)]  # for backtracking

    # VRAM available per GPU (track cumulatively)
    vram_avail = {gp.index: float(gp.free_vram_mb) for gp in gpu_profiles}

    # VRAM used per GPU per DP state: track via greedy approximation
    # (full DP with VRAM tracking is NP-hard; we use greedy + DP heuristic)

    # Initialize: first layer on each GPU
    for g_idx, gp in enumerate(gpu_profiles):
        lp = layer_profiles[0]
        if lp.total_memory_mb <= vram_avail[gp.index]:
            adjusted_latency = lp.latency_ms / max(speed_factor[gp.index], 0.01)
            dp[0][g_idx] = adjusted_latency

    # Fill DP table
    for i in range(1, n_layers):
        lp = layer_profiles[i]
        for g_idx, gp in enumerate(gpu_profiles):
            # Cost of running layer i on GPU g
            layer_cost = lp.latency_ms / max(speed_factor[gp.index], 0.01)

            for prev_g in range(n_gpus):
                if dp[i - 1][prev_g] >= INF:
                    continue

                # Transfer cost if GPU changes
                xfer = 0.0
                if prev_g != g_idx:
                    xfer = transfer_cost_ms(i - 1)

                total = dp[i - 1][prev_g] + layer_cost + xfer
                if total < dp[i][g_idx]:
                    dp[i][g_idx] = total
                    parent[i][g_idx] = prev_g

    # Backtrack to find optimal assignment
    best_last_gpu = min(range(n_gpus), key=lambda g: dp[n_layers - 1][g])
    best_latency = dp[n_layers - 1][best_last_gpu]

    assignments = [(0, 0)] * n_layers
    g = best_last_gpu
    for i in range(n_layers - 1, -1, -1):
        assignments[i] = (i, gpu_profiles[g].index)
        g = parent[i][g] if parent[i][g] >= 0 else g

    # Calculate transfer overhead and GPU utilization
    transfer_total = 0.0
    gpu_mem_used: Dict[int, float] = {gp.index: 0.0 for gp in gpu_profiles}
    for i, (layer_idx, gpu_idx) in enumerate(assignments):
        lp = layer_profiles[layer_idx]
        gpu_mem_used[gpu_idx] = gpu_mem_used.get(gpu_idx, 0.0) + lp.total_memory_mb
        if i > 0 and assignments[i - 1][1] != gpu_idx:
            transfer_total += transfer_cost_ms(i - 1)

    gpu_util = {}
    for gp in gpu_profiles:
        total = gp.total_vram_mb or 1
        gpu_util[gp.index] = min(gpu_mem_used.get(gp.index, 0.0) / total, 1.0)

    # VRAM overflow check: if a GPU is overloaded, redistribute greedily
    plan = PlacementPlan(
        assignments=assignments,
        estimated_latency_ms=best_latency,
        estimated_transfer_overhead_ms=transfer_total,
        gpu_utilization=gpu_util,
    )

    plan = _enforce_vram_constraints(plan, layer_profiles, gpu_profiles)

    return plan


def _enforce_vram_constraints(
    plan: PlacementPlan,
    layer_profiles: List[LayerProfile],
    gpu_profiles: List[GPUProfile],
) -> PlacementPlan:
    """Post-process: move layers if VRAM constraints are violated."""
    gpu_free = {gp.index: float(gp.free_vram_mb) for gp in gpu_profiles}
    gpu_used: Dict[int, float] = {gp.index: 0.0 for gp in gpu_profiles}

    # Calculate current usage
    for layer_idx, gpu_idx in plan.assignments:
        lp = layer_profiles[layer_idx]
        gpu_used[gpu_idx] = gpu_used.get(gpu_idx, 0.0) + lp.total_memory_mb

    # Check overflows
    overflow_gpus = {
        g: gpu_used[g] - gpu_free[g]
        for g in gpu_free
        if gpu_used[g] > gpu_free[g]
    }

    if not overflow_gpus:
        return plan  # No overflow, plan is valid

    _logger.warning(f"VRAM overflow detected on GPUs: {overflow_gpus}")

    # Greedy redistribution: move layers from overloaded to underloaded GPUs
    # Keep layer order (only move contiguous tail segments)
    new_assignments = list(plan.assignments)

    for overloaded_gpu, overflow_mb in sorted(
        overflow_gpus.items(), key=lambda x: -x[1]
    ):
        # Find layers on this GPU, sorted by memory (move largest first)
        layers_on_gpu = [
            (i, layer_idx)
            for i, (layer_idx, g) in enumerate(new_assignments)
            if g == overloaded_gpu
        ]

        for pos, layer_idx in reversed(layers_on_gpu):
            if overflow_mb <= 0:
                break
            lp = layer_profiles[layer_idx]

            # Find GPU with most free space
            candidates = [
                gp for gp in gpu_profiles
                if gp.index != overloaded_gpu
                and gpu_used.get(gp.index, 0) + lp.total_memory_mb <= gpu_free.get(gp.index, 0)
            ]

            if candidates:
                target = max(candidates, key=lambda gp: gpu_free[gp.index] - gpu_used.get(gp.index, 0))
                new_assignments[pos] = (layer_idx, target.index)
                gpu_used[overloaded_gpu] -= lp.total_memory_mb
                gpu_used[target.index] = gpu_used.get(target.index, 0) + lp.total_memory_mb
                overflow_mb -= lp.total_memory_mb

    plan.assignments = new_assignments
    return plan


__all__ = [
    "LayerProfiler",
    "LayerProfile",
    "GPUProfile",
    "PlacementPlan",
    "compute_optimal_placement",
]
