"""VRAMancer Heterogeneous GPU Auto-Configuration.

Automatically detects mixed GPU setups (e.g., RTX 3090 + RTX 5070 Ti)
and computes optimal configuration for:
  - Model split ratios (VRAM-proportional with compute weighting)
  - Tier assignment (tier-0 = primary inference, tier-1 = KV cache overflow)
  - VRAM lending policies (per-GPU max lend/reclaim thresholds)
  - PCIe topology and transfer strategy
  - Continuous batching parameters

Usage:
    from core.hetero_config import auto_configure, HeteroConfig

    config = auto_configure()
    print(config.summary())

    # Apply to pipeline
    config.apply_to_pipeline(pipeline)
"""

from __future__ import annotations

import os
import math
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    from core.logger import LoggerAdapter
    _log = LoggerAdapter("hetero_config")
except Exception:
    _log = logging.getLogger("vramancer.hetero_config")

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    import torch
    _TORCH = True
except ImportError:
    torch = None  # type: ignore
    _TORCH = False


# ═══════════════════════════════════════════════════════════════════════════
# GPU Profile Database
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GPUProfile:
    """Static performance profile for a GPU model."""
    name: str
    architecture: str           # Ampere, Ada, Blackwell, RDNA3...
    vram_gb: float
    memory_bandwidth_gbps: float
    fp16_tflops: float
    pcie_gen: int               # 3, 4, 5
    pcie_lanes: int             # typically 16
    compute_capability: Tuple[int, int] = (0, 0)
    is_consumer: bool = True
    tdp_watts: int = 0

    @property
    def pcie_bandwidth_gbps(self) -> float:
        """Max theoretical PCIe bandwidth (GB/s, unidirectional)."""
        per_lane = {3: 0.985, 4: 1.969, 5: 3.938}
        return per_lane.get(self.pcie_gen, 1.0) * self.pcie_lanes

    @property
    def compute_score(self) -> float:
        """Normalized compute score (higher = faster for inference)."""
        return self.fp16_tflops * 1.0

    @property
    def memory_score(self) -> float:
        """Normalized memory score (higher = better for KV cache)."""
        return self.vram_gb * 1.0


# Known GPU profiles (expandable)
_GPU_DB: Dict[str, GPUProfile] = {
    # NVIDIA Blackwell (consumer)
    "rtx 5090": GPUProfile("RTX 5090", "Blackwell", 32.0, 1792, 209.0, 5, 16, (12, 0), True, 575),
    "rtx 5080": GPUProfile("RTX 5080", "Blackwell", 16.0, 960, 112.0, 5, 16, (12, 0), True, 360),
    "rtx 5070 ti": GPUProfile("RTX 5070 Ti", "Blackwell", 16.0, 896, 94.6, 5, 16, (12, 0), True, 300),
    "rtx 5070": GPUProfile("RTX 5070", "Blackwell", 12.0, 672, 62.4, 5, 16, (12, 0), True, 250),

    # NVIDIA Ada Lovelace
    "rtx 4090": GPUProfile("RTX 4090", "Ada", 24.0, 1008, 165.0, 4, 16, (8, 9), True, 450),
    "rtx 4080": GPUProfile("RTX 4080", "Ada", 16.0, 717, 97.5, 4, 16, (8, 9), True, 320),
    "rtx 4070 ti": GPUProfile("RTX 4070 Ti", "Ada", 12.0, 504, 73.4, 4, 16, (8, 9), True, 285),

    # NVIDIA Ampere
    "rtx 3090": GPUProfile("RTX 3090", "Ampere", 24.0, 936, 71.0, 4, 16, (8, 6), True, 350),
    "rtx 3090 ti": GPUProfile("RTX 3090 Ti", "Ampere", 24.0, 1008, 80.0, 4, 16, (8, 6), True, 450),
    "rtx 3080": GPUProfile("RTX 3080", "Ampere", 10.0, 760, 59.5, 4, 16, (8, 6), True, 320),
    "rtx 3080 ti": GPUProfile("RTX 3080 Ti", "Ampere", 12.0, 912, 68.0, 4, 16, (8, 6), True, 350),
    "rtx 3070": GPUProfile("RTX 3070", "Ampere", 8.0, 448, 40.6, 4, 16, (8, 6), True, 220),

    # NVIDIA Datacenter
    "a100": GPUProfile("A100", "Ampere", 80.0, 2039, 312.0, 4, 16, (8, 0), False, 400),
    "h100": GPUProfile("H100", "Hopper", 80.0, 3350, 756.0, 5, 16, (9, 0), False, 700),

    # AMD RDNA 3
    "rx 7900 xtx": GPUProfile("RX 7900 XTX", "RDNA3", 24.0, 960, 122.8, 4, 16, (0, 0), True, 355),
    "rx 7900 xt": GPUProfile("RX 7900 XT", "RDNA3", 20.0, 800, 103.0, 4, 16, (0, 0), True, 315),
    "rx 7800 xt": GPUProfile("RX 7800 XT", "RDNA3", 16.0, 624, 74.6, 4, 16, (0, 0), True, 263),
}


def lookup_gpu_profile(device_name: str) -> Optional[GPUProfile]:
    """Look up a GPU profile by device name (fuzzy match)."""
    name_lower = device_name.lower().strip()
    # Exact match first
    for key, profile in _GPU_DB.items():
        if key in name_lower:
            return profile
    # Partial match
    for key, profile in _GPU_DB.items():
        if any(word in name_lower for word in key.split()):
            if any(num in name_lower for num in [str(n) for n in range(3000, 10000, 10)]):
                return profile
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Detected GPU Info
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DetectedGPU:
    """Runtime-detected GPU with static + dynamic properties."""
    index: int
    name: str
    total_vram_bytes: int
    free_vram_bytes: int
    profile: Optional[GPUProfile] = None
    vendor: str = "nvidia"
    compute_capability: Tuple[int, int] = (0, 0)

    # Assigned during configuration
    tier: int = -1                  # 0 = primary, 1 = secondary, 2 = overflow
    split_ratio: float = 0.0       # fraction of model layers
    max_lend_ratio: float = 0.5    # max VRAM to lend
    reclaim_threshold: float = 0.8 # reclaim when utilization > this

    @property
    def total_vram_gb(self) -> float:
        return self.total_vram_bytes / (1024 ** 3)

    @property
    def free_vram_gb(self) -> float:
        return self.free_vram_bytes / (1024 ** 3)

    @property
    def effective_compute(self) -> float:
        """Effective compute score for inference weighting."""
        if self.profile:
            return self.profile.compute_score
        # Fallback: estimate from compute capability
        cc = self.compute_capability
        base = cc[0] * 10 + cc[1]
        return base * 1.0  # rough proxy


# ═══════════════════════════════════════════════════════════════════════════
# Heterogeneous Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HeteroConfig:
    """Complete configuration for a heterogeneous multi-GPU setup."""

    gpus: List[DetectedGPU] = field(default_factory=list)

    # Global settings
    total_vram_gb: float = 0.0
    total_pool_gb: float = 0.0    # VRAM available in lending pool
    model_max_gb: float = 0.0     # Max model that fits in pool

    # Split strategy
    split_ratios: Dict[int, float] = field(default_factory=dict)
    split_strategy: str = "vram_weighted"  # vram_weighted, compute_weighted, balanced

    # Transfer strategy
    p2p_capable: bool = False
    transfer_method: str = "cpu_staged"  # p2p, cpu_staged, cross_vendor

    # Recommended batch settings
    recommended_batch_size: int = 1
    recommended_max_tokens: int = 2048

    def summary(self) -> str:
        """Human-readable summary of the configuration."""
        lines = [
            "",
            "=" * 60,
            "  VRAMancer — Heterogeneous GPU Configuration",
            "=" * 60,
        ]
        for gpu in self.gpus:
            arch = gpu.profile.architecture if gpu.profile else "?"
            fp16 = f"{gpu.profile.fp16_tflops:.1f}" if gpu.profile else "?"
            pcie = f"Gen{gpu.profile.pcie_gen}" if gpu.profile else "?"
            lines.append(
                f"  GPU {gpu.index}: {gpu.name}"
            )
            lines.append(
                f"    VRAM: {gpu.total_vram_gb:.1f} GB (free: {gpu.free_vram_gb:.1f} GB)"
            )
            lines.append(
                f"    Arch: {arch}, FP16: {fp16} TFLOPS, PCIe: {pcie}"
            )
            lines.append(
                f"    Role: Tier-{gpu.tier}, Split: {gpu.split_ratio:.1%}"
            )
            lines.append(
                f"    Lending: max {gpu.max_lend_ratio:.0%}, reclaim @ {gpu.reclaim_threshold:.0%}"
            )
        lines.extend([
            "-" * 60,
            f"  Total VRAM:           {self.total_vram_gb:.1f} GB",
            f"  Cooperative Pool:     {self.total_pool_gb:.1f} GB",
            f"  Max model size:       ~{self.model_max_gb:.1f} GB",
            f"  Split strategy:       {self.split_strategy}",
            f"  Transfer:             {self.transfer_method}",
            f"  P2P capable:          {self.p2p_capable}",
            f"  Recommended batch:    {self.recommended_batch_size}",
            f"  Recommended max_tok:  {self.recommended_max_tokens}",
            "=" * 60,
            "",
        ])
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export as dict (for JSON/YAML serialization)."""
        return {
            "gpus": [
                {
                    "index": g.index,
                    "name": g.name,
                    "vram_gb": round(g.total_vram_gb, 1),
                    "free_gb": round(g.free_vram_gb, 1),
                    "tier": g.tier,
                    "split_ratio": round(g.split_ratio, 4),
                    "max_lend_ratio": g.max_lend_ratio,
                    "reclaim_threshold": g.reclaim_threshold,
                    "architecture": g.profile.architecture if g.profile else "unknown",
                    "fp16_tflops": g.profile.fp16_tflops if g.profile else 0,
                    "pcie_gen": g.profile.pcie_gen if g.profile else 0,
                }
                for g in self.gpus
            ],
            "total_vram_gb": round(self.total_vram_gb, 1),
            "cooperative_pool_gb": round(self.total_pool_gb, 1),
            "model_max_gb": round(self.model_max_gb, 1),
            "split_strategy": self.split_strategy,
            "split_ratios": {str(k): round(v, 4) for k, v in self.split_ratios.items()},
            "transfer_method": self.transfer_method,
            "p2p_capable": self.p2p_capable,
            "recommended_batch_size": self.recommended_batch_size,
            "recommended_max_tokens": self.recommended_max_tokens,
        }

    def apply_to_pipeline(self, pipeline: Any) -> None:
        """Apply this configuration to an InferencePipeline.

        Sets environment variables and attributes that the pipeline
        reads during load().
        """
        # Set split ratios as env var (pipeline reads this)
        ratios_str = ",".join(
            f"{r:.4f}" for r in self.split_ratios.values()
        )
        os.environ["VRM_SPLIT_RATIOS"] = ratios_str

        # Set lending parameters
        for gpu in self.gpus:
            os.environ[f"VRM_LEND_RATIO_GPU{gpu.index}"] = str(gpu.max_lend_ratio)

        # Set batch parameters
        os.environ["VRM_MAX_BATCH_SIZE"] = str(self.recommended_batch_size)

        _log.info("Applied hetero config: %s", ratios_str)


# ═══════════════════════════════════════════════════════════════════════════
# Detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_gpus() -> List[DetectedGPU]:
    """Detect all GPUs and match them against the profile database."""
    gpus: List[DetectedGPU] = []

    if not _TORCH or _MINIMAL:
        _log.debug("No torch or minimal mode — returning empty GPU list")
        return gpus

    if not torch.cuda.is_available():
        _log.debug("CUDA not available")
        return gpus

    for i in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)
            total = props.total_mem
            try:
                free = total - torch.cuda.memory_allocated(i)
            except Exception:
                free = total

            name = props.name
            cc = (props.major, props.minor)
            profile = lookup_gpu_profile(name)

            # Detect vendor
            vendor = "nvidia"
            name_lower = name.lower()
            if "amd" in name_lower or "radeon" in name_lower:
                vendor = "amd"

            gpu = DetectedGPU(
                index=i,
                name=name,
                total_vram_bytes=total,
                free_vram_bytes=free,
                profile=profile,
                vendor=vendor,
                compute_capability=cc,
            )

            # Override profile CC if detected
            if profile and cc != (0, 0):
                profile.compute_capability = cc

            gpus.append(gpu)
            _log.info(
                "Detected GPU %d: %s (%.1f GB, CC %d.%d, profile=%s)",
                i, name, gpu.total_vram_gb, cc[0], cc[1],
                profile.architecture if profile else "unknown",
            )
        except Exception as e:
            _log.warning("Failed to detect GPU %d: %s", i, e)

    return gpus


def probe_p2p(gpu_a: int, gpu_b: int) -> bool:
    """Probe if P2P (peer-to-peer) DMA is possible between two GPUs."""
    if not _TORCH or _MINIMAL:
        return False
    try:
        return torch.cuda.can_device_access_peer(gpu_a, gpu_b)
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Configuration Engine
# ═══════════════════════════════════════════════════════════════════════════

def auto_configure(
    strategy: str = "balanced",
    model_size_gb: Optional[float] = None,
) -> HeteroConfig:
    """Auto-detect GPUs and compute optimal heterogeneous configuration.

    Parameters
    ----------
    strategy : str
        Split strategy:
        - "vram_weighted": Split proportional to free VRAM (simple)
        - "compute_weighted": Split proportional to FP16 TFLOPS (compute-bound)
        - "balanced": Weighted combination of VRAM + compute (recommended)
    model_size_gb : float, optional
        Expected model size. Used to compute if model fits and optimal split.

    Returns
    -------
    HeteroConfig
        Complete configuration ready to apply.
    """
    gpus = detect_gpus()
    config = HeteroConfig(gpus=gpus, split_strategy=strategy)

    if not gpus:
        _log.info("No GPUs detected — returning CPU-only config")
        return config

    # ── Total VRAM ──
    config.total_vram_gb = sum(g.total_vram_gb for g in gpus)

    # ── Tier assignment ──
    #   Tier-0: GPU with highest compute score (primary inference)
    #   Tier-1+: Others, sorted by VRAM (secondary / KV cache overflow)
    gpus_sorted_by_compute = sorted(
        gpus, key=lambda g: g.effective_compute, reverse=True
    )
    for rank, gpu in enumerate(gpus_sorted_by_compute):
        gpu.tier = rank

    # ── Split ratios ──
    if len(gpus) == 1:
        gpus[0].split_ratio = 1.0
        config.split_ratios = {gpus[0].index: 1.0}
    else:
        total_weight = 0.0
        weights: Dict[int, float] = {}

        for gpu in gpus:
            if strategy == "vram_weighted":
                w = gpu.free_vram_gb
            elif strategy == "compute_weighted":
                w = gpu.effective_compute
            else:
                # Balanced: sqrt(VRAM * compute) — geometric mean
                vram_score = gpu.free_vram_gb
                compute_score = gpu.effective_compute
                # Normalize compute score relative to other GPUs
                max_compute = max(g.effective_compute for g in gpus)
                if max_compute > 0:
                    compute_norm = compute_score / max_compute
                else:
                    compute_norm = 1.0
                # Weight: VRAM (primary constraint) * sqrt(compute efficiency)
                w = vram_score * math.sqrt(compute_norm)

            weights[gpu.index] = w
            total_weight += w

        for gpu in gpus:
            ratio = weights[gpu.index] / total_weight if total_weight > 0 else 1.0 / len(gpus)
            gpu.split_ratio = ratio
            config.split_ratios[gpu.index] = ratio

    # ── Lending policy per GPU ──
    for gpu in gpus:
        if gpu.tier == 0:
            # Primary GPU: lend less, reclaim early
            gpu.max_lend_ratio = 0.3
            gpu.reclaim_threshold = 0.75
        elif gpu.tier == 1:
            # Secondary GPU: lend more aggressively
            gpu.max_lend_ratio = 0.7
            gpu.reclaim_threshold = 0.85
        else:
            # Overflow GPU: maximum lending
            gpu.max_lend_ratio = 0.8
            gpu.reclaim_threshold = 0.90

    # ── Transfer strategy ──
    config.p2p_capable = False
    if len(gpus) >= 2:
        # Test P2P between first two GPUs
        config.p2p_capable = probe_p2p(gpus[0].index, gpus[1].index)

    same_vendor = len(set(g.vendor for g in gpus)) == 1
    if config.p2p_capable:
        config.transfer_method = "p2p"
    elif not same_vendor:
        config.transfer_method = "cross_vendor"
    else:
        config.transfer_method = "cpu_staged"

    # ── Pool calculation ──
    # Cooperative pool = sum of all free VRAM minus safety margin per GPU
    safety_margin_gb = 1.0  # Reserve 1 GB per GPU for OS/driver
    available = sum(
        max(g.free_vram_gb - safety_margin_gb, 0) for g in gpus
    )
    config.total_pool_gb = available

    # Max model size: ~80% of pool (need room for KV cache)
    config.model_max_gb = available * 0.80

    # ── Batch size recommendation ──
    # Based on smallest GPU's free VRAM (bottleneck)
    min_free = min(g.free_vram_gb for g in gpus) if gpus else 0
    if min_free >= 16:
        config.recommended_batch_size = 8
        config.recommended_max_tokens = 4096
    elif min_free >= 8:
        config.recommended_batch_size = 4
        config.recommended_max_tokens = 2048
    elif min_free >= 4:
        config.recommended_batch_size = 2
        config.recommended_max_tokens = 1024
    else:
        config.recommended_batch_size = 1
        config.recommended_max_tokens = 512

    _log.info(
        "Auto-config: %d GPUs, %.1f GB pool, strategy=%s, transfer=%s",
        len(gpus), config.total_pool_gb, strategy, config.transfer_method,
    )

    return config


# ═══════════════════════════════════════════════════════════════════════════
# Convenience for known setups
# ═══════════════════════════════════════════════════════════════════════════

def config_for_3090_5070ti() -> HeteroConfig:
    """Pre-computed optimal config for RTX 3090 + RTX 5070 Ti.

    This is the developer's own setup:
      - RTX 3090: 24 GB VRAM, Ampere, PCIe 4.0, 71 FP16 TFLOPS
      - RTX 5070 Ti: 16 GB VRAM, Blackwell, PCIe 5.0, 94.6 FP16 TFLOPS

    The 5070 Ti is faster at compute despite less VRAM, so:
      - Tier-0 = 5070 Ti (primary compute, fewer layers but faster)
      - Tier-1 = 3090 (more VRAM, hosts KV cache overflow)

    Split: balanced strategy gives ~58% layers to 3090 (VRAM-heavy)
    and ~42% to 5070 Ti (compute-heavy), which maximizes throughput
    because the 5070 Ti processes its share faster.

    Transfer: CPU-staged (different architectures, P2P may not work).
    Pool: 40 GB cooperative (24 + 16), ~32 GB usable for models.
    """
    config = auto_configure(strategy="balanced")

    if len(config.gpus) < 2:
        _log.warning("Expected 2 GPUs for 3090+5070Ti config, got %d", len(config.gpus))

    return config


__all__ = [
    "auto_configure",
    "detect_gpus",
    "config_for_3090_5070ti",
    "HeteroConfig",
    "DetectedGPU",
    "GPUProfile",
    "lookup_gpu_profile",
    "probe_p2p",
]
