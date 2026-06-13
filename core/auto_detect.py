"""Auto-detection helpers for quantization + virtualization environment.

Two functions exposed:

* :func:`recommend_quantization()` returns the best quant mode
  (``nvfp4`` / ``bf16`` / ``nf4`` / ``int8`` / ``gguf_q4_k_m``) for the
  detected GPUs based on compute capability and free VRAM.
* :func:`detect_virtualization()` returns one of ``"baremetal"``,
  ``"proxmox"``, ``"kvm"``, ``"vmware"``, ``"hyperv"``, ``"docker"``,
  ``"unknown"``.  Used to auto-disable P2P CUDA paths in IOMMU-bound VMs.

Both functions are pure-Python, side-effect free, and degrade gracefully
when ``torch`` or ``pynvml`` are missing (returns ``"bf16"`` / ``"unknown"``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Quantization auto-pick
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GpuSpec:
    index: int
    name: str
    sm_major: int
    sm_minor: int
    vram_gb: float

    @property
    def sm(self) -> float:
        return self.sm_major + self.sm_minor / 10.0


def _enumerate_gpus() -> List[GpuSpec]:
    """Best-effort GPU enumeration via torch (CUDA/ROCm). Empty list on failure."""
    try:
        import torch  # type: ignore
    except Exception:
        return []
    if not torch.cuda.is_available():
        return []
    out: List[GpuSpec] = []
    try:
        n = torch.cuda.device_count()
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            out.append(GpuSpec(
                index=i,
                name=props.name,
                sm_major=props.major,
                sm_minor=props.minor,
                vram_gb=props.total_memory / (1024 ** 3),
            ))
    except Exception:
        return []
    return out


# Quant choice rules (bf16 baseline, decision tree):
#
#   SM ≥ 10.0                 → nvfp4   (Blackwell native FP4)
#   SM ≥ 8.0  AND vram ≥ 22GB → bf16    (Ampere/Ada with room)
#   SM ≥ 8.0  AND vram ≥ 10GB → nf4     (BnB 4-bit, fits 7-13B)
#   SM ≥ 7.5                  → int8    (LLM.int8)
#   SM <  7.5                 → gguf_q4_k_m (CPU+small VRAM)
#
# All thresholds are intentionally conservative.
_RULES: List[Tuple[str, float, float]] = [
    ("nvfp4",       10.0, 0.0),
    ("bf16",         8.0, 22.0),
    ("nf4",          8.0, 10.0),
    ("int8",         7.5, 0.0),
    ("gguf_q4_k_m",  0.0, 0.0),
]


def recommend_quantization(
    gpus: Optional[List[GpuSpec]] = None,
    *,
    min_vram_for_bf16_gb: float = 22.0,
) -> str:
    """Return the recommended quantization mode for the detected GPUs.

    Picks the *worst* GPU as the constraint (heterogeneous setups must
    fit the smallest member). Falls back to ``"bf16"`` when no GPU info
    is available — caller can still override via ``VRM_QUANTIZATION``.
    """
    if gpus is None:
        gpus = _enumerate_gpus()
    if not gpus:
        return "bf16"

    # Constrain to the smallest/oldest GPU.
    weakest = min(gpus, key=lambda g: (g.sm, g.vram_gb))
    sm = weakest.sm
    vram = weakest.vram_gb

    if sm >= 10.0:
        return "nvfp4"
    if sm >= 8.0 and vram >= min_vram_for_bf16_gb:
        return "bf16"
    if sm >= 8.0 and vram >= 10.0:
        return "nf4"
    if sm >= 7.5:
        return "int8"
    return "gguf_q4_k_m"


# ---------------------------------------------------------------------------
# Virtualization detection
# ---------------------------------------------------------------------------

def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except OSError:
        return ""


def detect_virtualization() -> str:
    """Return one of: baremetal | proxmox | kvm | vmware | hyperv | docker | unknown.

    Detection order (cheapest first):
      1. ``/proc/1/cgroup`` for container markers (docker/lxc).
      2. ``systemd-detect-virt`` if available (most reliable on Linux).
      3. ``/sys/class/dmi/id/{sys_vendor,product_name}`` fallback.
      4. ``/proc/cpuinfo`` flag ``hypervisor`` → at least KVM-ish.

    Proxmox = QEMU/KVM with the typical sys_vendor/product strings
    ("QEMU", "Standard PC (i440FX + PIIX, 1996)") — heuristic.
    """
    # Step 1 — container check
    cgroup = _read("/proc/1/cgroup") + _read("/proc/self/cgroup")
    if "docker" in cgroup or "containerd" in cgroup:
        return "docker"
    if "lxc" in cgroup:
        return "lxc"

    # Step 2 — systemd-detect-virt
    sdv = shutil.which("systemd-detect-virt")
    if sdv:
        try:
            res = subprocess.run([sdv], capture_output=True, text=True, timeout=2)
            v = res.stdout.strip().lower()
            if v in ("none", ""):
                return "baremetal"
            if v in ("kvm", "qemu"):
                # disambiguate proxmox vs vanilla kvm via DMI
                vendor = _read("/sys/class/dmi/id/sys_vendor").lower()
                product = _read("/sys/class/dmi/id/product_name").lower()
                if "qemu" in vendor and ("i440fx" in product or "q35" in product):
                    return "proxmox"
                return "kvm"
            if v == "vmware":
                return "vmware"
            if v in ("microsoft", "hyperv"):
                return "hyperv"
            return v  # xen, oracle, etc.
        except (subprocess.TimeoutExpired, OSError):
            pass

    # Step 3 — DMI fallback
    vendor = _read("/sys/class/dmi/id/sys_vendor").lower()
    product = _read("/sys/class/dmi/id/product_name").lower()
    if "qemu" in vendor:
        if "i440fx" in product or "q35" in product:
            return "proxmox"
        return "kvm"
    if "vmware" in vendor:
        return "vmware"
    if "microsoft" in vendor and "virtual" in product:
        return "hyperv"

    # Step 4 — CPU hypervisor flag (last resort)
    cpuinfo = _read("/proc/cpuinfo")
    if "hypervisor" in cpuinfo:
        return "kvm"

    return "baremetal" if cpuinfo else "unknown"


def should_disable_p2p() -> bool:
    """True if running in a virtualization environment where CUDA P2P is
    typically unreliable due to IOMMU constraints (Proxmox, VMware, KVM).

    Caller can still override with ``VRM_TRANSFER_P2P=1``.
    """
    # User explicit override wins both ways.
    raw = os.environ.get("VRM_TRANSFER_P2P", "").strip().lower()
    if raw in ("0", "false", "no"):
        return True
    if raw in ("1", "true", "yes"):
        return False

    virt = detect_virtualization()
    return virt in {"proxmox", "vmware", "hyperv", "kvm"}


__all__ = [
    "GpuSpec",
    "recommend_quantization",
    "detect_virtualization",
    "should_disable_p2p",
    "recommend_backend",
]


# ---------------------------------------------------------------------------
# Backend auto-pick
# ---------------------------------------------------------------------------

# Backend availability is checked at first call (cached forever — backends
# don't appear/disappear at runtime).
_BACKEND_AVAIL: Optional[dict] = None


def _check_backend_availability() -> dict:
    """Probe which backends are importable. Cached.

    Returns a dict of ``{backend_name: bool}`` for: huggingface, vllm,
    llamacpp, ollama. Never raises.
    """
    global _BACKEND_AVAIL
    if _BACKEND_AVAIL is not None:
        return _BACKEND_AVAIL
    avail = {}

    def _try(name: str) -> bool:
        try:
            __import__(name)
            return True
        except Exception:
            return False

    avail["huggingface"] = _try("transformers")
    avail["vllm"] = _try("vllm")
    avail["llamacpp"] = _try("llama_cpp")
    # Ollama is a separate REST service — probe only the client lib here.
    # The actual REST endpoint check is left to the caller.
    avail["ollama"] = _try("requests")  # ollama just needs HTTP

    _BACKEND_AVAIL = avail
    return avail


def recommend_backend(model: str, *, available: Optional[dict] = None) -> str:
    """Pick the best backend for a given model identifier.

    Heuristics
    ----------
    * ``.gguf`` extension or ``GGUF`` in the path           → ``llamacpp``
    * ``ollama://`` scheme or starts with ``ollama:``      → ``ollama``
    * ``-fp8`` / ``FP8`` / ``-awq`` / ``-gptq`` substring  → ``vllm``
      (vLLM has the best kernels for those)
    * default                                              → ``huggingface``

    The chosen backend is then validated against availability — falls back
    to ``"huggingface"`` if the preferred one is not installed.

    Pure function: no I/O, no model loading. Returns one of:
    ``"huggingface" | "vllm" | "llamacpp" | "ollama"``.
    """
    avail = available if available is not None else _check_backend_availability()

    name_lower = model.lower()
    chosen: str

    if model.startswith("ollama://") or model.startswith("ollama:"):
        chosen = "ollama"
    elif name_lower.endswith(".gguf") or "gguf" in name_lower or "/q4_" in name_lower or "/q5_" in name_lower:
        chosen = "llamacpp"
    elif any(tag in name_lower for tag in ("-fp8", "fp8-", "-awq", "awq-", "-gptq", "gptq-")):
        chosen = "vllm"
    else:
        chosen = "huggingface"

    # Fallback if chosen backend isn't installed.
    if not avail.get(chosen, False):
        # Prefer the next most general backend.
        order = ["huggingface", "vllm", "llamacpp", "ollama"]
        for fallback in order:
            if avail.get(fallback, False):
                return fallback
        # Truly nothing — return chosen anyway and let import error surface.
    return chosen
