"""VRAMancer Production Health Diagnostics.

Comprehensive health checking for production deployments:
  - Per-GPU health: VRAM usage, temperature, utilization, fault state
  - VRAM lending pool status (active leases, pool utilization)
  - KV cache utilization and overflow stats
  - Transfer manager status (P2P vs CPU-staged, cross-vendor)
  - Fault tolerance state (healthy/degraded/failed GPUs)
  - System resources (RAM, CPU, disk)

Exécution :
    python -m core.health             # Quick check
    python -m core.health --full      # Full diagnostics
    python -m core.health --json      # JSON output
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import torch  # type: ignore
except ImportError:
    torch = None  # type: ignore

_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

OPTIONAL = ["vllm", "ollama", "requests", "prometheus_client", "psutil"]


# ═══════════════════════════════════════════════════════════════════════════
# Individual checks
# ═══════════════════════════════════════════════════════════════════════════

def check_optional() -> Dict[str, bool]:
    """Check availability of optional dependencies."""
    available = {}
    for m in OPTIONAL:
        try:
            importlib.import_module(m)
            available[m] = True
        except Exception:
            available[m] = False
    return available


def gpu_summary() -> Dict[str, Any]:
    """Basic GPU summary (count, names, VRAM)."""
    if torch is None or not torch.cuda.is_available():
        return {"available": False, "count": 0, "devices": []}

    devices = []
    for i in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = props.total_mem
            devices.append({
                "index": i,
                "name": props.name,
                "total_vram_mb": round(total / 1e6),
                "allocated_mb": round(allocated / 1e6),
                "reserved_mb": round(reserved / 1e6),
                "free_mb": round((total - allocated) / 1e6),
                "utilization_pct": round(allocated / max(total, 1) * 100, 1),
                "compute_capability": f"{props.major}.{props.minor}",
            })
        except Exception as e:
            devices.append({"index": i, "error": str(e)})

    return {
        "available": True,
        "count": len(devices),
        "devices": devices,
    }


def gpu_detailed_health() -> Dict[str, Any]:
    """Per-GPU health with temperature and fault tolerance state."""
    result: Dict[str, Any] = {"gpus": {}}

    if torch is None or not torch.cuda.is_available():
        return result

    # Get fault tolerance state if available
    fault_states: Dict[int, Dict] = {}
    try:
        from core.gpu_fault_tolerance import get_fault_manager
        fm = get_fault_manager()
        stats = fm.stats()
        fault_states = stats.get("gpu_states", {})
    except Exception:
        pass

    # Get temperature via pynvml if available
    nvml_handles = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        for i in range(torch.cuda.device_count()):
            try:
                nvml_handles[i] = pynvml.nvmlDeviceGetHandleByIndex(i)
            except Exception:
                pass
    except ImportError:
        pass

    for i in range(torch.cuda.device_count()):
        gpu_info: Dict[str, Any] = {}
        try:
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i)
            total = props.total_mem

            gpu_info["name"] = props.name
            gpu_info["vram_total_mb"] = round(total / 1e6)
            gpu_info["vram_used_mb"] = round(allocated / 1e6)
            gpu_info["vram_free_mb"] = round((total - allocated) / 1e6)
            gpu_info["vram_pct"] = round(allocated / max(total, 1) * 100, 1)

            # Temperature
            handle = nvml_handles.get(i)
            if handle:
                try:
                    import pynvml
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    gpu_info["temperature_c"] = temp
                    gpu_info["thermal_ok"] = temp < 85
                except Exception:
                    gpu_info["temperature_c"] = -1

            # Fault state
            fs = fault_states.get(i, fault_states.get(str(i), {}))
            if fs:
                gpu_info["health"] = fs.get("health", "UNKNOWN")
                gpu_info["fault_count"] = fs.get("fault_count", 0)
                gpu_info["oom_count"] = fs.get("oom_count", 0)
            else:
                gpu_info["health"] = "HEALTHY"
                gpu_info["fault_count"] = 0

        except Exception as e:
            gpu_info["error"] = str(e)
            gpu_info["health"] = "ERROR"

        result["gpus"][str(i)] = gpu_info

    return result


def lending_pool_status() -> Dict[str, Any]:
    """VRAM lending pool status."""
    try:
        from core.vram_lending import get_lending_pool
        pool = get_lending_pool()
        return pool.stats()
    except Exception:
        return {"active": False}


def kv_cache_status() -> Dict[str, Any]:
    """PagedKV cache utilization."""
    try:
        from core.inference_pipeline import get_pipeline
        pipeline = get_pipeline()
        if pipeline.paged_kv:
            return pipeline.paged_kv.stats()
    except Exception:
        pass
    return {"active": False}


def transfer_status() -> Dict[str, Any]:
    """Transfer manager status including cross-vendor bridge."""
    try:
        from core.inference_pipeline import get_pipeline
        pipeline = get_pipeline()
        if pipeline.transfer_manager:
            return pipeline.transfer_manager.stats()
    except Exception:
        pass
    return {"active": False}


def cross_vendor_status() -> Dict[str, Any]:
    """Cross-vendor bridge status."""
    try:
        from core.cross_vendor_bridge import get_cross_vendor_bridge
        bridge = get_cross_vendor_bridge()
        return bridge.stats()
    except Exception:
        return {"active": False}


def hetero_config_status() -> Dict[str, Any]:
    """Heterogeneous GPU configuration."""
    try:
        from core.hetero_config import auto_configure
        config = auto_configure()
        return config.to_dict()
    except Exception:
        return {"detected": False}


def system_resources() -> Dict[str, Any]:
    """System resource usage (RAM, CPU, disk)."""
    info: Dict[str, Any] = {}
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / (1024 ** 3), 1)
        info["ram_available_gb"] = round(mem.available / (1024 ** 3), 1)
        info["ram_pct"] = mem.percent
        info["cpu_count"] = psutil.cpu_count()
        info["cpu_pct"] = psutil.cpu_percent(interval=0.1)
        disk = psutil.disk_usage("/")
        info["disk_free_gb"] = round(disk.free / (1024 ** 3), 1)
    except ImportError:
        info["psutil"] = "not installed"
    return info


# ═══════════════════════════════════════════════════════════════════════════
# Full diagnostic report
# ═══════════════════════════════════════════════════════════════════════════

def full_diagnostic() -> Dict[str, Any]:
    """Complete production health diagnostic."""
    try:
        from core import __version__
        version = __version__
    except Exception:
        version = "0.2.4"

    report: Dict[str, Any] = {
        "timestamp": time.time(),
        "version": version,
        "python": sys.version.split()[0],
        "torch": torch.__version__ if torch else "not installed",
        "cuda": (torch.version.cuda if torch and hasattr(torch, "version") and torch.version.cuda else None),
        "gpu": gpu_summary(),
        "gpu_health": gpu_detailed_health(),
        "system": system_resources(),
        "optional_deps": check_optional(),
    }

    # Production subsystems (only if initialized)
    report["lending_pool"] = lending_pool_status()
    report["kv_cache"] = kv_cache_status()
    report["transfer"] = transfer_status()
    report["cross_vendor"] = cross_vendor_status()

    # Overall status
    all_healthy = True
    warnings: List[str] = []

    gpu_data = report["gpu"]
    if not gpu_data.get("available"):
        warnings.append("No GPU available")
        all_healthy = False

    gpu_health = report["gpu_health"]
    for gid, info in gpu_health.get("gpus", {}).items():
        health = info.get("health", "UNKNOWN")
        if health not in ("HEALTHY", "DEGRADED"):
            all_healthy = False
            warnings.append(f"GPU {gid}: {health}")
        temp = info.get("temperature_c", -1)
        if temp > 80:
            warnings.append(f"GPU {gid}: high temperature ({temp}°C)")

    sys_info = report["system"]
    ram_pct = sys_info.get("ram_pct", 0)
    if ram_pct > 90:
        warnings.append(f"High RAM usage ({ram_pct}%)")

    report["overall"] = "healthy" if all_healthy and not warnings else "degraded"
    report["warnings"] = warnings

    return report


def quick_check() -> Dict[str, Any]:
    """Quick health check (for /health endpoint)."""
    try:
        from core import __version__
        version = __version__
    except Exception:
        version = "0.2.4"

    return {
        "status": "healthy",
        "version": version,
        "python": sys.version.split()[0],
        "torch": torch.__version__ if torch else "not installed",
        "cuda": (torch.version.cuda if torch and hasattr(torch, "version") and torch.version.cuda else None),
        "gpu": gpu_summary(),
        "optional": check_optional(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    full = "--full" in sys.argv or "-f" in sys.argv
    as_json = "--json" in sys.argv or "-j" in sys.argv

    if full:
        report = full_diagnostic()
    else:
        report = quick_check()

    if as_json or full:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()