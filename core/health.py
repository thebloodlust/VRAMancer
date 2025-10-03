"""Healthcheck central VRAMancer.

Exécution :
    python -m core.health
"""
from __future__ import annotations
import importlib
import json
import torch
import sys

OPTIONAL = ["vllm", "ollama", "requests"]

def check_optional():
    available = {}
    for m in OPTIONAL:
        try:
            importlib.import_module(m)
            available[m] = True
        except Exception:
            available[m] = False
    return available

def gpu_summary():
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "count": torch.cuda.device_count(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    }

def main():
    report = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu_summary(),
        "optional": check_optional(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()