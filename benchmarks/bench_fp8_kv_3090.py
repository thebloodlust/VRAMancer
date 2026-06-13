"""Standalone FP8 KV bench — subprocess per config (clean VRAM)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

CONFIGS = [
    {"label": "kv_auto", "kv_dtype": "auto", "ctx": 2048},
    {"label": "kv_auto", "kv_dtype": "auto", "ctx": 4096},
    {"label": "kv_fp8",  "kv_dtype": "fp8",  "ctx": 2048},
    {"label": "kv_fp8",  "kv_dtype": "fp8",  "ctx": 4096},
    {"label": "kv_fp8",  "kv_dtype": "fp8",  "ctx": 8192},
    {"label": "kv_fp8",  "kv_dtype": "fp8",  "ctx": 16384},
]

OUT = Path("benchmarks/results/bench_fp8_kv_3090.json")
OUT.parent.mkdir(parents=True, exist_ok=True)
WORKER = Path("benchmarks/_fp8_kv_worker.py")

results = []
for cfg in CONFIGS:
    print(f"\n>>> {cfg['label']} ctx={cfg['ctx']}", flush=True)
    env = {**os.environ, "KV_DTYPE": cfg["kv_dtype"], "CTX": str(cfg["ctx"])}
    try:
        out = subprocess.run(
            [sys.executable, str(WORKER)],
            env=env, capture_output=True, text=True, timeout=900,
        )
        last = ""
        for line in out.stdout.strip().split("\n"):
            line = line.strip()
            if line.startswith("{"):
                last = line
        if last:
            r = json.loads(last)
        else:
            r = {"status": "no_json", "stderr": out.stderr[-300:]}
    except subprocess.TimeoutExpired:
        r = {"status": "timeout"}
    except Exception as e:
        r = {"status": "exception", "err": str(e)}
    r.update(cfg)
    print(f"    {r}", flush=True)
    results.append(r)
    OUT.write_text(json.dumps({"results": results}, indent=2))

print("\n=== Summary ===")
print(f"{'cfg':10s} {'ctx':>6s}  {'tok/s':>8s}  {'kv_MB':>10s}  status")
for r in results:
    ts = r.get("avg_tok_s", "-")
    kv = r.get("kv_cache_mb", "-")
    print(f"{r['label']:10s} {r['ctx']:6d}  {ts!s:>8s}  {kv!s:>10s}  {r.get('status','?')}")
print(f"\nResults: {OUT}")
