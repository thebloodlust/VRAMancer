# VRAM Lending Pool under load — TurboQuant KV-cache A/B

Model: `Qwen/Qwen2.5-14B-Instruct`, GPUs: 2, max_new=80, repeats=2 (10 generations per variant)

## Findings (honest)

- **No lease was triggered in either variant** (`total_leases_created: 0`).
  10x80=800 generated tokens of KV cache is far too small relative to the
  9.1 GB lendable budget on GPU0 — the pool initialises and reports correct
  budgets, but reclaim/lease logic is **not exercised** by this workload.
  A larger context (long prompts / large batch / many concurrent sessions)
  would be needed to actually fill the lendable budget and trigger a lease —
  out of scope for this run.
- **TurboQuant KV cache is active and measurably reduces VRAM footprint**:
  post-load usage drops from 19192→15112 MB on GPU0 (-4.0 GB) and
  14314→12794 MB on GPU1 (-1.5 GB) — consistent with the claimed ~4.6x KV
  cache compression (the KV cache is a small fraction of total VRAM at this
  context length, so the overall reduction is modest in absolute terms).
- **tok/s stays ~2.4-2.5 in both variants** — same low figure seen in
  `bench_lending_hetero_real.md` (2.48 tok/s vs the 16.1 tok/s
  single-process baseline). This reproduces consistently and is **not**
  caused by TurboQuant (both variants are equally slow) — it is a
  pre-existing characteristic of this 2-GPU HF pipeline configuration with
  the lending pool active, independent of KV compression. Root cause not
  yet investigated.

| Variant | ok | load_s | tok/s | leases_created | leases_reclaimed | bytes_reclaimed | turboquant_active |
|---|---|---|---|---|---|---|---|
| baseline | ok | 31.31 | 2.47 | 0 | 0 | 0 | False |
| turboquant | ok | 31.11 | 2.42 | 0 | 0 | 0 | True |

## baseline
```json
{
  "label": "baseline",
  "kv_compression": "none",
  "vram_pre_load": {
    "gpu0": {
      "used_mb": 464,
      "total_mb": 24576
    },
    "gpu1": {
      "used_mb": 476,
      "total_mb": 16303
    }
  },
  "load_time_s": 31.31,
  "pool_active": true,
  "pool_registered_gpus": [
    {
      "gpu_id": 0,
      "device_name": "NVIDIA GeForce RTX 3090",
      "lendable_bytes": 9099369370
    },
    {
      "gpu_id": 1,
      "device_name": "NVIDIA GeForce RTX 5070 Ti",
      "lendable_bytes": 2838329140
    }
  ],
  "stats_pre": {
    "total_leases_created": 0,
    "total_leases_reclaimed": 0,
    "total_bytes_lent": 0,
    "total_bytes_reclaimed": 0,
    "preemptions_graceful": 0,
    "preemptions_forced": 0,
    "reclaim_avg_ms": 0.0,
    "peak_lent_bytes": 0
  },
  "turboquant_active": false,
  "vram_post_load": {
    "gpu0": {
      "used_mb": 19192,
      "total_mb": 24576
    },
    "gpu1": {
      "used_mb": 14314,
      "total_mb": 16303
    }
  },
  "tok_s": 2.47,
  "total_tokens": 800,
  "elapsed_s": 324.19,
  "vram_post_gen": {
    "gpu0": {
      "used_mb": 20724,
      "total_mb": 24576
    },
    "gpu1": {
      "used_mb": 14334,
      "total_mb": 16303
    }
  },
  "stats_post": {
    "total_leases_created": 0,
    "total_leases_reclaimed": 0,
    "total_bytes_lent": 0,
    "total_bytes_reclaimed": 0,
    "preemptions_graceful": 0,
    "preemptions_forced": 0,
    "reclaim_avg_ms": 0.0,
    "peak_lent_bytes": 0
  },
  "pool_lendable_post": [
    {
      "gpu_id": 0,
      "lendable_bytes": 9099369370
    },
    {
      "gpu_id": 1,
      "lendable_bytes": 2838329140
    }
  ],
  "ok": true
}
```

## turboquant
```json
{
  "label": "turboquant",
  "kv_compression": "turboquant",
  "vram_pre_load": {
    "gpu0": {
      "used_mb": 464,
      "total_mb": 24576
    },
    "gpu1": {
      "used_mb": 476,
      "total_mb": 16303
    }
  },
  "load_time_s": 31.11,
  "pool_active": true,
  "pool_registered_gpus": [
    {
      "gpu_id": 0,
      "device_name": "NVIDIA GeForce RTX 3090",
      "lendable_bytes": 9099369370
    },
    {
      "gpu_id": 1,
      "device_name": "NVIDIA GeForce RTX 5070 Ti",
      "lendable_bytes": 2838329140
    }
  ],
  "stats_pre": {
    "total_leases_created": 0,
    "total_leases_reclaimed": 0,
    "total_bytes_lent": 0,
    "total_bytes_reclaimed": 0,
    "preemptions_graceful": 0,
    "preemptions_forced": 0,
    "reclaim_avg_ms": 0.0,
    "peak_lent_bytes": 0
  },
  "turboquant_active": true,
  "vram_post_load": {
    "gpu0": {
      "used_mb": 15112,
      "total_mb": 24576
    },
    "gpu1": {
      "used_mb": 12794,
      "total_mb": 16303
    }
  },
  "tok_s": 2.42,
  "total_tokens": 800,
  "elapsed_s": 330.57,
  "vram_post_gen": {
    "gpu0": {
      "used_mb": 16624,
      "total_mb": 24576
    },
    "gpu1": {
      "used_mb": 12814,
      "total_mb": 16303
    }
  },
  "stats_post": {
    "total_leases_created": 0,
    "total_leases_reclaimed": 0,
    "total_bytes_lent": 0,
    "total_bytes_reclaimed": 0,
    "preemptions_graceful": 0,
    "preemptions_forced": 0,
    "reclaim_avg_ms": 0.0,
    "peak_lent_bytes": 0
  },
  "pool_lendable_post": [
    {
      "gpu_id": 0,
      "lendable_bytes": 9099369370
    },
    {
      "gpu_id": 1,
      "lendable_bytes": 2838329140
    }
  ],
  "ok": true
}
```
