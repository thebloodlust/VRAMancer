# VRAM Lending Pool under artificial VRAM pressure

Model: `Qwen/Qwen2.5-14B-Instruct`, GPUs: 2, filler=10000MB pre-allocated on gpu1 before model load.

## Finding: LENDING_ON OOMs in a scenario where LENDING_OFF succeeds (real bug, root-caused)

**Unexpected and reproducible**: with 10 GB of dummy data pre-allocated on
GPU1 (simulating "another tenant already using this GPU" — the exact
scenario the lending pool exists to handle), `LENDING_ON` **fails to load**
the model with a CUDA OOM during weight dispatch, while `LENDING_OFF`
**loads successfully** (GPU1 ends up at 16154/16303 MB, essentially full but
fitting).

Root cause, found by reading the code (not guessed):
`experimental/vram_lending.py:1268-1289`
(`VRAMLendingPool.suggest_placement_budget`) computes each GPU's "usable"
placement budget as:

```python
safety = int(b.total_bytes * self.policy.min_free_ratio)
runtime = int(b.total_bytes * runtime_headroom_ratio)
usable = max(0, b.total_bytes - safety - runtime)
```

i.e. **`usable` is derived from `total_bytes` (the GPU's total VRAM
capacity), not from currently-free VRAM** — the code comment even says
*"We ignore current model_bytes/kv_bytes here"*. So with 16 GB total on
GPU1, the pool suggests a budget of roughly `16 GB * (1 - min_free_ratio -
0.05)` regardless of the fact that 10 GB are already occupied by another
allocation in the process. `_build_compute_aware_memory_map`
(`core/backends.py:611-631`) takes this suggestion as `max_memory` for
`accelerate`, which then tries to place more weights on GPU1 than the
~5.5 GB actually free → OOM during `caching_allocator_warmup`.

`LENDING_OFF` falls back to the static formula in
`_build_compute_aware_memory_map`, which uses `gpu.free_vram_gb` (real free
VRAM from `hetero_config.auto_configure`) — correctly sized, so it fits.

**Caveat**: the "filler" here is a same-process dummy `torch.empty()`
allocation, not a truly separate foreign process. NVML-based free-memory
queries (used by `hetero_config`) should report the same numbers either way,
but this hasn't been independently verified against a real second process.
The conclusion (the budget formula ignores current usage) is read directly
from the source and is independent of the filler methodology.

**This is the opposite of `bench_lending_hetero_real.md`**, where
`LENDING_ON` succeeded and `LENDING_OFF` OOM'd — that scenario had the
*other* process's VRAM usage present *before* `InferencePipeline` was even
constructed (so `_init_lending_pool`'s initial `_budgets` were registered
against the already-reduced free VRAM, i.e. `total_bytes` there reflected a
different, healthier topology). Here, the pressure is applied *within* the
same totals, exposing that `suggest_placement_budget` doesn't re-check free
VRAM at suggestion time.

**Recommendation for the architect**: `suggest_placement_budget` should
subtract currently-used VRAM (`total_bytes - free_bytes` at call time, via
`torch.cuda.mem_get_info` / NVML) from `usable`, not just apply a fixed
ratio of `total_bytes`. Until fixed, the lending-pool-aware placement path
can be *less* robust than the static fallback under real VRAM contention.

| Variant | ok | load_s | tok/s | leases_created | leases_reclaimed | bytes_reclaimed | error |
|---|---|---|---|---|---|---|---|
| LENDING_ON | FAIL | - | - | - | - | - | CUDA out of memory. Tried to allocate 11.28 GiB. GPU 1 has a total capacity of 1 |
| LENDING_OFF | ok | 73.97 | 1.03 | 0 | 0 | 0 | |

## LENDING_ON
```json
{
  "label": "LENDING_ON",
  "lending_enabled": true,
  "filler_gpu": 1,
  "filler_mb": 10000,
  "vram_pre_filler": {
    "gpu0": {
      "used_mb": 464,
      "total_mb": 24576
    },
    "gpu1": {
      "used_mb": 476,
      "total_mb": 16303
    }
  },
  "vram_post_filler": {
    "gpu0": {
      "used_mb": 728,
      "total_mb": 24576
    },
    "gpu1": {
      "used_mb": 10706,
      "total_mb": 16303
    }
  },
  "ok": false,
  "error": "CUDA out of memory. Tried to allocate 11.28 GiB. GPU 1 has a total capacity of 15.47 GiB of which 5.46 GiB is free. Including non-PyTorch memory, this process has 9.98 GiB memory in use. Of the allocated memory 9.77 GiB is allocated by PyTorch, and 2.00 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)",
  "traceback": "Traceback (most recent call last):\n  File \"<string>\", line 77, in <module>\n  File \"/home/jeremie/VRAMancer/VRAMancer/core/inference_pipeline.py\", line 269, in load\n    self.backend.load_model(model_name, num_gpus=self.num_gpus, **model_kwargs)\n  File \"/home/jeremie/VRAMancer/VRAMancer/core/backends.py\", line 1380, in load_model\n    self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)\n                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/jeremie/VRAMancer/VRAMancer/.venv/lib/python3.12/site-packages/transformers/models/auto/auto_factory.py\", line 405, in from_pretrained\n    return model_class.from_pretrained(\n           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/jeremie/VRAMancer/VRAMancer/.venv/lib/python3.12/site-packages/transformers/modeling_utils.py\", line 4245, in from_pretrained\n    loading_info, disk_offload_index = cls._load_pretrained_model(model, state_dict, checkpoint_files, load_config)\n                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n  File \"/home/jeremie/VRAMancer/VRAMancer/.venv/lib/python3.12/site-packages/transformers/modeling_utils.py\", line 4326, in _load_pretrained_model\n    caching_allocator_warmup(model, expanded_device_map, load_config.hf_quantizer)\n  File \"/home/jeremie/VRAMancer/VRAMancer/.venv/lib/python3.12/site-packages/transformers/modeling_utils.py\", line 4980, in caching_allocator_warmup\n    _ = torch.empty(int(byte_count // 2), dtype=torch.float16, device=device, requires_grad=False)\n        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 11.28 GiB. GPU 1 has a total capacity of 15.47 GiB of which 5.46 GiB is free. Including non-PyTorch memory, this process has 9.98 GiB memory in use. Of the allocated memory 9.77 GiB is allocated by PyTorch, and 2.00 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)\n"
}
```

## LENDING_OFF
```json
{
  "label": "LENDING_OFF",
  "lending_enabled": false,
  "filler_gpu": 1,
  "filler_mb": 10000,
  "vram_pre_filler": {
    "gpu0": {
      "used_mb": 464,
      "total_mb": 24576
    },
    "gpu1": {
      "used_mb": 476,
      "total_mb": 16303
    }
  },
  "vram_post_filler": {
    "gpu0": {
      "used_mb": 728,
      "total_mb": 24576
    },
    "gpu1": {
      "used_mb": 10706,
      "total_mb": 16303
    }
  },
  "load_time_s": 73.97,
  "pool_active": false,
  "vram_post_load": {
    "gpu0": {
      "used_mb": 19192,
      "total_mb": 24576
    },
    "gpu1": {
      "used_mb": 16154,
      "total_mb": 16303
    }
  },
  "tok_s": 1.03,
  "total_tokens": 800,
  "elapsed_s": 773.78,
  "vram_post_gen": {
    "gpu0": {
      "used_mb": 20728,
      "total_mb": 24576
    },
    "gpu1": {
      "used_mb": 16188,
      "total_mb": 16303
    }
  },
  "ok": true
}
```
