# V6.E Expert Pinning — Phase B Design Plan

**Branch target**: `feat/v6-lending-cooperative` (continuation)
**Goal**: pin top-K MoE experts on GPU 0 (5070 Ti, 16 GB), keep cold experts
materialized as a per-expert sliced view of a persistent lending tensor on
GPU 1 (3090, 24 GB), routed via cudaMemcpyPeerAsync at ~11.7 GB/s. Target:
**15–20 tok/s on FP8** (vs 5 tok/s V6.D baseline, vs 30 tok/s AWQ ceiling).

> Authored by the V6.E Plan agent based on `JOURNAL_V6D_V6E_SESSION.md`,
> the expert histogram, and a read of vLLM 0.20.1 internals. This is the
> brief to feed a fresh implementation session.

---

## A. Hot/cold registry format

### Source data
`/home/jeremie/VRAMancer/VRAMancer/benchmarks/results/qwen3_coder_expert_histogram.json`.
Schema confirmed: `histogram[layer][expert]` int counts, `num_layers=48`,
`num_experts_per_layer=128`, `topk=8`, top-25/layer = 56% of activations.

### Lookup table contract
Generate at worker start time a `dict[int, frozenset[int]]` keyed by **physical
layer index**, value = the set of `expert_id` declared "hot" for that layer.
Pre-compute on the parent process side and serialize through env (path) plus a
side cache file. Two access shapes are needed at runtime:

- **Weight loader hook** (load-time, called per-layer): needs
  `is_hot(layer_idx, expert_id) -> bool` to decide each expert's residency.
- **Forward hook** (per-token, hot path): needs to be a contiguous int8/bool
  tensor `expert_residency[layer_idx]: shape (num_experts,)` on the compute
  GPU, so the MoE kernel can branch without Python overhead. Build once after
  weight load, copy to `cuda:0`.

### Cache format
Store a derived `hot_experts.json` next to the histogram (same
`benchmarks/results/` directory):
```json
{
  "model": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
  "source_histogram": "qwen3_coder_expert_histogram.json",
  "topk_pct": 20,
  "topk_per_layer": 25,
  "hot_share_observed": 0.560,
  "hot_experts": {"0": [19, 28, 50, ...], "1": [...], ..., "47": [...]}
}
```
Reasons to persist:
1. Worker process can load it with one `json.load()` call instead of re-scanning
   48×128 floats.
2. The set is pinned per session; reproducibility for benchmarks.
3. We can hand-edit it (e.g. to test "what if we lie and mark 50% hot") without
   re-running the profile.

### Compute helper
A new module `core/expert_pinning.py` (does not exist yet) holds:
- `load_hot_registry(path: str) -> dict[int, frozenset[int]]`
- `build_residency_tensor(reg, num_layers, num_experts, device) -> torch.Tensor`
  returning a `(num_layers, num_experts)` `torch.bool` tensor on the compute GPU
- `compute_hot_from_histogram(histogram_path, topk_pct) -> dict` (writes
  `hot_experts.json` if missing)

## B. Weight-placement step at model load

### The intercept site
For **AWQ Int4**:
`vllm/model_executor/layers/quantization/awq_marlin.py:508` —
`AWQMarlinMoEMethod.create_weights`. The function allocates 6 expert-batched
tensors on the current default device:
- `w13_qweight`: `(num_experts, hidden_size, 2*intermediate/pack_factor)` int32
- `w2_qweight`: `(num_experts, intermediate, hidden_size/pack_factor)` int32
- `w13_scales`, `w2_scales`, `w13_qzeros`, `w2_qzeros`

For **FP8**: `vllm/model_executor/layers/quantization/fp8.py:600` —
`Fp8MoEMethod.create_weights`. Similarly produces:
- `w13_weight`: `(num_experts, 2*intermediate, hidden_size)` fp8e4m3
- `w2_weight`: `(num_experts, hidden_size, intermediate)` fp8e4m3
- `w13_weight_scale`, `w2_weight_scale`, plus optional input scales

**Both paths allocate the expert dimension as a single contiguous tensor of
shape `(num_experts, ...)`**. We CANNOT trivially put expert-by-expert tensors
on different devices because Marlin/FP8 kernels expect a single base pointer
addressed by `expert_id * stride[0]`. This is the single most important
constraint of the plan and drives the design of part C.

### Strategy: split storage, unified address space via shadow tensor

We replace the single `w13_qweight` (and friends) parameter with a **router-side
shadow** that the kernel still sees as a single 3D tensor, but whose backing
storage is split as follows:
1. `w13_hot`: `(num_hot, ...)` allocated on `cuda:0` (the compute GPU).
   `num_hot ≈ 25` per layer.
2. `w13_cold`: `(num_cold, ...)` allocated on `cuda:1` (lending tensor slice).
   `num_cold ≈ 103` per layer.
3. A `(num_experts,)` int32 `expert_loc` tensor on `cuda:0` mapping global
   expert id → row index in the hot tensor (or in the staged scratch).

The kernel cannot index across devices, so for the **first PoC (Phase B-1)** we
keep the canonical `w13_qweight` parameter as a contiguous `(num_experts, ...)`
tensor on `cuda:0` and treat the GPU-1 buffer as a **prefetch source** (option
C-2 below). This is the path that minimizes risk and is the only one compatible
with the existing `fused_marlin_moe` kernel signature without patching kernels.

### Patch site mechanics
In the worker process, we monkey-patch `AWQMarlinMoEMethod.create_weights` and
`Fp8MoEMethod.create_weights` with a wrapper installed via
`core/expert_pinning.py:install_create_weights_hooks()`. The wrapper:
1. Calls the original `create_weights` on the layer (allocates the full
   `(num_experts, …)` tensor on `cuda:0`).
2. Reads `layer.layer_name` (or extracts `layer_id` via `extract_layer_index`)
   to get the physical layer index.
3. Allocates **shadow cold tensors** on `cuda:1` of the cold-experts subset
   (e.g. `w13_qweight_cold: (num_cold, hidden_size, ...)`) by carving slices
   out of the lending pool's persistent staging buffer.
4. Stores them as new attributes: `layer._vrm_cold_w13`, `layer._vrm_cold_w2`,
   `layer._vrm_cold_w13_scales`, `layer._vrm_cold_w2_scales`,
   `layer._vrm_cold_w13_qzeros`, `layer._vrm_cold_w2_qzeros`,
   `layer._vrm_hot_mask` (bool, num_experts), `layer._vrm_cold_remap`
   (int32: global expert id → row in cold tensor or `-1` if hot).

### Where the cold weights actually come from
At weight-load time vLLM populates `layer.w13_qweight[expert_id]` slice-by-slice
via the `weight_loader` callable. We hook **after** `process_weights_after_loading`
runs (this is when AWQ does the Marlin repack at line 645). Right after Marlin
repack:
1. For each cold expert id, we copy `layer.w13_qweight[expert_id]` (on cuda:0)
   → `layer._vrm_cold_w13[cold_row_for(expert_id)]` (on cuda:1) via
   `tensor.copy_()` which goes through cudaMemcpyPeerAsync.
2. Once all cold experts are mirrored on cuda:1, **we zero out** the cold rows
   of `layer.w13_qweight` on cuda:0 to reclaim VRAM. At runtime the forward
   hook restores them on demand into a small scratch.

This is the trade-off pragma: we keep the canonical tensor signature for the
kernel AND have authoritative cold-expert weights on cuda:1. Cost: one full
mirror at load-time, no mirror at runtime.

### Patch installation timing
Critical — the patches must be installed **inside the worker process**, not the
parent. The journal section "Chasse au coupable" documents that vLLM uses
`multiprocessing.spawn`; the parent's monkey-patches are not inherited. Two
possible install points:

1. **vLLM's `WorkerWrapperBase`** in `vllm/worker/worker_base.py` — overridable
   via `VLLM_WORKER_CLS` env var. Too invasive.
2. **PyTorch dispatch import hook**: set `VLLM_PLUGINS` (vLLM 0.20.1 supports
   plugin entry points) — too heavyweight for a PoC.
3. **Recommended**: a sitecustomize-style import that runs at
   `import vllm.model_executor.layers.quantization` time, gated by
   `VRM_EXPERT_PIN_HISTOGRAM`. We add a tiny `vrm_expert_pin_init.py` and
   inject it into the worker via `PYTHONSTARTUP` or `python -c` shim launched
   from the bench. Simpler: register the patch in
   `core/inference_pipeline.py:_load_with_vllm()` **before**
   `LLMEngine.from_engine_args` returns control, by setting
   `os.environ["VRM_EXPERT_PIN_INIT"]="1"` and importing a tiny shim in the
   worker boot path via `vllm`'s `init_logger` hook.

For Phase B-1, the simplest robust path: **patch
`vllm.model_executor.layers.quantization.awq_marlin` and
`vllm.model_executor.layers.quantization.fp8` from inside
`core/backends_vllm.py:vLLMBackend.load_model`, AFTER
`LLMEngine.from_engine_args` returns the engine but BEFORE generation starts**
— but this only works if the engine exposes the worker's quant methods. It
does not in V1. So fallback is: **set up a `vllm.plugins`-style hook** by
writing a tiny entry-point package OR — simplest of all — **launch the worker
with `PYTHONSTARTUP=path/to/vrm_pin_init.py`**, and that file imports
`core.expert_pinning` and registers the create_weights wrappers in the same
process.

We will validate the patch fired by asserting
`getattr(layer, "_vrm_hot_mask", None) is not None` for at least one layer
once the engine has loaded.

## C. Forward routing

Two implementable strategies; we adopt strategy **(2) per-call scratch buffer**
for Phase B-1 because it's the only one that does not require kernel surgery.

### Strategy 1: Implicit cross-device tensor view (rejected for B-1)
Rely on `can_device_access_peer = True` so a tensor on `cuda:1` can be read by a
kernel on `cuda:0`. **In practice this does NOT work** for `fused_marlin_moe`
and `cutlass_fp8_moe` because:
- Triton/CUTLASS kernels assume a single CUDA stream tied to one device;
  cross-device pointer dereference triggers per-access PCIe roundtrip with no
  coalescing.
- The kernel launches on `cuda:0`'s stream; a `cuda:1` source pointer compiles
  but every grid block stalls on PCIe latency (~1.5 µs per access).
- Empirically the same issue killed the V6.D weight-prefetch scheme.

Keep this in the journal as a "tested and discarded" path.

### Strategy 2: Explicit per-call scratch on GPU 0 (chosen)
At forward time, before `quant_method.apply()` runs:
1. Read `topk_ids` (shape `(num_tokens, topk)`, ints on `cuda:0`).
2. Compute the **set of cold experts hit by this batch** with
   `cold_hits = torch.unique(topk_ids[~layer._vrm_hot_mask[topk_ids]])`. For
   batch=1, decode-phase Qwen3 with 8 experts/token → at most 8 unique cold
   experts per layer. Empirical worst case ≤ 12.
3. For each cold expert `e` in `cold_hits`, copy
   `layer._vrm_cold_w13[remap[e]]` (on cuda:1) → `layer.w13_qweight[e]` (on
   cuda:0) via async `copy_()` on a dedicated stream. Same for `w2`, scales,
   zeros.
4. Synchronize the prefetch stream against the compute stream via a CUDA event
   before `quant_method.apply()` runs.
5. Run the unmodified kernel — it sees a fully populated `w13_qweight[e]` for
   all experts hit this token.

The kernel sees a "lazily filled" `(num_experts, ...)` tensor: hot rows are
always valid, cold rows are either zero (slow path: kernel computes against
zero, output drops to zero) or filled-on-demand. **Critical**: we MUST stage
cold rows BEFORE the kernel reads them.

### Per-call stage cost calculation
Per layer, per token (decode):
- topk=8 experts, ~44% of them are cold on average → 3.5 cold experts per layer
  per token.
- Each cold expert weight set: w13 (`hidden=2048, inter=768/AWQ pack 8 →
  393K int32 = 1.5 MB`), w2 (~0.75 MB), scales ~tiny → ~2.5 MB / expert.
- 48 layers × 3.5 cold × 2.5 MB = ~420 MB / token at 11.7 GB/s = **36 ms/token
  = 28 tok/s peak data-bound**.
- Add kernel time (5070 Ti FP8 / Int4 path ~1-2 ms/layer × 48 = 50-100 ms).
- Realistic projection: **15–18 tok/s** (PCIe and compute can partially overlap
  with cuda graphs / multi-stream).

This matches the journal's section 7 prediction (15–20 tok/s).

### Optimizations layered later (Phase B-2)
- **Lookahead prefetch**: while layer N's compute runs, prefetch layer N+1's
  expected cold experts. Requires a (cheap) router-prediction prepass —
  feasible because the router is just `softmax(W_router @ x)`, fast on cuda:0.
- **Cold-expert caching**: keep an LRU on cuda:0 of the last K cold experts
  streamed in; many tokens hit the same cold expert across consecutive layers.
- **Pinned host fallback**: if cold expert > scratch capacity, fall back to
  UVA DRAM (V6.D path).

## D. Lending pool wiring

### Persistent materialization
The journal's section 3 documents that materializing a persistent buffer on the
3090 in the parent process leaks device preference into the spawned vLLM worker
(the worker elects cuda:0 == 3090 because its NVML probe sees it has more free
VRAM after parent's allocation lives there). The fix: **allocate the persistent
tensor in the WORKER process, not the parent**.

The worker, with `CUDA_VISIBLE_DEVICES=0,1` (we expand visibility for B-1 — see
section E), allocates the staging buffer on its `cuda:1` once
`torch.cuda.set_device(0)` has been called for compute. This bypasses the
original placement bug because the parent process never touches GPU 1 directly.

### Allocation shape
A single flat `torch.empty(N_BYTES, dtype=torch.uint8, device="cuda:1")` of
size = sum-of-cold-expert-bytes-across-layers. Roughly:
- AWQ: 30 GB total / 128 experts × 103 cold/128 × 48/48 layers ≈ **24 GB** of
  cold experts in raw "compressed-on-disk" form. After Marlin repack, similar
  order. This BUSTS the 12 GB lending buffer assumption.
- FP8: same arithmetic, ~24 GB. Even closer to 3090's 24 GB cap.

**Decision**: bump default `VRM_BENCH_LEND_GB` from 12 to 18 GB, and cap
`VRM_EXPERT_PIN_TOPK_PCT` such that cold footprint ≤ lending budget. With
topk_pct=30% (top 38/128), cold footprint drops to ~24 GB × (90/128) = ~17 GB.
Workable. Document the trade-off: more topk_pct → more VRAM on cuda:0 → less to
spare for KV cache; less topk_pct → more cold traffic → lower tok/s.

### Pool API extension
Extend `core/vram_lending.py:VRAMLendingPool` with one new method:

```python
pool.materialize_lease(lease, total_bytes) -> torch.Tensor
```
- Asserts the lease is ACTIVE.
- Allocates a `torch.empty(total_bytes, dtype=torch.uint8, device=cuda:lender)`
  and stores it in `lease.tensor_ref`.
- Sets `lease.staging_materialized = True` (new attribute on `VRAMLease`).
- Returns the tensor.

Caller then slices it per layer / per cold expert via
`tensor[start:end].view(dtype).reshape(shape)`. We expose a thin helper:
```python
pool.slice_for_expert(lease, layer_idx, expert_local_idx, shape, dtype) -> torch.Tensor
```
which maintains an internal offset map (a layer × cold_expert × {w13, w2,
scales, zeros, ...} → byte_offset registry).

### Bench wiring
The bench creates and registers the lending pool in the **parent**, but
`materialize_lease` is called in the **worker** (or the bench delegates the
materialization to a callable hook that runs at worker start). For B-1 simplest:
run the entire bench as a single process with `CUDA_VISIBLE_DEVICES=0,1` (no
spawn separation), and use vLLM's `enforce_eager=True` plus
`tensor_parallel_size=1` to stay in-process. The journal's spawn argument
applies when the parent has done CUDA work on GPU 1; if we keep parent-side
allocations on GPU 0 only and let the worker do the GPU 1 staging, we sidestep
the issue.

## E. VRAMancer surface

New env vars (consistent with the existing `VRM_*` namespace):

| Var | Default | Purpose |
|---|---|---|
| `VRM_EXPERT_PIN_HISTOGRAM` | unset | Path to histogram JSON. If set, enables the pinning path. |
| `VRM_EXPERT_PIN_TOPK_PCT` | `20` | Top-K percentile to consider hot (drives cold footprint). |
| `VRM_EXPERT_PIN_CACHE` | `benchmarks/results/hot_experts.json` | Where to write the derived registry. |
| `VRM_EXPERT_PIN_LENDER_GPU` | `1` | GPU index that hosts the cold-expert lending tensor. |
| `VRM_EXPERT_PIN_PREFETCH_LOOKAHEAD` | `0` | Phase B-2 toggle for layer N+1 prefetch. |
| `VRM_EXPERT_PIN_LOG_LEVEL` | `INFO` | Logger noise level for hot/cold decisions. |

Also add a new wrapper kwarg to `core/backends_vllm.py:vLLMBackend.load_model`:
- `expert_pin_histogram: Optional[str]` — if set, triggers the hook
  installation before engine creation.

The wrapper auto-detects: if `expert_pin_histogram` is set AND
`tensor_parallel_size=1` AND >=2 GPUs are visible, install the pinning hooks.
Otherwise log a one-line "expert pinning disabled (reason)".

The `VRM_VLLM_TARGET_GPU` semantics are preserved: if pinning is on,
`VRM_VLLM_TARGET_GPU=0` pins compute to GPU 0 and pinning treats GPU
`VRM_EXPERT_PIN_LENDER_GPU=1` as the cold-expert host, and we leave
`CUDA_VISIBLE_DEVICES` unset (workers see both GPUs). Compute placement still
relies on `torch.cuda.set_device(0)` at worker init.

## F. Bench structure: `benchmarks/bench_qwen3_coder_pinning.py`

Mirrors `bench_qwen3_coder_lending.py` (V6.D); only the differences are listed.

### Setup section diff
1. Same `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
   `VLLM_WORKER_MULTIPROC_METHOD=spawn`.
2. **Do NOT narrow `CUDA_VISIBLE_DEVICES`** before `pipeline.load()`. Both GPUs
   must be visible to the worker so it can allocate the cold-expert lending
   tensor on GPU 1.
3. Set
   `VRM_EXPERT_PIN_HISTOGRAM=benchmarks/results/qwen3_coder_expert_histogram.json`
   and `VRM_EXPERT_PIN_TOPK_PCT=20`.
4. Setup the lending pool with `staging_materialized=True` semantics: register
   both GPUs, then explicitly materialize a flat 18 GB `torch.empty(...)` on
   `cuda:1` AT WORKER STARTUP via the install hook, NOT in the parent.

### Inference loop diff
- Same 3-context grid `[512, 1024, 2048]` × `MAX_NEW=32`.
- Add a per-step counter for `cold_experts_streamed_in` (instrumented by the
  create_weights wrapper) and `bytes_p2p_per_token` (sum of cold copies).
- Emit a Markdown summary comparing:
  - V6.D FP8 lending baseline: 5 tok/s (read from
    `bench_qwen3_coder_lending_v6.json`)
  - AWQ baseline (single GPU): 30 tok/s (read from
    `bench_qwen3_coder_awq_baseline_v6.json`)
  - Speculative AWQ (current SOTA): 55 tok/s
  - **V6.E FP8 with pinning**: target 15–20 tok/s
- Same placement guard (`compute Δ > 1 GB` AND `lender Δ > 12 GB now` since we
  materialize the buffer there). Adjust threshold for the new "lender holds
  18 GB" assertion.

### Outputs
- `benchmarks/results/bench_qwen3_coder_pinning_v6.json`
- `benchmarks/results/bench_qwen3_coder_pinning_v6.md`
- `benchmarks/results/bench_qwen3_coder_pinning_vllm.log`

### Honest-claims requirement
Following the `feedback_honest_bench_claims.md` memory: the bench MUST verify
and dump:
- VRAM Δ on GPU 1 ≥ 17 GB (proves cold experts actually live there).
- Counter of cold experts streamed during inference > 0 (proves the forward
  hook fired).
- If either assertion fails, write `note: "ABORTED — pinning hook did not fire"`
  and refuse to claim a tok/s figure.

## G. Risks and mitigations

### Risk 1: vLLM 0.20.1's CUDA-graph capture conflicts with on-the-fly tensor mutation
**Likelihood**: high (when `enforce_eager=False`).
**Symptom**: cold rows of `w13_qweight` are captured at zero, every cold-expert
dispatch returns garbage.
**Mitigation**: pin `enforce_eager=True` in the bench (consistent with V6.D —
already set there). Phase B-2 future work: write a custom CUDAGraph that
includes the prefetch stream, using `torch.cuda.graph` with `pool` argument.
Acceptable to lose some perf in B-1 to keep correctness deterministic.

### Risk 2: AWQ Marlin repack happens on the FULL `(num_experts, ...)` tensor and assumes contiguity
**Likelihood**: confirmed (line 637, `awq_marlin_moe_repack` op).
**Symptom**: if we mirror cold rows to GPU 1 BEFORE repack, the cuda:1 mirror
has un-repacked layout and the kernel reads garbage.
**Mitigation**: mirror cold rows AFTER `process_weights_after_loading` returns
(i.e. after Marlin repack at line 655). Wrap
`AWQMarlinMoEMethod.process_weights_after_loading`, call original, then run our
cold-mirror copy. Sequence: original_create_weights → vLLM weight loader fills
tensor → original_process_weights_after_loading repacks → our hook mirrors cold
rows to cuda:1 and zeros cold rows on cuda:0.

### Risk 3: Spawn worker doesn't inherit our monkey-patches
**Likelihood**: confirmed by the journal (V6.D).
**Symptom**: hooks never fire; bench placement guard catches it but tok/s is
whatever V6.D was.
**Mitigation**: package the install logic in `core/expert_pinning.py:install()`
and trigger it via `PYTHONSTARTUP` env var pointed at a 4-line bootstrap file
`core/_vllm_worker_pin_init.py`. The bench sets
`os.environ["PYTHONSTARTUP"]=str(repo_root / "core" / "_vllm_worker_pin_init.py")`
BEFORE `pipeline.load()`. This file imports `core.expert_pinning` and calls
`install()`, gated by `VRM_EXPERT_PIN_HISTOGRAM` env presence. PYTHONSTARTUP is
honored by `python -c`-launched child processes only when interactive — for
spawn, use `multiprocessing.spawn`'s `_check_not_importing_main` is bypassed by
setting `VLLM_WORKER_MULTIPROC_METHOD=spawn` AND adding our patch via vLLM's
plugin system. **Cleanest path**: register a `vllm.platform_plugins` entry
point in `pyproject.toml` that points to our hook. vLLM 0.20.1 imports plugins
very early in worker startup. This is robust and survives spawn. Validated by
the patch firing assertion in section F.

### Risk 4: Persistent 18 GB allocation on GPU 1 breaks P2P validation transfers
**Likelihood**: medium. The V6.D `measure_lending_p2p_throughput` allocates a
256 MB transient buffer on GPU 1 — if GPU 1 has only ~6 GB free after the
persistent 18 GB buffer plus other system overhead, a transient 256 MB still
fits, but if we bump default to bigger we may OOM the 3090.
**Mitigation**: do P2P validation BEFORE materializing the persistent buffer
(current bench order respects this; preserve it). Keep a
`LendingPolicy.min_free_ratio=0.10` margin (2.4 GB on the 3090) — after 18 GB
persistent, free = ~6 GB minus the policy reserve = ~3.6 GB usable for
transient probes. Plenty for 256 MB.

### Risk 5: AWQ is single-pack, not per-expert addressable in some Marlin variants
**Likelihood**: low for `awq_marlin_moe_repack` (confirmed expert-batched
output) but possible for fused variants.
**Symptom**: `replace_parameter(layer, "w13_qweight", marlin_w13_qweight)`
outputs a tensor whose layout doesn't admit `[expert_id, :, :]` slicing without
extra repack.
**Mitigation**: `awq_marlin_moe_repack` returns shape `(num_experts, …)` per
the op signature (line 637-644). Verified by reading the source. If it changes
in a future vLLM version (out of scope here per constraints), we adapt then.

### Risk 6: Cold-expert prefetch latency spikes hurt p99 even if mean tok/s improves
**Likelihood**: medium. A token that needs 8 cold experts out of 8 (worst case,
~5% of tokens by Pareto) pays 8× the per-expert PCIe latency.
**Mitigation**: instrument per-token tok/s and report p50/p99 in the bench
output. If p99 is catastrophic, B-2 can add the LRU on hot cold-experts.

## H. Phasing and milestones

### Phase B-1: Minimum viable pinning (1.5–2 days)
**Milestone**: bench runs end-to-end, FP8 Qwen3-Coder-30B-A3B yields ≥ 10 tok/s
with proven cold-expert P2P traffic > 0.
- B-1.1: Implement `core/expert_pinning.py` with `compute_hot_from_histogram`,
  `load_hot_registry`, and the
  create_weights/process_weights_after_loading wrappers. **Unit-testable**
  without a model: feed a fake `FusedMoE` layer with `(128, 2048, 96)` int32
  dummies, assert the hot/cold split matches `hot_experts.json`.
- B-1.2: Implement the per-forward staging hook. For Phase B-1, **do this in
  eager mode** by patching `MoERunner.forward` at
  `vllm/model_executor/layers/fused_moe/runner/moe_runner.py:531` with a
  wrapper that stages cold experts before delegating to original. **Stop-gate**:
  confirm the wrapper fires on every layer for every step.
- B-1.3: Extend `core/vram_lending.py:VRAMLendingPool` with
  `materialize_lease(lease, n_bytes)` and `slice_for_expert(...)`. Keep API
  additive so existing callers (KV cache, paged attention) are unaffected.
- B-1.4: Write `benchmarks/bench_qwen3_coder_pinning.py`. Run it. Fix any
  spawn-worker issues until the placement guard passes AND the
  cold-expert-streamed counter is non-zero.
- B-1.5: Capture results, write the MD summary, commit. **Early-stop here** if
  measured tok/s ≥ 12. The user wants something measurable before polish.

### Phase B-2: Lookahead and CUDA graphs (1–2 days, only if B-1 hits ≥ 12 tok/s)
**Milestone**: tok/s climbs to 15–20.
- B-2.1: Implement layer N+1 router pre-prediction (run the gate's softmax for
  the next layer's hidden states using current activations as a cheap proxy).
  Issue prefetch async on a dedicated stream.
- B-2.2: Investigate `enforce_eager=False` — write a CUDA-graph-safe variant of
  the staging hook. May require keeping cold rows as a fixed-size scratch
  updated outside the graph.
- B-2.3: Add LRU cache for last-K cold experts to skip redundant streams.
- B-2.4: Re-run bench. Update MD.

### Phase B-3 (stretch, optional): hot/cold split for AWQ as well
Run the same scheme on the AWQ checkpoint (faster steady-state but tighter
VRAM budget). Verify gain composes with speculative decoding (target: 55 tok/s
spec + pinning → 65–70 tok/s on FP8 effective quality).

---

## Open questions to spike before commit

1. **Plugin entry-point timing**: confirm `vllm.platform_plugins` entry points
   fire BEFORE `vllm.model_executor.layers.quantization.awq_marlin` is imported
   by the worker. 30-minute spike: drop a `print("PIN INIT")` in a fake plugin
   and watch the worker log order.
2. **AWQ `awq_marlin_moe_repack` output stride**: confirm
   `marlin_w13_qweight.is_contiguous()` returns True and
   `marlin_w13_qweight[expert_id]` is a clean stride-0 slice. 10-minute spike
   with a tiny model.
3. **Cold-mirror byte budget at Marlin-repacked layout**: read 1 layer
   post-repack, `sys.getsizeof(marlin_w13_qweight)`, multiply by
   48 × cold_fraction. If > 18 GB, lower `VRM_EXPERT_PIN_TOPK_PCT` default to
   25–30.
4. **Multi-stream P2P scaling**: does using two streams for w13 + w2 prefetch
   in parallel double effective bandwidth, or is the PCIe link the bottleneck?
   20-minute microbench.

---

## Critical Files for Implementation

- `/home/jeremie/VRAMancer/VRAMancer/core/expert_pinning.py` (new module —
  registry + create_weights/runner.forward hooks)
- `/home/jeremie/VRAMancer/VRAMancer/core/vram_lending.py` (extend with
  `materialize_lease` and `slice_for_expert` around line 573 `allocate_on_lease`)
- `/home/jeremie/VRAMancer/VRAMancer/core/backends_vllm.py` (wire
  `expert_pin_histogram` kwarg + plugin/PYTHONSTARTUP install around
  `load_model` lines 24–82)
- `/home/jeremie/VRAMancer/VRAMancer/benchmarks/bench_qwen3_coder_pinning.py`
  (new bench, fork of `bench_qwen3_coder_lending.py`)
- `/home/jeremie/VRAMancer/VRAMancer/benchmarks/results/qwen3_coder_expert_histogram.json`
  (input data — read-only)
