# VRAMancer Roadmap

## Completed (v1.5.0)

### NVFP4 Performance
- [x] Identify bottleneck (`torch._scaled_mm` overhead on M=1 / Decode)
- [x] Implement `core/triton_gemv_nvfp4.py` — custom LUT kernel, nibble unpacking, per-device cache
- [x] Implement `core/nvfp4_direct.py` — DirectFP4 bypass (plain buffers + `_scaled_mm`), +7% vs torchao
- [x] M=1 decode: Triton GEMV auto-selected, fallback to cuBLAS for batch > 1

### TurboQuant KV Compression
- [x] `core/turboquant.py` — Walsh-Hadamard, recursive polar, QJL 1-bit, asymmetric estimator
- [x] Wired into `core/paged_attention.py` — compress-on-write, compress-on-evict, `compute_attention_turbo()`
- [x] Env vars `VRM_KV_COMPRESSION=turboquant`, `VRM_KV_COMPRESSION_BITS=3`
- [x] ~3.5 bits/dim, ~4.6x KV reduction, 0 regressions (821 tests pass)

### ReBAR Full-Window Transfers
- [x] Detection: sysfs BAR0 size, activation threshold > 4 GB
- [x] `ReBarTransport.transfer()` — Rust async pipeline / PipelinedTransport with BAR-optimal chunks
- [x] Wired as explicit Strategy 1.7 in TransferManager (between Rust DtoD and plain CPU-staged)

### VRAM Lending
- [x] `core/vram_lending.py` — lease state machine, scoring, borrow/reclaim/release
- [x] Wired into `inference_pipeline.py` — register_gpu at load, background monitoring
- [x] Wired into `paged_attention.py` — overflow borrow, lease release on eviction
- [x] Env vars: `VRM_LEND_RATIO`, `VRM_RECLAIM_THRESHOLD`, `VRM_LENDING_INTERVAL`

---

## Future Roadmap

### P1 — Wire Existing Code (DONE — all 6 items completed)

- [x] **Tensor Parallel default activation** — Wired into `inference_pipeline.py` via `VRM_PARALLEL_MODE=tp`. Step 6b in load() calls `apply_tensor_parallel()`, generate/infer use TP model when active. PP split skipped when TP.
- [x] **VRAM Lending stress test** — `tests/test_lending_stress.py` (6 tests): concurrent borrow/release (8 threads), borrow+reclaim races, budget consistency (100 cycles), exhaust/recover, close cleanup, 16-thread deadlock detection. All pass.
- [x] **Dashboard CLI fix** — Port 8000→5030, auth token header, correct GPU/status response parsing, added `/api/pipeline/status` section.
- [x] **Dashboard web real data** — Fixed `/api/swarm/status` for missing metrics, added `/api/gpu` endpoint with real torch data, added `/api/pipeline/status` connecting to global pipeline.
- [x] **batch_inference real batching** — `_do_batch()` in production_api batch endpoint now uses `continuous_batcher.submit()` for concurrent processing when batcher running, falls back to sequential otherwise.
- [x] **Swarm Ledger integration** — Wired into `block_orchestrator.py`: `ledger.reward_node()` on successful network migration, `blocks_processed`/`ledger_available` in `get_state()`.

### P2 — Implement Missing Features (DONE — all 6 items completed)

- [x] **DMA-BUF cross-vendor zero-copy** — `csrc/dmabuf_bridge.c` C extension with DRM PRIME ioctls (`drmPrimeHandleToFD`/`drmPrimeFDToHandle`). `DMABufTransport.transfer()` wired: native probe → open → transfer → close, CUDA IPC fallback. Makefile target `native-dmabuf`.
- [x] **ReBAR mmap direct VRAM access** — `csrc/rebar_mmap.c` C extension for PCIe BAR0 mmap (WC mapping, read/write/copy). `ReBarTransport` loads native lib, opens mappings per GPU BDF. Makefile target `native-rebar`.
- [x] **WebGPU Swarm compute** — `backends_webgpu.py` `generate()` implements draft-verify speculative decoding. `_verify_speculative_tokens()` re-runs each draft token through sequential forward pass, accepts only verified prefix.
- [x] **WebGPU Node browser runtime** — `web/vramancer_worker.js` (VRAMancerWorker class with tiled 16×16 WGSL GEMM shader, WebSocket binary protocol, capability reporting) + `web/index.html` (browser UI with connect/disconnect, stats, log).
- [x] **VTP Network Protocol** — `VTPServer._handle_client()` rewritten: full CONTROL-opcode handshake (JSON node info), `_recv_loop()` (HEARTBEAT ack, TENSOR/KV_CACHE decode+store, credit grants). `LLMTransport.connect_peer_tcp()` performs client-side handshake.
- [x] **Parity recovery in PagedAttention** — `_evict_lru()` encodes page data via `parity_kv.store_engram()` (3 XOR shards) before eviction. `recover_page()` method heals evicted pages from parity engrams.

### P3 — Dead Code Cleanup (DONE — all 3 items completed)

- [x] **Move to `_deprecated/`**: `network/packets.py` (dead, 0 imports), `network/interface_selector.py` (dead, 0 imports), `network/vramancer_link.py` (7-line re-export shim, inlined in block_orchestrator.py → `from core.network.transmission`). `core/telemetry.py` KEPT (active in supervision_api, telemetry_cli, tests). `network/resource_aggregator.py` already deprecated prior session.
- [x] **Fix or remove**: `network/interface_selector.py` moved to `_deprecated/`. `network/security.py` KEPT (test-only, logging bug already fixed prior session). Fixed `core/network/__init__.py` (removed interface_selector import) and `tests/test_imports.py`.
- [x] **Clean dead code in `inference_pipeline.py` + `backends.py`**: Removed WebGPUNodeManager init/assignment/shutdown (~30 LOC), orphaned Connectome startup, and 2 WebGPU intercept blocks in backends.py (~100 LOC). Total ~150 LOC removed.

### P4 — Performance & Research (DONE — all 5 items completed)

- [x] **CUDA Graph capture for decode** — `core/cuda_graph_decode.py`: `CUDAGraphRunner` with per-batch-size graph cache, warmup-then-capture, automatic replay. Wired into `inference_pipeline.py` (`_init_cuda_graph_runner`, `_protected_infer`). Opt-in via `VRM_CUDA_GRAPH=1`. Skipped for vLLM/Ollama/llama.cpp backends.
- [x] **Speculative decoding auto-tuning** — `SwarmSpeculativeDecoder` now has `adaptive=True` (default), rolling window of last N rounds, adjusts gamma between `gamma_min=2` and `gamma_max=12`. High acceptance (>80%) → increase K, low (<30%) → decrease. Env: `VRM_SPEC_ADAPTIVE=0` to disable, `VRM_SPEC_WINDOW=10`.
- [x] **GGUF multi-GPU** — `_compute_tensor_split()` now weights by VRAM × compute capability (SM count × clock). 3-level fallback: hetero_config → `torch.cuda.mem_get_info` → pynvml. `_select_split_mode()` auto-detects P2P via `can_device_access_peer` → row split (mode 2) if available, layer split (mode 1) otherwise.
- [x] **TurboQuant CUDA kernel** — `paged_attention_decode_q4_kernel` in `csrc/paged_attention_kernel.cu`: inline 4-bit dequantization (packed uint8 nibbles, per-group fp16 scale/zero-point, group_size=32) fused with warp-level online softmax. Python wrapper `paged_attention_decode_q4()` in `paged_attention_cuda.py` with PyTorch fallback.
- [x] **Continuous batching async tokenizer** — `ContinuousBatcher` Phase 1b now uses `ThreadPoolExecutor` (default 4 workers, `VRM_TOKENIZER_WORKERS` env) for parallel tokenization. Falls back to sequential for single requests. Pool shutdown in `stop()`.

### P5 — Infrastructure (DONE — all 4 items completed)

- [x] **Rust vramancer_rust CI** — Enhanced `build-rust.yml`: added `lint` job (cargo fmt + clippy), `test` job (cargo test), Rust toolchain setup (dtolnay/rust-toolchain), Python import verification after wheel build. Build-wheels now depends on lint+test passing.
- [x] **Schema versioning** — `core/persistence.py`: added `schema_version` table, `CURRENT_SCHEMA_VERSION=2`, `_get_schema_version()`, `_apply_migrations()` with sequential migration system. V2 adds `created_at` column to workflows. Legacy v1 databases auto-migrated. `get_schema_version()` public API.
- [x] **Config hot-reload subsystem restart** — `core/config.py`: added `register_reload_hook(fn)`, `unregister_reload_hook(fn)`. `reload_config()` now calls all hooks with `(old_config, new_config)`. Hooks called outside lock, exceptions caught per-hook. Duplicate registration prevention.
- [x] **Metrics lifecycle** — `core/metrics.py`: added `reset_metrics()` that clears labeled gauges (removes stale GPU label sets) and zeros simple gauges. 40+ metrics tracked in `_LABELED_GAUGES`/`_SIMPLE_GAUGES`. Wired into `InferencePipeline.shutdown()`. Counters/Histograms left monotonic per Prometheus convention.

### Post-Audit — Honest Naming & UX (DONE — all 4 items completed)

- [x] **One-command CLI** — Added `vramancer run <model>` command: auto-detects GPUs, loads model with VRAM-proportional split, supports interactive mode (REPL) and one-shot (`-p "prompt"`). Options: `--gpus`, `-q` (quantization), `--backend`, `--temperature`, `--max-tokens`.
- [x] **README rewrite** — Complete rewrite focused on the core use case. 10-line quickstart showing `vramancer run`, real benchmark table (14B OOM proof), install/usage/backends/config sections. Replaced marketing D- README with honest B+ documentation.
- [x] **Rename marketing modules** — `core/turboquant.py` → `core/kv_quantizer.py` (`KVCacheCompressor` class), `core/network/fibre_fastpath.py` → `core/network/network_transport.py`. Backward-compat shims in old locations. All 20+ imports updated in production code and tests. Env var `VRM_KV_COMPRESSION=turboquant` preserved for backward compat.
- [x] **Benchmark vs accelerate** — `benchmarks/bench_vs_accelerate.py`: side-by-side comparison of `accelerate device_map="balanced"` vs VRAMancer VRAM-proportional split. Measures tok/s, VRAM usage, supports quantization modes. JSON results output.

