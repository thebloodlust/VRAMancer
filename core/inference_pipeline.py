"""VRAMancer Inference Pipeline — the conductor that wires all components.

This module connects:
  - Backend (HuggingFace / vLLM / Ollama) → model loading + text generation
  - Scheduler → VRAM-aware block allocation
  - ModelSplitter → multi-GPU model partitioning
  - TransferManager → GPU-to-GPU activation transfer
  - StreamManager → layer prefetching and swapping
  - ComputeEngine → layer execution with profiling
  - ClusterDiscovery → network node detection
  - Metrics → Prometheus instrumentation

Usage:
    pipeline = InferencePipeline()
    pipeline.load("gpt2", num_gpus=2)
    result = pipeline.generate("Hello, world!", max_new_tokens=50)
    pipeline.shutdown()

    # Or via the global singleton:
    from core.inference_pipeline import get_pipeline
    pipe = get_pipeline()
    pipe.load("gpt2")
    logger.info(pipe.generate("Once upon a time"))
"""

from __future__ import annotations

import os
import sys
import time
import logging
import threading
from typing import Any, Dict, List, Optional
from contextlib import contextmanager

_logger = logging.getLogger("vramancer.pipeline")
_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")


@contextmanager
def _nullcontext():
    """Minimal no-op context manager (stand-in when OTEL is off)."""
    yield None

# Conditional imports — never crash on missing dep
try:
    import torch
    _TORCH = True
except ImportError:
    torch = None  # type: ignore
    _TORCH = False

# GPU Fault Tolerance
try:
    from core.gpu_fault_tolerance import (
        GPUFaultManager, get_fault_manager, reset_fault_manager,
        GPUHealth, FaultType,
    )
    _FAULT_TOLERANCE = True
except ImportError:
    _FAULT_TOLERANCE = False

try:
    from core.metrics import (
        INFER_REQUESTS, INFER_ERRORS, INFER_LATENCY, GPU_MEMORY_USED,
        metrics_server_start,
    )
    _METRICS = True
except ImportError:
    _METRICS = False

# Optional OpenTelemetry tracing
_OTEL = False
_tracer = None
try:
    if os.environ.get("VRM_TRACING") == "1":
        from opentelemetry import trace
        _tracer = trace.get_tracer("vramancer.pipeline")
        _OTEL = True
except ImportError:
    pass


class InferencePipeline:
    """End-to-end inference pipeline connecting all VRAMancer subsystems.

    Lifecycle:
        1. ``__init__`` — instantiate with optional config overrides
        2. ``load(model_name, num_gpus)`` — load + split model
        3. ``generate(prompt)`` / ``infer(input_ids)`` — run inference
        4. ``shutdown()`` — cleanup resources
    """

    def __init__(
        self,
        backend_name: str = "auto",
        enable_metrics: bool = True,
        enable_discovery: bool = False,
        verbose: bool = True,
    ):
        self.backend_name = backend_name
        self.verbose = verbose
        self._loaded = False
        self._lock = threading.Lock()

        # Subsystem references (initialized lazily)
        self.backend = None
        self.scheduler = None
        self.transfer_manager = None
        self.stream_manager = None
        self.compute_engine = None
        self.discovery = None
        self.monitor = None
        self.gpu_hotplug = None
        self.continuous_batcher = None
        self.paged_kv = None
        self.lending_pool = None
        self.hierarchical_memory = None
        self.fault_manager: Optional[Any] = None
        self.turbo_engine: Optional[Any] = None  # TurboEngine for compiled decode
        self.cuda_graph_runner: Optional[Any] = None  # CUDA Graph for decode steps
        self.tp_model: Optional[Any] = None  # Tensor Parallel model wrapper
        self._turboquant_cache_factory: Optional[Any] = None  # HF-native TurboQuant cache

        # Dynamic rebalancing
        self._rebalance_thread: Optional[threading.Thread] = None
        self._rebalancing = False
        self._rebalance_interval = float(os.environ.get("VRM_REBALANCE_INTERVAL", "5.0"))

        # Model info
        self.model_name: Optional[str] = None
        self.num_gpus: int = 0
        self.blocks: List[Any] = []

        # Start metrics server
        if enable_metrics and _METRICS:
            try:
                metrics_server_start()
            except Exception as e:
                _logger.debug(f"Exception silencieuse dans l'exécution: {e}", exc_info=True)

        # Start cluster discovery in background
        if enable_discovery:
            self._start_discovery()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        model_name: str,
        num_gpus: Optional[int] = None,
        **model_kwargs,
    ) -> "InferencePipeline":
        """Load a model and prepare for inference.

        Parameters
        ----------
        model_name : str
            HuggingFace model name, or model identifier for vLLM/Ollama.
        num_gpus : int, optional
            Number of GPUs to use. Auto-detected if None.
        **model_kwargs
            Additional kwargs passed to the backend's load_model().

        Returns
        -------
        self
            For chaining: ``pipeline.load("gpt2").generate("Hello")``
        """
        with self._lock:
            _logger.info("Loading model: %s (backend=%s)", model_name, self.backend_name)
            self.model_name = model_name

            # 1. Init scheduler (detects GPUs) first
            from core.scheduler import SimpleScheduler
            self.scheduler = SimpleScheduler(blocks=[])

            gpus = self.scheduler.get_available_gpus()
            if num_gpus is None:
                num_gpus = len(gpus)
            self.num_gpus = min(num_gpus, len(gpus))
            _logger.info("GPUs: %d available, using %d", len(gpus), self.num_gpus)

            # 1b. Auto single-GPU bypass: skip multi-GPU overhead when model fits
            if self.num_gpus > 1:
                self.num_gpus = self._auto_select_num_gpus(
                    model_name, self.num_gpus, model_kwargs,
                )

            # 2. Select backend
            from core.backends import select_backend
            self.backend = select_backend(model_name, backend=self.backend_name, num_gpus=self.num_gpus)

            # 3. Init GPU monitor
            try:
                from core.monitor import GPUMonitor
                self.monitor = GPUMonitor()
            except Exception:
                self.monitor = None

            # 4. Init transfer manager (topology detection, P2P probing)
            try:
                from core.transfer_manager import TransferManager
                self.transfer_manager = TransferManager(
                    protocol="nccl",
                    secure=False,
                    verbose=self.verbose,
                )
            except Exception as e:
                _logger.warning("TransferManager init failed: %s", e)
                self.transfer_manager = None

            # 5. Load model via backend
            try:
                self.backend.load_model(model_name, num_gpus=self.num_gpus, **model_kwargs)
            except Exception as e:
                _logger.error("Model load failed: %s", e)
                raise

            # 6. Inject transfer manager into backend for multi-GPU activation transfer
            if hasattr(self.backend, 'transfer_manager'):
                self.backend.transfer_manager = self.transfer_manager

            # 6b. Tensor Parallel mode (VRM_PARALLEL_MODE=tp)
            # When selected, wraps the loaded model with apply_tensor_parallel()
            # which shards weights across GPUs and uses NCCL all-reduce.
            # Default: "pp" (pipeline parallelism via model_splitter).
            _parallel_mode = os.environ.get("VRM_PARALLEL_MODE", "pp").lower()
            if self.num_gpus > 1 and _parallel_mode == "tp":
                try:
                    from core.tensor_parallel import apply_tensor_parallel
                    model = getattr(self.backend, "model", None)
                    if model is not None and _TORCH:
                        devices = [f"cuda:{i}" for i in range(self.num_gpus)]
                        self.tp_model = apply_tensor_parallel(model, devices=devices)
                        _logger.info(
                            "Tensor Parallel activated: %d GPUs, devices=%s",
                            self.num_gpus, devices,
                        )
                    else:
                        _logger.warning("TP requested but no model on backend — falling back to PP")
                except Exception as e:
                    _logger.warning("Tensor Parallel init failed (%s), falling back to PP", e)
                    self.tp_model = None

            # 7. Split model across GPUs (only in PP mode, skip if TP)
            if self.num_gpus > 1 and self.tp_model is None:
                try:
                    self.blocks = self.backend.split_model(self.num_gpus)
                    _logger.info("Model split into %d blocks", len(self.blocks))
                except Exception as e:
                    _logger.warning("Model split failed (%s), using single-GPU", e)
                    self.blocks = []
            else:
                self.blocks = []
                _logger.info("Single GPU/CPU mode, no split needed")

            # 8. Init stream manager for prefetch
            try:
                from core.stream_manager import StreamManager
                self.stream_manager = StreamManager(
                    scheduler=self.scheduler,
                    monitor=self.monitor,
                    verbose=self.verbose,
                )
            except Exception as e:
                _logger.warning("StreamManager init failed: %s", e)

            # 9. Init compute engine
            try:
                from core.compute_engine import ComputeEngine
                self.compute_engine = ComputeEngine(
                    backend="auto",
                    verbose=self.verbose,
                )
            except Exception:
                self.compute_engine = None

            self._loaded = True
            _logger.info("Pipeline ready: model=%s, gpus=%d, blocks=%d",
                         model_name, self.num_gpus, len(self.blocks))

            # 10. Start GPU hot-plug monitoring
            self._setup_gpu_hotplug()

            # 11. Start dynamic rebalancing if multi-GPU
            if self.num_gpus > 1:
                self.start_rebalancing()

            # 12. Init continuous batcher + paged KV cache
            # Skip when vLLM is the backend — it has its own batching engine
            _backend_type = getattr(self.backend, 'backend_type', '')
            if _backend_type != 'vllm':
                self._init_continuous_batching()

            # 12b. Init TurboQuant HF-native KV cache (if compression requested)
            self._init_turboquant_cache()

            # 13. Init VRAM Lending Pool (cooperative GPU memory)
            # Skip when vLLM/llama.cpp manage their own VRAM
            # Controlled by VRM_VRAM_LENDING env var (default: enabled for multi-GPU)
            _lending_enabled = os.environ.get("VRM_VRAM_LENDING", "1").lower() not in ("0", "false", "no")
            if self.num_gpus > 1 and _backend_type not in ('vllm', 'llamacpp') and _lending_enabled:
                self._init_lending_pool()

            # 13b. Init Hierarchical Memory Manager (L1-L6 tiered memory)
            self._init_hierarchical_memory()

            # 14. Init GPU Fault Tolerance
            self._init_fault_tolerance()

            # 15. Init TurboEngine (compiled decode — ~2x speedup)
            self._init_turbo_engine()

            # 16. Init CUDA Graph runner for decode step acceleration
            self._init_cuda_graph_runner()

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        enable_speculative: bool = False,
        draft_model_callable = None,
        **kwargs,
    ) -> str:
        """Generate text from a prompt.

        Parameters
        ----------
        prompt : str
            Input text prompt.
        max_new_tokens : int
            Maximum number of tokens to generate.
        temperature : float
            Sampling temperature (1.0 = no change).
        top_p : float
            Nucleus sampling threshold.
        top_k : int
            Top-k sampling.
        enable_speculative : bool
            Awaken the Swarm Brain for predictive speculative decoding.
        draft_model_callable : Callable
            A small local model hook for generating draft tokens.

        Returns
        -------
        str
            Generated text.
        """
        self._ensure_loaded()

        # Wake on Inference - Dynamically pull nodes from sleep
        try:
            from core.wake_on_inference import get_woi_manager
            get_woi_manager().wake_all()
        except Exception as e:
            _logger.debug(f"Exception silencieuse dans l'exécution: {e}", exc_info=True)

        if _METRICS:
            INFER_REQUESTS.inc()

        # OpenTelemetry span (no-op if VRM_TRACING!=1)
        span_ctx = (
            _tracer.start_as_current_span(
                "pipeline.generate",
                attributes={
                    "model": self.model_name or "",
                    "max_new_tokens": max_new_tokens,
                    "num_gpus": self.num_gpus,
                },
            )
            if _OTEL and _tracer
            else _nullcontext()
        )

        start = time.perf_counter()
        with span_ctx as span:
            try:
                # Build generation kwargs
                gen_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    **kwargs,
                }
                if temperature != 1.0:
                    gen_kwargs["temperature"] = temperature
                if top_p != 1.0:
                    gen_kwargs["top_p"] = top_p
                    gen_kwargs["do_sample"] = True
                if top_k != 50:
                    gen_kwargs["top_k"] = top_k
                    gen_kwargs["do_sample"] = True
                if temperature != 1.0:
                    gen_kwargs["do_sample"] = True

                # --- TurboQuant KV Cache injection ---
                # If TurboQuant compression is active, create a HF-native
                # TurboQuantCache and inject it as past_key_values so HF's
                # generate() uses compressed KV storage (~4.6x reduction).
                if self._turboquant_cache_factory is not None:
                    gen_kwargs["past_key_values"] = self._turboquant_cache_factory()

                # --- Speculative Decoding ---
                if enable_speculative:
                    from core.speculative_decoding import (
                        SwarmSpeculativeDecoder, create_draft_callable,
                    )
                    # Auto-create draft callable from backend if not provided
                    _draft = draft_model_callable
                    if _draft is None:
                        draft_name = os.environ.get("VRM_DRAFT_MODEL")
                        _draft = create_draft_callable(
                            self.backend,
                            draft_model_name=draft_name,
                            main_model_name=self.model_name,
                        )
                    if _draft is not None:
                        decoder = SwarmSpeculativeDecoder(
                            draft_model_callable=_draft,
                            swarm_verify_callable=self.infer,
                            gamma=int(os.environ.get("VRM_SPEC_GAMMA", "5")),
                            temperature=temperature,
                            adaptive=os.environ.get("VRM_SPEC_ADAPTIVE", "1") != "0",
                        )
                        input_ids = self.backend.tokenizer.encode(
                            prompt, return_tensors="pt",
                        )
                        # Move input_ids to model device
                        _model = getattr(self.backend, "model", None)
                        if _model is not None and hasattr(_model, "device"):
                            input_ids = input_ids.to(_model.device)
                        elif torch is not None and torch.cuda.is_available():
                            input_ids = input_ids.to(f"cuda:{torch.cuda.current_device()}")
                        out_ids = decoder.generate(input_ids, max_new_tokens)
                        result = self.backend.tokenizer.decode(
                            out_ids[0], skip_special_tokens=True,
                        )
                    else:
                        _logger.debug("Speculative decoding requested but no draft model available")
                        result = self._protected_generate(prompt, gen_kwargs)
                elif (self.continuous_batcher is not None
                      and self.continuous_batcher._running):
                    # Route through continuous batcher for automatic
                    # request batching when multiple requests are in flight
                    future = self.continuous_batcher.submit(
                        prompt,
                        max_new_tokens=gen_kwargs.get("max_new_tokens", max_new_tokens),
                        temperature=gen_kwargs.get("temperature", temperature),
                        top_k=gen_kwargs.get("top_k", top_k),
                        top_p=gen_kwargs.get("top_p", top_p),
                    )
                    result = future.result(
                        timeout=float(os.environ.get("VRM_GENERATE_TIMEOUT", "300"))
                    )
                else:
                    # Execute with fault tolerance protection
                    result = self._protected_generate(prompt, gen_kwargs)

                elapsed = time.perf_counter() - start
                if _METRICS:
                    INFER_LATENCY.observe(elapsed)
                if span:
                    span.set_attribute("elapsed_s", round(elapsed, 4))
                    span.set_attribute("result_len", len(result))
                _logger.info("Generated %d chars in %.2fs", len(result), elapsed)

                self._update_gpu_metrics()
                return result

            except NotImplementedError:
                # Backend doesn't support generate() — fall back to infer()
                return self._generate_fallback(prompt, max_new_tokens)
            except Exception as e:
                if _METRICS:
                    INFER_ERRORS.inc()
                if span:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(e))
                _logger.error("Generation failed: %s", e, exc_info=True)
                raise

    def infer(self, input_ids: Any) -> Any:
        """Run raw tensor inference (no tokenization).

        Parameters
        ----------
        input_ids : torch.Tensor or Any
            Token input IDs.

        Returns
        -------
        torch.Tensor or Any
            Model output (logits or hidden states).
        """
        self._ensure_loaded()

        if _METRICS:
            INFER_REQUESTS.inc()

        span_ctx = (
            _tracer.start_as_current_span(
                "pipeline.infer",
                attributes={
                    "model": self.model_name or "",
                    "num_gpus": self.num_gpus,
                },
            )
            if _OTEL and _tracer
            else _nullcontext()
        )

        start = time.perf_counter()
        with span_ctx as span:
            try:
                # Execute with fault tolerance protection
                result = self._protected_infer(input_ids)
                elapsed = time.perf_counter() - start
                if _METRICS:
                    INFER_LATENCY.observe(elapsed)
                if span:
                    span.set_attribute("elapsed_s", round(elapsed, 4))
                self._update_gpu_metrics()
                return result
            except Exception as e:
                if _METRICS:
                    INFER_ERRORS.inc()
                if span:
                    span.set_attribute("error", True)
                    span.set_attribute("error.message", str(e))
                _logger.error("Inference failed: %s", e)
                raise

    # ------------------------------------------------------------------
    # Fault-tolerant execution wrappers
    # ------------------------------------------------------------------

    def _protected_generate(self, prompt: str, gen_kwargs: dict) -> str:
        """Execute generate() with GPU fault protection.

        Prefers TurboEngine (compiled decode, ~2x speedup) when available.
        Falls back to TP model or backend.generate() otherwise.
        """
        # TurboEngine path: compiled Inductor decode (fastest)
        if self.turbo_engine is not None:
            try:
                return self.turbo_engine.generate(
                    prompt,
                    max_new_tokens=gen_kwargs.get("max_new_tokens", 128),
                    temperature=gen_kwargs.get("temperature", 1.0),
                    top_k=gen_kwargs.get("top_k", 0),
                    top_p=gen_kwargs.get("top_p", 1.0),
                    do_sample=gen_kwargs.get("do_sample", False),
                )
            except Exception as e:
                _logger.warning("TurboEngine generate failed (%s), falling back", e)

        # Tensor Parallel path: TP model greedy generation
        if self.tp_model is not None and _TORCH:
            try:
                tokenizer = getattr(self.backend, "tokenizer", None)
                if tokenizer is not None:
                    input_ids = tokenizer.encode(prompt, return_tensors="pt")
                    max_new = gen_kwargs.get("max_new_tokens", 128)
                    out_ids = self.tp_model.generate_greedy(input_ids, max_new_tokens=max_new)
                    return tokenizer.decode(out_ids[0], skip_special_tokens=True)
                else:
                    _logger.warning("TP model active but no tokenizer — falling back to backend")
            except Exception as e:
                _logger.warning("TP generate failed (%s), falling back to backend", e)

        if self.fault_manager and _FAULT_TOLERANCE:
            # Determine primary GPU (device 0 for single-GPU, or first healthy)
            primary_gpu = self._get_primary_gpu()
            try:
                return self.fault_manager.protected_call(
                    gpu_id=primary_gpu,
                    fn=self.backend.generate,
                    args=(prompt,),
                    kwargs=gen_kwargs,
                    retry_on_oom=True,
                    max_retries=2,
                )
            except RuntimeError as e:
                _logger.error("Protected generate failed: %s — trying fallback", e)
                # Last resort: try on any surviving GPU
                return self._generate_on_survivor(prompt, gen_kwargs, exclude=primary_gpu)
        else:
            return self.backend.generate(prompt, **gen_kwargs)

    def _protected_infer(self, input_ids: Any) -> Any:
        """Execute infer() with GPU fault protection."""
        # CUDA Graph path: replay captured graph for decode step
        if self.cuda_graph_runner is not None and _TORCH:
            try:
                return self.cuda_graph_runner.forward(input_ids)
            except Exception as e:
                _logger.debug("CUDA Graph forward failed (%s), falling back", e)

        # Tensor Parallel path: direct forward
        if self.tp_model is not None and _TORCH:
            try:
                return self.tp_model(input_ids)
            except Exception as e:
                _logger.warning("TP infer failed (%s), falling back to backend", e)

        if self.fault_manager and _FAULT_TOLERANCE:
            primary_gpu = self._get_primary_gpu()
            try:
                return self.fault_manager.protected_call(
                    gpu_id=primary_gpu,
                    fn=self.backend.infer,
                    args=(input_ids,),
                    retry_on_oom=True,
                    max_retries=2,
                )
            except RuntimeError as e:
                _logger.error("Protected infer failed: %s", e)
                raise
        else:
            return self.backend.infer(input_ids)

    def _get_primary_gpu(self) -> int:
        """Return the primary GPU for inference (first healthy GPU)."""
        if self.fault_manager and _FAULT_TOLERANCE:
            healthy = self.fault_manager.get_healthy_gpus()
            if healthy:
                return healthy[0]
        return 0

    def _generate_on_survivor(self, prompt: str, gen_kwargs: dict, exclude: int) -> str:
        """Try to generate on any GPU other than the excluded one."""
        if not self.fault_manager or not _FAULT_TOLERANCE:
            raise RuntimeError("No fault manager — cannot find survivor GPU")

        healthy = [g for g in self.fault_manager.get_healthy_gpus() if g != exclude]
        if not healthy:
            raise RuntimeError("All GPUs failed — no survivor available for inference")

        alt_gpu = healthy[0]
        _logger.warning("Attempting generation on survivor GPU %d", alt_gpu)
        return self.fault_manager.protected_call(
            gpu_id=alt_gpu,
            fn=self.backend.generate,
            args=(prompt,),
            kwargs=gen_kwargs,
            retry_on_oom=False,
            max_retries=0,
        )

    # ------------------------------------------------------------------
    # Fault tolerance initialization
    # ------------------------------------------------------------------

    def _init_fault_tolerance(self) -> None:
        """Initialize GPU fault tolerance with block migration support."""
        if not _FAULT_TOLERANCE:
            _logger.debug("GPU fault tolerance not available (import failed)")
            return

        try:
            self.fault_manager = get_fault_manager(num_gpus=self.num_gpus)

            # Register block migration callback
            if self.scheduler and self.transfer_manager:
                self.fault_manager.set_migrate_callback(self._migrate_blocks)

            # Register failure callback — triggers emergency rebalance
            self.fault_manager.on_gpu_failed(self._on_gpu_failure)

            # Register recovery callback — re-register GPU in scheduler
            self.fault_manager.on_gpu_recovered(self._on_gpu_recovery)

            # Register existing blocks with fault manager
            if self.blocks and self.scheduler:
                for i in range(self.num_gpus):
                    block_ids = [
                        idx for idx, b in enumerate(self.blocks)
                        if getattr(b, 'gpu_id', None) == i
                        or (isinstance(b, dict) and b.get('gpu_id') == i)
                    ]
                    if block_ids:
                        self.fault_manager.register_blocks(i, block_ids)

            _logger.info(
                "GPU fault tolerance active: %d GPUs monitored, "
                "max_consecutive_failures=%d, recovery_probe_interval=%.0fs",
                self.fault_manager.num_gpus,
                self.fault_manager.max_consecutive_failures,
                self.fault_manager.recovery_probe_interval,
            )
        except Exception as e:
            _logger.warning("Fault tolerance init failed: %s", e)

    def _init_turbo_engine(self) -> None:
        """Initialize TurboEngine for compiled decode (~2x speedup).

        Uses torch.compile with Inductor kernel fusion on the model body.
        Skipped for vLLM/Ollama backends (they have own optimized runtimes).
        """
        if os.environ.get("VRM_MINIMAL_TEST") or os.environ.get("VRM_DISABLE_TURBO"):
            return

        backend_type = getattr(self.backend, 'backend_type', 'huggingface')
        if backend_type in ('vllm', 'ollama', 'llamacpp'):
            _logger.debug("TurboEngine skipped for %s backend", backend_type)
            return

        model = getattr(self.backend, 'model', None)
        tokenizer = getattr(self.backend, 'tokenizer', None)
        if model is None or tokenizer is None:
            return

        try:
            from core.turbo_engine import create_turbo_engine
            self.turbo_engine = create_turbo_engine(
                model,
                tokenizer,
                transfer_manager=self.transfer_manager,
                max_seq_len=int(os.environ.get("VRM_TURBO_MAX_SEQ", "2048")),
            )
            # Warmup triggers torch.compile JIT (slow first time, fast after)
            self.turbo_engine.warmup()
            _logger.info(
                "TurboEngine active: compiled=%s",
                getattr(self.turbo_engine, '_compiled', False),
            )
        except Exception as e:
            _logger.warning("TurboEngine init failed (%s), using standard decode", e)
            self.turbo_engine = None
            self.fault_manager = None

    def _init_cuda_graph_runner(self) -> None:
        """Initialize CUDA Graph runner for decode step acceleration.

        Captures the model forward as a CUDA graph after warmup,
        eliminating CPU dispatch overhead on repeated decode steps.
        Requires VRM_CUDA_GRAPH=1 (opt-in — fragile with dynamic shapes).
        """
        if not os.environ.get("VRM_CUDA_GRAPH"):
            return
        if os.environ.get("VRM_MINIMAL_TEST"):
            return

        backend_type = getattr(self.backend, 'backend_type', 'huggingface')
        if backend_type in ('vllm', 'ollama', 'llamacpp'):
            return  # these backends manage their own graphs

        model = getattr(self.backend, 'model', None)
        if model is None:
            return

        try:
            from core.cuda_graph_decode import CUDAGraphRunner
            self.cuda_graph_runner = CUDAGraphRunner(
                model=model,
                max_cache_entries=int(os.environ.get("VRM_CUDA_GRAPH_CACHE", "4")),
                warmup_steps=int(os.environ.get("VRM_CUDA_GRAPH_WARMUP", "3")),
            )
            _logger.info("CUDA Graph runner initialized (opt-in, %d cache slots)",
                         self.cuda_graph_runner.max_cache_entries)
        except Exception as e:
            _logger.warning("CUDA Graph runner init failed: %s", e)
            self.cuda_graph_runner = None

    def _migrate_blocks(self, source_gpu: int, target_gpu: int) -> int:
        """Migrate model blocks from a failed GPU to a healthy one.

        Called by GPUFaultManager when a GPU is isolated.
        Returns the number of blocks successfully migrated.
        """
        migrated = 0
        if not self.transfer_manager:
            return migrated

        for idx, block in enumerate(self.blocks):
            block_gpu = getattr(block, 'gpu_id', None)
            if block_gpu is None and isinstance(block, dict):
                block_gpu = block.get('gpu_id')
            if block_gpu != source_gpu:
                continue

            try:
                # Use TransferManager for the actual data transfer
                if _TORCH and hasattr(block, 'data'):
                    self.transfer_manager.transfer(
                        tensor=block.data,
                        src_device=source_gpu,
                        dst_device=target_gpu,
                    )
                # Update block assignment
                if hasattr(block, 'gpu_id'):
                    block.gpu_id = target_gpu
                elif isinstance(block, dict):
                    block['gpu_id'] = target_gpu

                migrated += 1
                _logger.info(
                    "Block %d migrated: GPU %d -> GPU %d",
                    idx, source_gpu, target_gpu,
                )
            except Exception as e:
                _logger.error(
                    "Block %d migration failed (GPU %d -> %d): %s",
                    idx, source_gpu, target_gpu, e,
                )

        # Update fault manager block registry
        if self.fault_manager and _FAULT_TOLERANCE:
            new_blocks = [
                idx for idx, b in enumerate(self.blocks)
                if getattr(b, 'gpu_id', None) == target_gpu
                or (isinstance(b, dict) and b.get('gpu_id') == target_gpu)
            ]
            self.fault_manager.register_blocks(target_gpu, new_blocks)
            self.fault_manager.register_blocks(source_gpu, [])

        return migrated

    def _on_gpu_failure(self, gpu_id: int, fault_type: "FaultType") -> None:
        """Callback when a GPU fails — trigger emergency rebalance."""
        _logger.error(
            "GPU %d FAILED (%s) — triggering emergency rebalance",
            gpu_id, fault_type.name if hasattr(fault_type, 'name') else fault_type,
        )
        # Mark GPU as unavailable in scheduler
        if self.scheduler and hasattr(self.scheduler, 'mark_gpu_unavailable'):
            self.scheduler.mark_gpu_unavailable(gpu_id)

        # Emergency rebalance via stream manager
        if self.stream_manager:
            try:
                self.stream_manager.swap_if_needed()
            except Exception as e:
                _logger.warning("Emergency rebalance failed: %s", e)

    def _on_gpu_recovery(self, gpu_id: int) -> None:
        """Callback when a GPU recovers — re-register in scheduler."""
        _logger.info("GPU %d recovered — re-registering in scheduler", gpu_id)
        if self.scheduler and hasattr(self.scheduler, 'mark_gpu_available'):
            self.scheduler.mark_gpu_available(gpu_id)

    def _generate_fallback(self, prompt: str, max_new_tokens: int) -> str:
        """Fallback text generation for backends without native generate()."""
        _logger.info("Using fallback auto-regressive generation")
        # For vLLM/Ollama, infer() already handles text in/out
        result = self.backend.infer(prompt)
        if isinstance(result, dict):
            return result.get("text", str(result))
        if isinstance(result, str):
            return result
        return str(result)

    # ------------------------------------------------------------------
    # Status / info
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return pipeline status as a dict."""
        result = {
            "loaded": self._loaded,
            "model": self.model_name,
            "backend": type(self.backend).__name__ if self.backend else None,
            "num_gpus": self.num_gpus,
            "num_blocks": len(self.blocks),
            "scheduler": self.scheduler is not None,
            "transfer_manager": self.transfer_manager is not None,
            "stream_manager": self.stream_manager is not None,
            "compute_engine": self.compute_engine is not None,
            "discovery": self.discovery is not None,
            "continuous_batcher": self.continuous_batcher is not None,
            "paged_kv_cache": self.paged_kv is not None,
            "fault_tolerance": self.fault_manager is not None,
            "tensor_parallel": self.tp_model is not None,
            "parallel_mode": "tp" if self.tp_model is not None else "pp",
            "gpus": self.scheduler.get_available_gpus() if self.scheduler else [],
            "transfer_stats": (
                self.transfer_manager.stats()
                if self.transfer_manager else None
            ),
            "batcher_stats": (
                self.continuous_batcher.stats()
                if self.continuous_batcher else None
            ),
        }
        # Fault tolerance details
        if self.fault_manager and _FAULT_TOLERANCE:
            result["fault_stats"] = self.fault_manager.stats()
            result["healthy_gpus"] = self.fault_manager.get_healthy_gpus()
        return result

    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Auto single-GPU bypass
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_model_vram_gb(model_name: str, model_kwargs: dict) -> Optional[float]:
        """Estimate model VRAM footprint (GB) from HuggingFace config.

        Returns None if estimation fails (safe fallback: use multi-GPU).
        Uses local cache first, then a quick network fetch.
        """
        if _MINIMAL:
            return None
        try:
            from transformers import AutoConfig
        except ImportError:
            return None

        config = None
        # Try local cache first (no network = fast)
        try:
            config = AutoConfig.from_pretrained(
                model_name, trust_remote_code=True, local_files_only=True,
            )
        except Exception:
            pass
        # Fallback: quick network fetch (only config.json, small file)
        if config is None:
            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                return None

        # Extract parameter count hints
        num_params = None
        # Some configs expose it directly
        if hasattr(config, 'num_parameters'):
            num_params = config.num_parameters
        # Otherwise estimate from architecture
        if num_params is None:
            # Handle different config styles (Llama/Mistral vs GPT-2 vs others)
            hidden = (
                getattr(config, 'hidden_size', None)
                or getattr(config, 'n_embd', None)
                or getattr(config, 'd_model', None)
            )
            n_layers = (
                getattr(config, 'num_hidden_layers', None)
                or getattr(config, 'n_layer', None)
                or getattr(config, 'num_layers', None)
            )
            vocab = getattr(config, 'vocab_size', None)
            intermediate = (
                getattr(config, 'intermediate_size', None)
                or getattr(config, 'n_inner', None)
                or getattr(config, 'd_ff', None)
            )
            if hidden and n_layers and vocab:
                if intermediate is None:
                    intermediate = hidden * 4
                # Rough transformer param estimate:
                # Each layer: 4*h^2 (attn) + 2*h*ff (mlp) + small
                # Embeddings: vocab * h
                per_layer = 4 * hidden * hidden + 2 * hidden * intermediate
                embeddings = vocab * hidden
                num_params = per_layer * n_layers + embeddings

        if num_params is None:
            return None

        # Bytes per parameter depends on quantization
        quant = os.environ.get("VRM_QUANTIZATION", "").lower()
        name_lower = model_name.lower()
        if quant in ("nvfp4", "nf4") or "nvfp4" in name_lower:
            # NF4: final weights are ~0.56 B/param.  With device_map={"":gpu}
            # + streaming load, BnB quantizes layer-by-layer in-place without
            # buffering all fp16.  14B NF4 → ~10.8 GB final on GPU.
            # Use 0.8 to account for quantization state + overhead.
            bytes_per_param = 0.8
        elif quant in ("int4", "gptq", "awq") or any(
            q in name_lower for q in ("gptq", "awq", "4bit", "int4")
        ):
            bytes_per_param = 0.6
        elif quant in ("int8", "fp8") or any(
            q in name_lower for q in ("int8", "fp8", "8bit")
        ):
            # INT8: final ~1 B/param + outlier columns in fp16 + absmax state.
            # With streaming load, peak is ~1.5 B/param (not full fp16).
            bytes_per_param = 1.5
        else:
            bytes_per_param = 2.0  # fp16/bf16 default

        model_gb = (num_params * bytes_per_param) / (1024 ** 3)
        # Add ~25% for KV cache + activations + overhead
        total_gb = model_gb * 1.25
        _logger.debug(
            "Model size estimate: %s → %.1f B params, %.1f GB weights, "
            "%.1f GB total (%.1f B/param)",
            model_name, num_params / 1e9, model_gb, total_gb, bytes_per_param,
        )
        return total_gb

    def _auto_select_num_gpus(
        self,
        model_name: str,
        num_gpus: int,
        model_kwargs: dict,
    ) -> int:
        """Reduce num_gpus to 1 when the model fits on the largest single GPU.

        Cross-GPU transfers add latency (~10-15 GB/s in VM environments).
        Avoiding them gives a significant speedup for models that fit.
        """
        if os.environ.get("VRM_FORCE_MULTI_GPU") == "1":
            _logger.info("VRM_FORCE_MULTI_GPU=1, keeping %d GPUs", num_gpus)
            return num_gpus

        estimated_gb = self._estimate_model_vram_gb(model_name, model_kwargs)
        if estimated_gb is None:
            _logger.info(
                "Could not estimate model size for %s — using %d GPUs (safe default)",
                model_name, num_gpus,
            )
            return num_gpus

        # Find largest GPU's free VRAM
        max_free_gb = 0.0
        best_gpu = 0
        if _TORCH and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                free_gb = free_bytes / (1024 ** 3)
                if free_gb > max_free_gb:
                    max_free_gb = free_gb
                    best_gpu = i

        # Safety margin: 90% of free VRAM (leave room for driver/runtime)
        usable_gb = max_free_gb * 0.90
        if estimated_gb <= usable_gb:
            _logger.info(
                "Model %s fits on single GPU %d (estimated %.1f GB, "
                "GPU has %.1f GB free → skipping multi-GPU overhead)",
                model_name, best_gpu, estimated_gb, max_free_gb,
            )
            return 1
        else:
            _logger.info(
                "Model %s needs multi-GPU (estimated %.1f GB, "
                "largest GPU has %.1f GB free)",
                model_name, estimated_gb, max_free_gb,
            )
            return num_gpus

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _start_discovery(self):
        """Start cluster discovery in background."""
        try:
            from core.network.cluster_discovery import ClusterDiscovery
            self.discovery = ClusterDiscovery(heartbeat_interval=5)
            self.discovery.start()
            _logger.info("Cluster discovery started")
        except Exception as e:
            _logger.warning("Discovery unavailable: %s", e)

    def get_nodes(self) -> Dict[str, Any]:
        """Return discovered cluster nodes."""
        if self.discovery:
            return self.discovery.get_nodes()
        return {}

    # ------------------------------------------------------------------
    # Dynamic rebalancing
    # ------------------------------------------------------------------

    def start_rebalancing(self, interval: Optional[float] = None) -> None:
        """Start background GPU rebalancing loop.

        Periodically checks for overloaded GPUs and migrates blocks
        to underutilized ones via the StreamManager.
        """
        if self._rebalancing:
            return
        if not self.stream_manager or not self.monitor:
            _logger.debug("Rebalancing requires stream_manager and monitor")
            return

        self._rebalancing = True
        ival = interval or self._rebalance_interval
        self._rebalance_thread = threading.Thread(
            target=self._rebalance_loop,
            args=(ival,),
            daemon=True,
            name="pipeline-rebalance",
        )
        self._rebalance_thread.start()
        _logger.info("Dynamic rebalancing started (interval=%.1fs)", ival)

    def stop_rebalancing(self) -> None:
        """Stop the rebalancing loop."""
        self._rebalancing = False
        if self._rebalance_thread and self._rebalance_thread.is_alive():
            self._rebalance_thread.join(timeout=5)
        _logger.info("Dynamic rebalancing stopped")

    def _rebalance_loop(self, interval: float) -> None:
        """Background loop: detect overload and trigger swaps."""
        while self._rebalancing:
            try:
                if self.monitor and self.stream_manager:
                    overloaded = self.monitor.detect_overload()
                    if overloaded is not None:
                        _logger.warning(
                            "GPU %d overloaded — triggering rebalance", overloaded
                        )
                        swapped = self.stream_manager.swap_if_needed()
                        if swapped:
                            _logger.info("Rebalance: block migrated from GPU %d", overloaded)
                        else:
                            _logger.debug("Rebalance: no eligible block to swap")
            except Exception as exc:
                _logger.debug("Rebalance loop error: %s", exc)
            time.sleep(interval)

    def _setup_gpu_hotplug(self) -> None:
        """Initialize GPU hot-plug monitoring with auto-rebalance."""
        try:
            from core.monitor import GPUHotPlugMonitor
            self.gpu_hotplug = GPUHotPlugMonitor(
                interval=5.0,
                gpu_monitor=self.monitor,
            )

            def _on_gpu_add(info):
                _logger.info(
                    "GPU hot-plug: device added [%d] %s — refreshing scheduler",
                    info.get("index", -1), info.get("name", "?"),
                )
                if self.scheduler:
                    self.scheduler.refresh_gpu_info()

            def _on_gpu_remove(info):
                _logger.warning(
                    "GPU hot-plug: device removed [%d] — refreshing scheduler",
                    info.get("index", -1),
                )
                if self.scheduler:
                    self.scheduler.refresh_gpu_info()

            self.gpu_hotplug.on_add(_on_gpu_add)
            self.gpu_hotplug.on_remove(_on_gpu_remove)
            self.gpu_hotplug.start()
        except Exception as exc:
            _logger.debug("GPU hot-plug init failed: %s", exc)

    def _init_turboquant_cache(self) -> None:
        """Initialize HF-native TurboQuant KV cache factory.

        When VRM_KV_COMPRESSION=turboquant, creates a factory that produces
        fresh TurboQuantCache instances.  Each call to generate() gets a new
        cache so HF's generate() compresses KV states in-flight via
        PolarQuant+QJL (~4.6x VRAM reduction on the KV cache).
        """
        kv_comp = os.environ.get("VRM_KV_COMPRESSION", "").lower()
        if kv_comp != "turboquant":
            return

        model = self.backend.model if self.backend else None
        if model is None:
            return

        try:
            from core.turboquant_cache import TurboQuantCache

            config = getattr(model, "config", None)
            if config is None:
                _logger.debug("TurboQuantCache skipped: no model config")
                return

            device = self._detect_device()
            bits = int(os.environ.get("VRM_KV_COMPRESSION_BITS", "3"))
            residual = int(os.environ.get("VRM_KV_CACHE_RESIDUAL", "128"))

            # Factory: each generate() call gets a fresh cache
            def _make_cache():
                return TurboQuantCache.from_model_config(
                    config,
                    bits_per_angle=bits,
                    residual_length=residual,
                    device=device,
                )

            # Validate with a smoke test
            test_cache = _make_cache()
            _logger.info(
                "TurboQuantCache ready: %d layers, %.1f bits/dim, "
                "residual=%d tokens, device=%s",
                len(test_cache.layers),
                test_cache._compressor.bits_per_dim(),
                residual,
                device,
            )
            del test_cache

            self._turboquant_cache_factory = _make_cache

        except Exception as e:
            _logger.warning("TurboQuantCache init failed: %s", e)

    def _init_continuous_batching(self) -> None:
        """Initialize continuous batcher + paged KV cache."""
        try:
            from core.paged_attention import PagedKVCacheManager, PagedKVConfig

            # Auto-detect KV config from loaded model
            model = self.backend.model if self.backend else None
            if model and _TORCH:
                kv_config = PagedKVConfig.from_model(model, device=self._detect_device())
                # Distribute KV cache across all GPUs used by the model
                self._distribute_kv_cache(kv_config, model)
            else:
                kv_config = PagedKVConfig(device="cpu")

            self.paged_kv = PagedKVCacheManager(kv_config)
            _logger.info("PagedKVCache initialized: %s", self.paged_kv)

            # Wire paged KV to stream manager for compress-on-evict
            if self.stream_manager and self.paged_kv:
                self.stream_manager.paged_kv = self.paged_kv
                if self.paged_kv.kv_compression_active:
                    _logger.info(
                        "KV compression wired to StreamManager "
                        "(%.1fx reduction on eviction)",
                        self.paged_kv.compression_ratio,
                    )
        except Exception as e:
            _logger.debug("PagedKV init skipped: %s", e)
            self.paged_kv = None

        try:
            from core.continuous_batcher import ContinuousBatcher

            model = self.backend.model if self.backend else None
            tokenizer = getattr(self.backend, 'tokenizer', None)

            self.continuous_batcher = ContinuousBatcher(
                model=model,
                tokenizer=tokenizer,
                max_batch_size=int(os.environ.get("VRM_MAX_BATCH_SIZE", "32")),
                device=self._detect_device(),
                paged_kv_manager=self.paged_kv,
            )
            # Don't auto-start — only start on first submit or explicit call
            _logger.info("ContinuousBatcher ready (call pipeline.submit() to use)")
        except Exception as e:
            _logger.debug("ContinuousBatcher init skipped: %s", e)
            self.continuous_batcher = None

    def _distribute_kv_cache(self, kv_config, model) -> None:
        """Distribute KV cache pages across all GPUs used by the model.

        Reads ``hf_device_map`` to discover GPUs, measures free VRAM after
        model loading, and caps total pages so the cache fits in ≤50% of
        the remaining memory.  Pages are split proportionally to free VRAM.
        """
        if not _TORCH or not torch.cuda.is_available():
            return
        dev_map = getattr(model, "hf_device_map", None)
        if not dev_map:
            return

        # Collect unique CUDA devices
        devices = sorted(set(
            f"cuda:{v}" if isinstance(v, int) else str(v)
            for v in dev_map.values()
            if str(v).startswith("cuda") or (isinstance(v, int) and str(v).isdigit())
        ))
        if len(devices) < 2:
            return

        # Measure free VRAM per device (post-model-load)
        free_vram: dict = {}
        total_free = 0.0
        for dev in devices:
            idx = int(dev.split(":")[1]) if ":" in dev else 0
            try:
                free_bytes = torch.cuda.mem_get_info(idx)[0]
                free_gb = free_bytes / (1024 ** 3)
            except Exception:
                free_gb = 2.0
            free_vram[dev] = free_gb
            total_free += free_gb

        if total_free <= 0:
            return

        # Cap total KV cache to a fraction of aggregate free VRAM.
        # With KV compression active, raw pools are only staging buffers —
        # compressed data is ~4.6x smaller, so allocate much less raw pool.
        page_bytes = kv_config.page_size_bytes
        if page_bytes <= 0:
            return
        vram_fraction = 0.50
        if kv_config.kv_compression:
            # Staging buffer: ~12% of free VRAM (enough for active pages,
            # rest will be compressed in _compressed_pages sidecar)
            vram_fraction = 0.12
            _logger.info(
                "KV compression active — reducing raw pool to %.0f%% of free "
                "VRAM (compressed sidecar handles the rest)",
                vram_fraction * 100,
            )
        max_kv_bytes = total_free * vram_fraction * (1024 ** 3)
        affordable_pages = int(max_kv_bytes / page_bytes)
        kv_config.max_pages = min(kv_config.max_pages, max(64, affordable_pages))

        # Assign pages per device proportionally
        kv_config.devices = devices
        for dev in devices:
            ratio = free_vram[dev] / total_free
            dev_pages = max(1, int(ratio * kv_config.max_pages))
            kv_config.pages_per_device[dev] = dev_pages

        _logger.info(
            "KV cache distributed: %d pages across %s (%.1f MB total, "
            "%.1f GiB free across GPUs)",
            kv_config.max_pages,
            devices,
            kv_config.max_pages * page_bytes / 1e6,
            total_free,
        )

    def _detect_device(self) -> str:
        """Detect the best available device."""
        if not _TORCH:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _init_lending_pool(self) -> None:
        """Initialize VRAM Lending Pool for cooperative GPU memory.

        Registers all GPUs and their VRAM budgets, enabling idle GPUs
        to lend free VRAM as KV cache overflow to busy GPUs.
        """
        try:
            from core.vram_lending import get_lending_pool, LendingPolicy

            policy = LendingPolicy(
                max_lend_ratio=float(os.environ.get("VRM_LEND_RATIO", "0.70")),
                reclaim_threshold=float(os.environ.get("VRM_RECLAIM_THRESHOLD", "0.80")),
            )
            self.lending_pool = get_lending_pool(
                policy=policy,
                monitor=self.monitor,
                transfer_manager=self.transfer_manager,
            )

            # Register each GPU with its VRAM budget
            if _TORCH and torch.cuda.is_available():
                for i in range(self.num_gpus):
                    try:
                        props = torch.cuda.get_device_properties(i)
                        total = getattr(props, 'total_memory', 0) or getattr(props, 'total_mem', 0)
                        allocated = torch.cuda.memory_allocated(i)
                        # Detect PCIe gen from device name heuristic
                        name = props.name
                        pcie_gen = 5 if any(x in name.lower() for x in ["50", "blackwell"]) else 4
                        self.lending_pool.register_gpu(
                            gpu_id=i,
                            total_bytes=total,
                            model_bytes=allocated,
                            device_name=name,
                            pcie_gen=pcie_gen,
                            compute_capability=(props.major, props.minor),
                        )
                    except Exception as e:
                        _logger.debug("GPU %d lending registration failed: %s", i, e)

            # Start background monitoring for auto-reclaim
            self.lending_pool.start_monitoring(
                interval=float(os.environ.get("VRM_LENDING_INTERVAL", "2.0"))
            )
            _logger.info("VRAM Lending Pool active: %s", self.lending_pool)

        except Exception as e:
            _logger.debug("VRAM Lending init skipped: %s", e)
            self.lending_pool = None

    def _init_hierarchical_memory(self) -> None:
        """Initialize the 6-level Hierarchical Memory Manager.

        Connects to the VRAMLendingPool (if available) for cooperative
        L1↔L2 GPU memory lending with auto-reclaim.
        """
        try:
            from core.hierarchical_memory import HierarchicalMemoryManager
            self.hierarchical_memory = HierarchicalMemoryManager(
                lending_pool=self.lending_pool,
            )
            _logger.info("Hierarchical Memory Manager active (L1-L6)")
        except Exception as e:
            _logger.debug("Hierarchical Memory init skipped: %s", e)
            self.hierarchical_memory = None

    def submit(self, prompt: str, max_new_tokens: int = 128, **kwargs):
        """Submit a request to the continuous batcher (non-blocking).

        Returns a Future whose .result() gives the generated text.
        Falls back to synchronous generate() if batcher unavailable.
        """
        self._ensure_loaded()

        if self.continuous_batcher is None:
            # Fallback: wrap in a Future
            from concurrent.futures import Future
            fut = Future()
            try:
                result = self.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
                fut.set_result(result)
            except Exception as e:
                fut.set_exception(e)
            return fut

        # Start batcher if not running (thread-safe via batcher.start() lock)
        self.continuous_batcher.start()

        return self.continuous_batcher.submit(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=kwargs.get("temperature", 1.0),
            top_k=kwargs.get("top_k", 50),
            top_p=kwargs.get("top_p", 1.0),
        )

    def batcher_stats(self) -> Optional[Dict[str, Any]]:
        """Return continuous batcher statistics."""
        if self.continuous_batcher:
            return self.continuous_batcher.stats()
        return None

    def benchmark(self, prompts: Optional[List[str]] = None, **kwargs):
        """Run a standardized benchmark and return results.

        See core.benchmark.BenchmarkRunner for full parameter docs.
        """
        from core.benchmark import BenchmarkRunner
        runner = BenchmarkRunner(verbose=True)
        return runner.run(
            model_name=self.model_name or "gpt2",
            prompts=prompts,
            num_gpus=self.num_gpus,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError(
                "Pipeline not loaded. Call pipeline.load(model_name) first."
            )

    def _update_gpu_metrics(self):
        """Update Prometheus GPU memory gauges."""
        if not _METRICS or not _TORCH:
            return
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    GPU_MEMORY_USED.labels(gpu=str(i)).set(
                        torch.cuda.memory_allocated(i)
                    )
        except Exception as e:
            _logger.debug(f"Exception silencieuse dans l'exécution: {e}", exc_info=True)

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self):
        """Cleanup all resources."""
        _logger.info("Shutting down inference pipeline...")

        # Stop fault tolerance recovery thread
        if self.fault_manager and hasattr(self.fault_manager, 'stop'):
            try:
                self.fault_manager.stop()
            except Exception as e:
                _logger.debug(f"Exception silencieuse dans l'exécution: {e}", exc_info=True)

        # Stop continuous batcher
        if self.continuous_batcher:
            try:
                self.continuous_batcher.stop()
            except Exception as e:
                _logger.debug(f"Exception silencieuse dans l'exécution: {e}", exc_info=True)

        # Stop rebalancing
        self.stop_rebalancing()

        if hasattr(self, 'monitor') and self.monitor:
            try:
                self.monitor.stop_polling()
            except Exception as e:
                _logger.debug(f"Exception silencieuse dans l'exécution: {e}", exc_info=True)

        if hasattr(self, 'stream_manager') and self.stream_manager:
            try:
                self.stream_manager.stop_monitoring()
            except Exception as e:
                _logger.debug(f"Exception silencieuse dans l'exécution: {e}", exc_info=True)

        # Stop GPU hot-plug
        if self.gpu_hotplug:
            try:
                self.gpu_hotplug.stop()
            except Exception as e:
                _logger.debug(f"Exception silencieuse dans l'exécution: {e}", exc_info=True)

        if self.discovery:
            try:
                self.discovery.stop()
            except Exception as e:
                _logger.debug(f"Exception silencieuse dans l'exécution: {e}", exc_info=True)

        if self.transfer_manager:
            try:
                self.transfer_manager.shutdown()
            except Exception as e:
                _logger.debug(f"Exception silencieuse dans l'exécution: {e}", exc_info=True)

        # Shutdown VRAM lending pool
        if self.lending_pool:
            try:
                self.lending_pool.close()
            except Exception as e:
                _logger.debug(f"Exception silencieuse dans l'exécution: {e}", exc_info=True)
            self.lending_pool = None

        # Reset Prometheus gauge metrics (clear stale label sets)
        try:
            from core.metrics import reset_metrics
            reset_metrics()
        except Exception as e:
            _logger.debug(f"Exception silencieuse dans l'exécution: {e}", exc_info=True)

        self._loaded = False
        # Release GPU-heavy references so memory can actually be freed
        if hasattr(self, 'paged_kv') and self.paged_kv:
            if hasattr(self.paged_kv, '_gpu_pools'):
                self.paged_kv._gpu_pools.clear()
            if hasattr(self.paged_kv, '_gpu_pool'):
                self.paged_kv._gpu_pool = None
            self.paged_kv = None
        if hasattr(self, 'backend') and self.backend:
            if hasattr(self.backend, 'model'):
                del self.backend.model
            self.backend = None
        self.blocks.clear()
        _logger.info("Pipeline shutdown complete")

    def __del__(self):
        try:
            self.shutdown()
        except Exception as e:
            _logger.debug(f"Exception silencieuse dans l'exécution: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_global_pipeline: Optional[InferencePipeline] = None
_global_lock = threading.Lock()


def get_pipeline(**kwargs) -> InferencePipeline:
    """Get or create the global InferencePipeline singleton."""
    global _global_pipeline
    with _global_lock:
        if _global_pipeline is None:
            _global_pipeline = InferencePipeline(**kwargs)
        return _global_pipeline


def reset_pipeline():
    """Shutdown and reset the global pipeline (for tests)."""
    global _global_pipeline
    with _global_lock:
        if _global_pipeline is not None:
            _global_pipeline.shutdown()
            _global_pipeline = None


__all__ = [
    "InferencePipeline",
    "get_pipeline",
    "reset_pipeline",
]
