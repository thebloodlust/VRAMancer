"""VRAMancer Benchmark Runner.

Provides standardized benchmarking for inference throughput, latency,
and GPU utilization. Produces results comparable to vLLM/TGI benchmarks.

Metrics reported:
  - tok/s (tokens per second, generation throughput)
  - TTFT (time to first token)
  - ITL (inter-token latency, P50/P95/P99)
  - Memory utilization (peak VRAM)
  - Request throughput (requests/s with continuous batching)

Usage:
    from core.benchmark import BenchmarkRunner

    runner = BenchmarkRunner()

    # Quick synthetic benchmark (no model needed)
    result = runner.synthetic_benchmark(num_tokens=1000)

    # Full model benchmark
    result = runner.run(
        model_name="gpt2",
        prompts=["Hello world"] * 10,
        max_new_tokens=128,
        num_concurrent=4,
    )
    runner.print_report(result)
"""

from __future__ import annotations

import os
import time
import json
import logging
import statistics
import threading
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

_logger = logging.getLogger("vramancer.benchmark")
_MINIMAL = os.environ.get("VRM_MINIMAL_TEST", "")

try:
    import torch
    _TORCH = True
except ImportError:
    torch = None  # type: ignore
    _TORCH = False


@dataclass
class BenchmarkResult:
    """Structured benchmark results."""

    # Identity
    model_name: str = ""
    device: str = "cpu"
    num_gpus: int = 0
    batch_size: int = 1
    num_requests: int = 0
    max_new_tokens: int = 128

    # Throughput
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0
    total_tokens_generated: int = 0
    total_time_s: float = 0.0

    # Latency (ms)
    ttft_ms: float = 0.0         # time to first token
    itl_p50_ms: float = 0.0      # inter-token latency P50
    itl_p95_ms: float = 0.0
    itl_p99_ms: float = 0.0
    latency_ms: float = 0.0      # avg per-request end-to-end

    # Memory
    peak_vram_mb: float = 0.0
    vram_utilization: float = 0.0

    # Raw samples
    per_request_latencies: List[float] = field(default_factory=list)
    per_token_latencies: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("per_request_latencies", None)
        d.pop("per_token_latencies", None)
        return d

    def to_json(self, path: Optional[str] = None) -> str:
        j = json.dumps(self.to_dict(), indent=2)
        if path:
            with open(path, "w") as f:
                f.write(j)
            _logger.info("Benchmark results saved to %s", path)
        return j


class BenchmarkRunner:
    """Standardized inference benchmark runner."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Synthetic benchmark (no model needed)
    # ------------------------------------------------------------------

    def synthetic_benchmark(
        self,
        num_tokens: int = 1000,
        batch_size: int = 1,
        hidden_dim: int = 768,
        num_layers: int = 12,
        device: str = "auto",
    ) -> Dict[str, Any]:
        """Benchmark raw compute throughput with synthetic tensors.

        Simulates the compute portion of inference (matmuls) without
        loading a real model. Useful for measuring GPU raw capability.
        """
        if device == "auto":
            if _TORCH and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        result = BenchmarkResult(
            model_name="synthetic",
            device=device,
            batch_size=batch_size,
            num_requests=1,
            max_new_tokens=num_tokens,
        )

        if not _TORCH:
            # Stub: measure Python overhead
            start = time.perf_counter()
            for _ in range(num_tokens):
                _ = [0.0] * hidden_dim  # simulate token generation
            elapsed = time.perf_counter() - start
            result.total_time_s = elapsed
            result.total_tokens_generated = num_tokens
            result.tokens_per_second = num_tokens / max(elapsed, 1e-9)
            result.latency_ms = elapsed * 1000 / num_tokens
            return result.to_dict()

        # Real torch benchmark
        if device != "cpu":
            torch.cuda.synchronize()

        # Simulate transformer forward: attention (QKV matmul) + FFN
        q = torch.randn(batch_size, hidden_dim, device=device)
        k = torch.randn(batch_size, hidden_dim, device=device)
        w_ffn = torch.randn(hidden_dim, hidden_dim * 4, device=device)
        w_proj = torch.randn(hidden_dim * 4, hidden_dim, device=device)

        # Warmup
        for _ in range(10):
            attn = (q @ k.T) / (hidden_dim ** 0.5)
            ffn = torch.relu(q @ w_ffn) @ w_proj

        if device != "cpu":
            torch.cuda.synchronize()

        token_latencies = []
        start = time.perf_counter()

        for step in range(num_tokens):
            t0 = time.perf_counter()

            # Simulate one token generation step across layers
            x = q
            for layer in range(num_layers):
                attn = (x @ k.T) / (hidden_dim ** 0.5)
                x = torch.relu(x @ w_ffn) @ w_proj

            if device != "cpu":
                torch.cuda.synchronize()

            token_latencies.append((time.perf_counter() - t0) * 1000)

        elapsed = time.perf_counter() - start

        result.total_time_s = elapsed
        result.total_tokens_generated = num_tokens
        result.tokens_per_second = num_tokens / max(elapsed, 1e-9)
        result.latency_ms = elapsed * 1000 / num_tokens
        result.per_token_latencies = token_latencies

        if token_latencies:
            sorted_lat = sorted(token_latencies)
            result.itl_p50_ms = sorted_lat[len(sorted_lat) // 2]
            result.itl_p95_ms = sorted_lat[int(len(sorted_lat) * 0.95)]
            result.itl_p99_ms = sorted_lat[int(len(sorted_lat) * 0.99)]

        if device != "cpu" and torch.cuda.is_available():
            result.peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

        if self.verbose:
            _logger.info(
                "Synthetic: %.0f tok/s, ITL P50=%.2fms P95=%.2fms (device=%s)",
                result.tokens_per_second, result.itl_p50_ms, result.itl_p95_ms, device,
            )

        return result.to_dict()

    # ------------------------------------------------------------------
    # Full model benchmark
    # ------------------------------------------------------------------

    def run(
        self,
        model_name: str = "gpt2",
        prompts: Optional[List[str]] = None,
        max_new_tokens: int = 128,
        num_concurrent: int = 1,
        num_gpus: Optional[int] = None,
        warmup: int = 2,
        use_continuous_batching: bool = False,
    ) -> BenchmarkResult:
        """Run a full inference benchmark.

        Parameters
        ----------
        model_name : str
            HuggingFace model to benchmark.
        prompts : list of str
            Input prompts. Default: 10 synthetic prompts.
        max_new_tokens : int
            Tokens to generate per prompt.
        num_concurrent : int
            Number of concurrent requests (tests continuous batching).
        num_gpus : int
            GPUs to use (auto if None).
        warmup : int
            Warmup iterations before measurement.
        use_continuous_batching : bool
            Use ContinuousBatcher instead of sequential.
        """
        if prompts is None:
            prompts = [
                "The future of artificial intelligence",
                "Once upon a time in a distant galaxy",
                "Explain quantum computing in simple terms",
                "Write a Python function to sort a list",
                "The relationship between machine learning and statistics",
                "Describe the architecture of a transformer model",
                "What are the benefits of renewable energy",
                "How does a neural network learn from data",
                "The history of programming languages",
                "Explain gradient descent optimization",
            ]

        result = BenchmarkResult(
            model_name=model_name,
            num_requests=len(prompts),
            max_new_tokens=max_new_tokens,
            batch_size=num_concurrent,
        )

        # Detect device
        if _TORCH and torch.cuda.is_available():
            result.device = "cuda"
            result.num_gpus = torch.cuda.device_count() if num_gpus is None else num_gpus
        elif _TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            result.device = "mps"
            result.num_gpus = 1
        else:
            result.device = "cpu"
            result.num_gpus = 0

        if use_continuous_batching:
            return self._run_continuous(result, prompts, max_new_tokens, warmup)
        elif num_concurrent > 1:
            return self._run_concurrent(result, prompts, max_new_tokens, num_concurrent, warmup)
        else:
            return self._run_sequential(result, prompts, max_new_tokens, warmup)

    def _run_sequential(
        self,
        result: BenchmarkResult,
        prompts: List[str],
        max_new_tokens: int,
        warmup: int,
    ) -> BenchmarkResult:
        """Sequential benchmark (one request at a time)."""
        from core.inference_pipeline import InferencePipeline

        pipe = InferencePipeline(backend_name="huggingface", verbose=False)
        pipe.load(result.model_name, num_gpus=result.num_gpus or None)

        # Warmup
        for i in range(min(warmup, len(prompts))):
            pipe.generate(prompts[i], max_new_tokens=min(max_new_tokens, 10))

        if _TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Measurement
        latencies = []
        total_tokens = 0
        start_total = time.perf_counter()

        for prompt in prompts:
            t0 = time.perf_counter()
            output = pipe.generate(prompt, max_new_tokens=max_new_tokens)
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed * 1000)  # ms

            # Estimate tokens generated
            if pipe.backend and pipe.backend.tokenizer:
                tokens = len(pipe.backend.tokenizer.encode(output)) - len(
                    pipe.backend.tokenizer.encode(prompt)
                )
                total_tokens += max(tokens, 1)
            else:
                total_tokens += max_new_tokens

        total_time = time.perf_counter() - start_total

        result.total_time_s = total_time
        result.total_tokens_generated = total_tokens
        result.tokens_per_second = total_tokens / max(total_time, 1e-9)
        result.requests_per_second = len(prompts) / max(total_time, 1e-9)
        result.per_request_latencies = latencies
        result.latency_ms = statistics.mean(latencies) if latencies else 0

        if latencies:
            sorted_lat = sorted(latencies)
            result.ttft_ms = sorted_lat[0]  # first request as proxy
            result.itl_p50_ms = result.latency_ms / max_new_tokens
            p95_idx = int(len(sorted_lat) * 0.95)
            result.itl_p95_ms = sorted_lat[p95_idx] / max_new_tokens

        if _TORCH and torch.cuda.is_available():
            result.peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

        pipe.shutdown()

        if self.verbose:
            self.print_report(result)

        return result

    def _run_concurrent(
        self,
        result: BenchmarkResult,
        prompts: List[str],
        max_new_tokens: int,
        num_concurrent: int,
        warmup: int,
    ) -> BenchmarkResult:
        """Concurrent benchmark using ThreadPoolExecutor."""
        from core.inference_pipeline import InferencePipeline

        pipe = InferencePipeline(backend_name="huggingface", verbose=False)
        pipe.load(result.model_name, num_gpus=result.num_gpus or None)

        # Warmup
        for i in range(min(warmup, len(prompts))):
            pipe.generate(prompts[i], max_new_tokens=min(max_new_tokens, 10))

        if _TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        latencies = []
        total_tokens = 0
        start_total = time.perf_counter()

        def _generate_one(prompt):
            t0 = time.perf_counter()
            output = pipe.generate(prompt, max_new_tokens=max_new_tokens)
            elapsed = time.perf_counter() - t0
            return elapsed, output

        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(_generate_one, p) for p in prompts]
            for fut in as_completed(futures):
                elapsed, output = fut.result()
                latencies.append(elapsed * 1000)
                total_tokens += max_new_tokens

        total_time = time.perf_counter() - start_total

        result.total_time_s = total_time
        result.total_tokens_generated = total_tokens
        result.tokens_per_second = total_tokens / max(total_time, 1e-9)
        result.requests_per_second = len(prompts) / max(total_time, 1e-9)
        result.per_request_latencies = latencies
        result.latency_ms = statistics.mean(latencies) if latencies else 0

        if _TORCH and torch.cuda.is_available():
            result.peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

        pipe.shutdown()

        if self.verbose:
            self.print_report(result)

        return result

    def _run_continuous(
        self,
        result: BenchmarkResult,
        prompts: List[str],
        max_new_tokens: int,
        warmup: int,
    ) -> BenchmarkResult:
        """Benchmark using ContinuousBatcher."""
        from core.continuous_batcher import ContinuousBatcher
        from core.inference_pipeline import InferencePipeline

        pipe = InferencePipeline(backend_name="huggingface", verbose=False)
        pipe.load(result.model_name, num_gpus=result.num_gpus or None)

        batcher = ContinuousBatcher(
            model=pipe.backend.model if pipe.backend else None,
            tokenizer=pipe.backend.tokenizer if pipe.backend else None,
            max_batch_size=32,
            device=result.device,
        )
        batcher.start()

        # Warmup
        for i in range(min(warmup, len(prompts))):
            f = batcher.submit(prompts[i], max_new_tokens=min(max_new_tokens, 10))
            f.result(timeout=30)

        if _TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Submit all at once
        start_total = time.perf_counter()
        futures = []
        for prompt in prompts:
            futures.append((time.perf_counter(), batcher.submit(prompt, max_new_tokens=max_new_tokens)))

        latencies = []
        total_tokens = 0
        for submit_time, fut in futures:
            try:
                output = fut.result(timeout=120)
                elapsed = time.perf_counter() - submit_time
                latencies.append(elapsed * 1000)
                total_tokens += max_new_tokens
            except Exception as e:
                _logger.warning("Request failed: %s", e)

        total_time = time.perf_counter() - start_total

        result.total_time_s = total_time
        result.total_tokens_generated = total_tokens
        result.tokens_per_second = total_tokens / max(total_time, 1e-9)
        result.requests_per_second = len(prompts) / max(total_time, 1e-9)
        result.per_request_latencies = latencies
        result.latency_ms = statistics.mean(latencies) if latencies else 0

        if _TORCH and torch.cuda.is_available():
            result.peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6

        batcher_stats = batcher.stats()
        batcher.stop()
        pipe.shutdown()

        if self.verbose:
            self.print_report(result)
            _logger.info("Batcher stats: %s", batcher_stats)

        return result

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def print_report(self, result: BenchmarkResult) -> str:
        """Print a formatted benchmark report."""
        lines = [
            "",
            "=" * 60,
            f"  VRAMancer Benchmark Report — {result.model_name}",
            "=" * 60,
            f"  Device:           {result.device} ({result.num_gpus} GPUs)",
            f"  Requests:         {result.num_requests}",
            f"  Max new tokens:   {result.max_new_tokens}",
            f"  Batch size:       {result.batch_size}",
            "-" * 60,
            f"  Throughput:       {result.tokens_per_second:.1f} tok/s",
            f"  Req throughput:   {result.requests_per_second:.2f} req/s",
            f"  Total tokens:     {result.total_tokens_generated}",
            f"  Total time:       {result.total_time_s:.2f}s",
            "-" * 60,
            f"  TTFT:             {result.ttft_ms:.1f} ms",
            f"  Avg latency:      {result.latency_ms:.1f} ms",
            f"  ITL P50:          {result.itl_p50_ms:.2f} ms",
            f"  ITL P95:          {result.itl_p95_ms:.2f} ms",
            f"  ITL P99:          {result.itl_p99_ms:.2f} ms",
            "-" * 60,
            f"  Peak VRAM:        {result.peak_vram_mb:.0f} MB",
            "=" * 60,
            "",
        ]
        report = "\n".join(lines)
        print(report)
        return report


__all__ = ["BenchmarkRunner", "BenchmarkResult"]
