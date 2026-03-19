#!/usr/bin/env python3
"""
VRAMancer Benchmark Suite (Sprint A)
Measures throughput, latency, and VRAM utilization across different configurations.
"""

import os
import sys
import time
import argparse
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.inference_pipeline import InferencePipeline, get_pipeline
from core.backends import select_backend
from core.config import resolve_config
from core.monitor import GPUMonitor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Benchmark")

def run_benchmark(model_name: str, num_gpus: int, prompts: List[str], concurrency: int) -> Dict[str, Any]:
    logger.info(f"Starting benchmark for {model_name} on {num_gpus} GPUs (Concurrency: {concurrency})")
    
    # Initialize pipeline
    config = resolve_config()
    os.environ["VRM_MINIMAL_TEST"] = "0" # Disable minimial test stub
    try:
        pipeline = InferencePipeline.load(model_name, num_gpus)
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}

    monitor = GPUMonitor()
    initial_vram = monitor.vram_usage()
    logger.info(f"Initial VRAM usage: {initial_vram}")

    results = []
    start_time = time.time()
    
    def process_prompt(prompt: str) -> Dict[str, Any]:
        req_start = time.time()
        try:
            # Note: generate() is an iterator in production_api, but might return strings directly in inference_pipeline depending on implementation
            tokens = 0
            for chunk in pipeline.generate(prompt, max_new_tokens=100):
                tokens += 1
            latency = time.time() - req_start
            return {"prompt": prompt, "latency": latency, "tokens": tokens, "success": True}
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return {"prompt": prompt, "latency": time.time() - req_start, "tokens": 0, "success": False}

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(process_prompt, p) for p in prompts]
        for f in futures:
            results.append(f.result())

    total_time = time.time() - start_time
    total_tokens = sum(r["tokens"] for r in results if r["success"])
    peak_vram = monitor.vram_usage()

    metrics = {
        "status": "success",
        "total_time_s": total_time,
        "total_tokens_generated": total_tokens,
        "throughput_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
        "avg_latency_s": sum(r["latency"] for r in results) / len(results) if results else 0,
        "peak_vram": peak_vram,
        "initial_vram": initial_vram
    }
    
    logger.info(f"Benchmark completed: {metrics}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VRAMancer Benchmark Suite")
    parser.add_argument("--model", type=str, default="gpt2", help="Model ID")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to target")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent requests")
    parser.add_argument("--prompts", type=int, default=10, help="Number of iterations")
    
    args = parser.parse_args()
    
    sample_prompts = [
        "What is the theory of relativity?",
        "Explain quantum computing in simple terms.",
        "Write a Python script to scrape a website.",
        "How do neural networks learn?"
    ] * (args.prompts // 4 + 1)
    
    sample_prompts = sample_prompts[:args.prompts]
    
    run_benchmark(args.model, args.gpus, sample_prompts, args.concurrency)
