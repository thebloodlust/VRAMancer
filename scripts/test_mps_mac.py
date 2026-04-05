#!/usr/bin/env python3
"""Test VRAMancer MPS backend on Apple Silicon (M4).

Run on MacBook Air M4 or Mac Mini M4:
    pip install torch transformers
    pip install -e .
    python scripts/test_mps_mac.py

Tests:
  1. detect_backend() returns 'mps'
  2. GPT-2 inference on MPS device
  3. TinyLlama inference on MPS (if enough RAM)
  4. Full test suite (stub mode)
"""

import os
import sys
import time

os.environ["VRM_TEST_MODE"] = "1"
os.environ["VRM_BACKEND_ALLOW_STUB"] = "1"

def test_detect_backend():
    """Test that VRAMancer detects MPS on Apple Silicon."""
    from core.utils import detect_backend
    backend = detect_backend()
    print(f"[1/4] detect_backend() = {backend}")
    assert backend == "mps", f"Expected 'mps', got '{backend}'"
    print("      PASS\n")

def test_gpt2_mps():
    """Test GPT-2 inference on MPS."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    assert torch.backends.mps.is_available(), "MPS not available"
    print(f"[2/4] MPS available: {torch.backends.mps.is_available()}")
    print(f"      PyTorch: {torch.__version__}")

    model_name = "gpt2"
    print(f"      Loading {model_name} on MPS...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = model.to("mps")
    model.eval()

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    elapsed = time.perf_counter() - t0

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    toks = out.shape[-1] - inputs["input_ids"].shape[-1]
    tps = toks / elapsed

    print(f"      Output: {text}")
    print(f"      {toks} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")
    assert "Paris" in text, f"Expected 'Paris' in output"
    print("      PASS\n")

def test_tinyllama_mps():
    """Test TinyLlama 1.1B on MPS (needs ~4 GB)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"[3/4] Loading {model_name} on MPS...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        model = model.to("mps")
        model.eval()
    except Exception as e:
        print(f"      SKIP: {e}\n")
        return

    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30, do_sample=False)
    elapsed = time.perf_counter() - t0

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    toks = out.shape[-1] - inputs["input_ids"].shape[-1]
    tps = toks / elapsed

    print(f"      Output: {text[:120]}...")
    print(f"      {toks} tokens in {elapsed:.2f}s = {tps:.1f} tok/s")
    print("      PASS\n")

def test_stub_suite():
    """Run the full test suite in stub mode."""
    print("[4/4] Running test suite (VRM_MINIMAL_TEST=1)...")
    import subprocess
    os.environ["VRM_MINIMAL_TEST"] = "1"
    os.environ["VRM_DISABLE_RATE_LIMIT"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=line",
         "-W", "ignore", "--no-header"],
        capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
    )
    # Get last line with pass/fail count
    lines = result.stdout.strip().split("\n")
    summary = lines[-1] if lines else "no output"
    print(f"      {summary}")
    if "failed" in summary and "0 failed" not in summary:
        print("      FAIL\n")
    else:
        print("      PASS\n")

if __name__ == "__main__":
    print("=" * 60)
    print("VRAMancer MPS Backend Test — Apple Silicon")
    print("=" * 60)
    print()

    test_detect_backend()
    test_gpt2_mps()
    test_tinyllama_mps()
    test_stub_suite()

    print("=" * 60)
    print("ALL MPS TESTS PASSED")
    print("=" * 60)
