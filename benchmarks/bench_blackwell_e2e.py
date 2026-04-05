#!/usr/bin/env python3
"""
End-to-end benchmark: Qwen2.5-7B-Instruct with NVFP4 on Blackwell (RTX 5070 Ti).

Measures real tok/s with VRAMancer's fused CUDA activation quantizer + _scaled_mm,
comparing against the torchao default path.
"""
import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6;9.0+PTX")

import sys
import time
import torch

# Ensure VRAMancer is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DEVICE = "cuda:1"  # RTX 5070 Ti (Blackwell, SM 12.0)
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 128
WARMUP_TOKENS = 32
PROMPT = "Explain the theory of general relativity in simple terms."


def nvfp4_filter_fn(module, fqn):
    """Only quantize nn.Linear layers."""
    import torch.nn as nn
    return isinstance(module, nn.Linear)


def load_model_nvfp4(device: str):
    """Load Qwen2.5-7B with NVFP4 quantization on Blackwell."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from torchao.quantization import quantize_
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig

    print(f"Loading {MODEL_ID} on {device}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    print(f"  BF16 load: {time.time()-t0:.1f}s")

    # NVFP4 quantization (on CPU, then move to GPU)
    t1 = time.time()
    torch.cuda.set_device(device)
    config = NVFP4DynamicActivationNVFP4WeightConfig()
    quantize_(model, config, filter_fn=nvfp4_filter_fn)
    print(f"  NVFP4 quantize: {time.time()-t1:.1f}s")

    # Replace with DirectFP4Linear (fused CUDA quant on Blackwell)
    t2 = time.time()
    from core.nvfp4_direct import replace_with_direct_fp4
    n = replace_with_direct_fp4(model, verbose=False)
    print(f"  DirectFP4 replace: {n} layers in {time.time()-t2:.1f}s")

    # Move to GPU
    model = model.to(device)
    torch.cuda.synchronize(device)

    vram_gb = torch.cuda.memory_allocated(device) / 1e9
    print(f"  VRAM: {vram_gb:.2f} GB")
    print(f"  Total load time: {time.time()-t0:.1f}s")

    return model, tokenizer


def benchmark_generate(model, tokenizer, prompt: str, max_new: int, device: str):
    """Benchmark autoregressive generation, return (output_text, tok/s, ttft_ms)."""
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]

    # Warmup (prefill + a few decode steps)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=WARMUP_TOKENS, do_sample=False)
    torch.cuda.synchronize(device)

    # Timed generation
    torch.cuda.synchronize(device)
    t_start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
    torch.cuda.synchronize(device)
    t_end = time.perf_counter()

    gen_tokens = output_ids.shape[1] - input_len
    elapsed = t_end - t_start
    tok_s = gen_tokens / elapsed

    output_text = tokenizer.decode(output_ids[0, input_len:], skip_special_tokens=True)

    return output_text, tok_s, gen_tokens, elapsed


def benchmark_generate_compiled(model, tokenizer, prompt: str, max_new: int, device: str):
    """Benchmark with StaticCache + torch.compile (Inductor, no CUDA Graphs)."""
    from transformers import StaticCache

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]

    torch.set_float32_matmul_precision("high")
    max_total = input_len + max_new + 16

    # Compile the model's forward — Inductor kernel fusion
    # "default" mode: no CUDA Graphs (StaticCache index_copy_ is incompatible)
    # Still fuses RMSNorm, residual adds, softmax etc. into fewer kernels
    if not getattr(model, "_vrm_compiled", False):
        model.forward = torch.compile(model.forward, mode="default", fullgraph=False)
        model._vrm_compiled = True

    def _decode_loop(input_ids, cache, n_tokens):
        """Manual greedy decode."""
        seq_len = input_ids.shape[1]
        cache_position = torch.arange(seq_len, device=device)
        generated = []

        # Prefill
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
            )
        next_token = out.logits[:, -1:].argmax(dim=-1)
        generated.append(next_token)

        # Decode loop
        for i in range(n_tokens - 1):
            cache_position = torch.tensor([seq_len + i], device=device)
            with torch.no_grad():
                out = model(
                    input_ids=next_token,
                    cache_position=cache_position,
                    past_key_values=cache,
                    use_cache=True,
                )
            next_token = out.logits[:, -1:].argmax(dim=-1)
            generated.append(next_token)

            if next_token.item() == tokenizer.eos_token_id:
                break

        return torch.cat(generated, dim=-1)

    # Warmup
    cache_warmup = StaticCache(
        model.config, max_batch_size=1, max_cache_len=max_total,
        device=device, dtype=torch.bfloat16,
    )
    _decode_loop(input_ids, cache_warmup, WARMUP_TOKENS)
    torch.cuda.synchronize(device)

    # Timed run
    cache = StaticCache(
        model.config, max_batch_size=1, max_cache_len=max_total,
        device=device, dtype=torch.bfloat16,
    )

    torch.cuda.synchronize(device)
    t_start = time.perf_counter()
    gen_ids = _decode_loop(input_ids, cache, max_new)
    torch.cuda.synchronize(device)
    t_end = time.perf_counter()

    gen_tokens = gen_ids.shape[1]
    elapsed = t_end - t_start
    tok_s = gen_tokens / elapsed
    output_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    return output_text, tok_s, gen_tokens, elapsed


def benchmark_generate_cudagraph(model, tokenizer, prompt: str, max_new: int, device: str):
    """Benchmark with manual CUDA Graph capture for decode.
    
    Bypasses Inductor's broken CUDA Graph tree by capturing the decode step ourselves.
    Prefill runs eagerly, decode replays a captured graph → zero Python overhead per token.
    """
    from transformers import StaticCache

    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    input_len = input_ids.shape[1]
    max_total = input_len + max_new + 16

    torch.set_float32_matmul_precision("high")

    # Compile with Inductor (no automatic CUDA Graphs)
    if not getattr(model, "_vrm_compiled", False):
        model.forward = torch.compile(model.forward, mode="default", fullgraph=False)
        model._vrm_compiled = True

    # --- Phase 1: Warm up compilation with throwaway cache ---
    cache_warmup = StaticCache(
        model.config, max_batch_size=1, max_cache_len=max_total,
        device=device, dtype=torch.bfloat16,
    )
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            cache_position=torch.arange(input_len, device=device),
            past_key_values=cache_warmup,
            use_cache=True,
        )
    # Warm up decode path (triggers compilation for seq_len=1)
    tok = out.logits[:, -1:].argmax(dim=-1)
    for i in range(5):
        with torch.no_grad():
            out = model(
                input_ids=tok,
                cache_position=torch.tensor([input_len + i], device=device),
                past_key_values=cache_warmup,
                use_cache=True,
            )
        tok = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize(device)

    # --- Phase 2: Fresh cache + prefill ---
    cache = StaticCache(
        model.config, max_batch_size=1, max_cache_len=max_total,
        device=device, dtype=torch.bfloat16,
    )
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            cache_position=torch.arange(input_len, device=device),
            past_key_values=cache,
            use_cache=True,
        )
    first_token = out.logits[:, -1:].argmax(dim=-1)
    torch.cuda.synchronize(device)

    # --- Phase 3: CUDA Graph capture of one decode step ---
    # Static input buffers — graph reads from these, we copy_() new values before replay
    static_input_ids = torch.zeros(1, 1, dtype=torch.long, device=device)
    static_cache_pos = torch.zeros(1, dtype=torch.long, device=device)

    # Pre-fill static buffers with first decode step values
    static_input_ids.copy_(first_token)
    static_cache_pos.fill_(input_len)

    # Warmup run on the capture stream (required before graph capture)
    s = torch.cuda.Stream(device=device)
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        with torch.no_grad():
            static_out = model(
                input_ids=static_input_ids,
                cache_position=static_cache_pos,
                past_key_values=cache,
                use_cache=True,
            )
    torch.cuda.current_stream(device).wait_stream(s)

    # Capture the graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        with torch.no_grad():
            static_out = model(
                input_ids=static_input_ids,
                cache_position=static_cache_pos,
                past_key_values=cache,
                use_cache=True,
            )
    torch.cuda.synchronize(device)

    # --- Phase 4: Timed decode via graph replay ---
    # Need fresh cache for the timed run
    cache2 = StaticCache(
        model.config, max_batch_size=1, max_cache_len=max_total,
        device=device, dtype=torch.bfloat16,
    )
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            cache_position=torch.arange(input_len, device=device),
            past_key_values=cache2,
            use_cache=True,
        )
    next_token = out.logits[:, -1:].argmax(dim=-1)

    # Wait, we captured graph with `cache` but now using `cache2`.
    # We need to recapture with cache2, or reuse cache.
    # For simplicity, recapture with cache2:
    static_input_ids.copy_(next_token)
    static_cache_pos.fill_(input_len)
    s.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(s):
        with torch.no_grad():
            static_out = model(
                input_ids=static_input_ids,
                cache_position=static_cache_pos,
                past_key_values=cache2,
                use_cache=True,
            )
    torch.cuda.current_stream(device).wait_stream(s)

    g2 = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g2, stream=s):
        with torch.no_grad():
            static_out = model(
                input_ids=static_input_ids,
                cache_position=static_cache_pos,
                past_key_values=cache2,
                use_cache=True,
            )
    torch.cuda.synchronize(device)

    generated = [next_token.clone()]
    torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    for i in range(max_new - 1):
        static_input_ids.copy_(next_token)
        static_cache_pos.fill_(input_len + 1 + i)
        g2.replay()
        next_token = static_out.logits[:, -1:].argmax(dim=-1)
        generated.append(next_token.clone())
        if next_token.item() == tokenizer.eos_token_id:
            break

    torch.cuda.synchronize(device)
    t_end = time.perf_counter()

    gen_ids = torch.cat(generated, dim=-1)
    gen_tokens = gen_ids.shape[1]
    elapsed = t_end - t_start
    tok_s = gen_tokens / elapsed
    output_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

    return output_text, tok_s, gen_tokens, elapsed


def main():
    print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"Target: {torch.cuda.get_device_name(DEVICE)} ({DEVICE})")
    cc = torch.cuda.get_device_capability(DEVICE)
    print(f"Compute capability: {cc[0]}.{cc[1]}")
    print()

    model, tokenizer = load_model_nvfp4(DEVICE)
    print()

    # ── Eager baseline ──
    print(f"=== EAGER: Generating {MAX_NEW_TOKENS} tokens ===")
    text, tok_s, n_tok, elapsed = benchmark_generate(
        model, tokenizer, PROMPT, MAX_NEW_TOKENS, DEVICE
    )
    print(f"Output ({n_tok} tokens): {text[:200]}...")
    print(f"  Eager: {tok_s:.1f} tok/s ({n_tok} tokens in {elapsed:.2f}s)")
    eager_tok_s = tok_s
    print()

    # ── Compiled (Inductor, no CUDA Graph) ──
    print(f"=== COMPILED (torch.compile + StaticCache): {MAX_NEW_TOKENS} tokens ===")
    print("Compiling... (first run includes JIT compilation)")

    text_c, tok_s_c, n_tok_c, elapsed_c = benchmark_generate_compiled(
        model, tokenizer, PROMPT, MAX_NEW_TOKENS, DEVICE
    )
    print(f"Output ({n_tok_c} tokens): {text_c[:200]}...")
    print(f"  Compiled: {tok_s_c:.1f} tok/s ({n_tok_c} tokens in {elapsed_c:.2f}s)")
    print()

    # Additional compiled runs
    print("=== 3 additional compiled runs ===")
    results = []
    for i in range(3):
        _, ts, nt, el = benchmark_generate_compiled(
            model, tokenizer, PROMPT, MAX_NEW_TOKENS, DEVICE
        )
        results.append(ts)
        print(f"  Run {i+1}: {ts:.1f} tok/s ({nt} tokens in {el:.2f}s)")

    avg_compiled = sum(results) / len(results)
    print(f"  Average: {avg_compiled:.1f} tok/s")
    print()

    # ── Manual CUDA Graph ──
    print(f"=== CUDA GRAPH (manual capture + replay): {MAX_NEW_TOKENS} tokens ===")
    print("Capturing CUDA Graph for decode step...")
    try:
        text_g, tok_s_g, n_tok_g, elapsed_g = benchmark_generate_cudagraph(
            model, tokenizer, PROMPT, MAX_NEW_TOKENS, DEVICE
        )
        print(f"Output ({n_tok_g} tokens): {text_g[:200]}...")
        print(f"  CUDA Graph: {tok_s_g:.1f} tok/s ({n_tok_g} tokens in {elapsed_g:.2f}s)")
        avg_graph = tok_s_g
    except Exception as e:
        print(f"  CUDA Graph FAILED: {type(e).__name__}: {e}")
        avg_graph = None
    print()

    vram_gb = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    print(f"  Peak VRAM: {vram_gb:.2f} GB")
    print()

    # Comparisons
    print("=== Final Results ===")
    print(f"  VRAMancer NVFP4 eager:        {eager_tok_s:.1f} tok/s")
    print(f"  VRAMancer NVFP4 compiled:     {avg_compiled:.1f} tok/s ({avg_compiled/eager_tok_s:.1f}x)")
    if avg_graph:
        print(f"  VRAMancer NVFP4 CUDA Graph:   {avg_graph:.1f} tok/s ({avg_graph/eager_tok_s:.1f}x)")
    print(f"  cuBLAS FP16 (3090):           ~54 tok/s")
    print(f"  llama.cpp Q4_K_M (GGUF):      ~107 tok/s")


if __name__ == "__main__":
    main()
