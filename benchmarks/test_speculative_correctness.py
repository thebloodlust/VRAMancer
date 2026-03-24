#!/usr/bin/env python3
"""Quick correctness test for SpeculativeTurboEngine (no compile).

Verifies that speculative decoding produces coherent text and
the acceptance rate is reasonable.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE = "cuda:0"
DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAIN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROMPT = "Explain general relativity in simple terms:"
MAX_NEW = 64


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from core.turbo_engine import TurboEngine, SpeculativeTurboEngine

    # Find best GPU
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        free, total = torch.cuda.mem_get_info(i)
        print(f"  GPU {i}: {name} ({free/1024**3:.1f} GB free / {total/1024**3:.1f} GB total)")
        if "3090" in name or free > 20e9:
            DEVICE_USE = f"cuda:{i}"
    print(f"Using {DEVICE_USE}")

    tokenizer = AutoTokenizer.from_pretrained(MAIN_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load draft model (fp16)
    print(f"Loading draft model: {DRAFT_MODEL}")
    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL, torch_dtype=torch.float16,
        device_map={"": DEVICE_USE}, trust_remote_code=True,
    )
    draft_model.eval()
    print(f"  Draft VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # Load main model (NF4)
    print(f"Loading main model: {MAIN_MODEL}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    main_model = AutoModelForCausalLM.from_pretrained(
        MAIN_MODEL, quantization_config=bnb_config,
        device_map={"": DEVICE_USE}, trust_remote_code=True,
    )
    main_model.eval()
    print(f"  Total VRAM: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

    # 1. Reference: TurboEngine (no compile, plain autoregressive)
    print("\n--- TurboEngine reference (no compile) ---")
    turbo = TurboEngine(main_model, tokenizer, device=DEVICE_USE, max_seq_len=2048, compile=False)
    ref_text = turbo.generate(PROMPT, max_new_tokens=MAX_NEW, do_sample=False)
    print(f'  Reference: "{ref_text[:120]}..."')

    # 2. SpeculativeTurboEngine (no compile)
    print("\n--- SpeculativeTurboEngine (no compile) ---")
    spec = SpeculativeTurboEngine(
        main_model=main_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        device=DEVICE_USE,
        gamma=5,
        compile_main=False,
        compile_draft=False,
    )
    spec_text = spec.generate(PROMPT, max_new_tokens=MAX_NEW, do_sample=False)
    stats = spec.stats
    print(f'  Speculative: "{spec_text[:120]}..."')
    print(f"  Acceptance: {stats['acceptance_rate']*100:.1f}% ({stats['accepted']}/{stats['drafted']})")
    print(f"  Effective tok/round: {stats['effective_tokens_per_round']:.1f}")

    # 3. Check quality
    # In greedy mode, speculative decoding should produce IDENTICAL output
    # to standard decoding (it's mathematically equivalent).
    ref_tokens = tokenizer.encode(ref_text)
    spec_tokens = tokenizer.encode(spec_text)

    match_len = 0
    for a, b in zip(ref_tokens, spec_tokens):
        if a == b:
            match_len += 1
        else:
            break

    total = max(len(ref_tokens), len(spec_tokens))
    print(f"\n--- Correctness ---")
    print(f"  Reference tokens: {len(ref_tokens)}")
    print(f"  Speculative tokens: {len(spec_tokens)}")
    print(f"  Matching prefix: {match_len}/{total} ({match_len/total*100:.0f}%)")

    if ref_tokens == spec_tokens:
        print("  ✓ PERFECT MATCH — speculative output is bit-identical to reference")
    elif match_len > total * 0.9:
        print("  ≈ CLOSE MATCH — minor divergence (acceptable for float rounding)")
    else:
        print("  ✗ MISMATCH — speculative decode has bugs!")
        print(f"  First divergence at position {match_len}:")
        if match_len < len(ref_tokens):
            print(f"    Reference: ...{tokenizer.decode(ref_tokens[max(0,match_len-3):match_len+5])}")
        if match_len < len(spec_tokens):
            print(f"    Speculative: ...{tokenizer.decode(spec_tokens[max(0,match_len-3):match_len+5])}")

    print(f"\n  Acceptance rate: {stats['acceptance_rate']*100:.1f}%")
    if stats['acceptance_rate'] < 0.3:
        print("  ⚠ LOW acceptance rate — draft model may be too different from main")
    elif stats['acceptance_rate'] > 0.6:
        print("  ✓ GOOD acceptance rate")


if __name__ == "__main__":
    main()
