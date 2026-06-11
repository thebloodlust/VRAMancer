#!/usr/bin/env python3
"""T7.1 -- Prompt lookup decoding (n-gram speculation, no draft model).

Hypothesis: in code generation, the output frequently re-copies n-grams
already present in the context (e.g. echoing a function signature back,
repeating an import block, restating surrounding code). transformers'
``prompt_lookup_num_tokens`` (PromptLookupCandidateGenerator) exploits this
losslessly: candidate continuations are looked up in the prompt, then
verified by the model in one forward pass (same algorithm as speculative
decoding, but the "draft" is just an n-gram match against the prompt).

Wired via ``VRM_PROMPT_LOOKUP=N`` (core/backends.py, HF backend, Path 1
single-device ``model.generate()``).

Usage:
    python benchmarks/bench_prompt_lookup.py --model Qwen/Qwen2.5-7B-Instruct --num-gpus 1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT_JSON = Path("benchmarks/results/phase7/T7.1_prompt_lookup.json")
OUT_MD = Path("benchmarks/results/phase7/T7.1_prompt_lookup.md")

# Three different ~200-line code-editing prompts. Each embeds a real
# VRAMancer source file and asks for a repetitive, copy-heavy edit -- the
# kind of task where n-gram lookup should help (echoing signatures,
# restating surrounding code).
_FILES = [
    "core/inference_pipeline.py",
    "core/backends.py",
    "core/api/registry.py",
]
_TASKS = [
    "Add a one-line Google-style docstring to every public method in the "
    "class above that doesn't already have one. Reproduce the full class "
    "with the new docstrings inserted, unchanged otherwise.",
    "Rename every occurrence of `self.model` to `self._model` in the code "
    "above and reproduce the full result.",
    "Add type hints (-> None) to every method in the code above that "
    "currently has no return annotation, and reproduce the full result.",
]


def _build_prompt(file_path: str, task: str, n_lines: int = 200) -> str:
    lines = Path(file_path).read_text(errors="ignore").splitlines()[:n_lines]
    code = "\n".join(lines)
    return f"```python\n{code}\n```\n\n{task}\n"


def _run_variant(label: str, lookup_n: int, model: str, num_gpus: int,
                  prompts: list[str], max_new: int, timeout: int) -> dict:
    print(f"\n=== Variant: {label} (VRM_PROMPT_LOOKUP={lookup_n}) ===")

    script = textwrap.dedent(f"""
        import os, sys, json, time
        os.environ['VRM_PROMPT_LOOKUP'] = {str(lookup_n)!r}
        os.environ['VRM_FORCE_MULTI_GPU'] = '1' if {num_gpus} > 1 else '0'
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        os.environ.pop('VRM_MINIMAL_TEST', None)
        os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        os.environ['VRM_VRAM_LENDING'] = '0'

        import torch
        from core.inference_pipeline import InferencePipeline, reset_pipeline
        reset_pipeline()

        pipe = InferencePipeline(
            backend_name='huggingface', enable_metrics=False,
            enable_discovery=False, verbose=False,
        )
        pipe.load({model!r}, num_gpus={num_gpus})

        prompts = {prompts!r}
        MAX_NEW = {max_new}

        results = []
        for i, p in enumerate(prompts):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = pipe.generate(p, max_new_tokens=MAX_NEW, do_sample=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            results.append({{
                'prompt_idx': i,
                'output': out,
                'elapsed_s': round(elapsed, 3),
                'tok_s': round(MAX_NEW / elapsed, 2) if elapsed > 0 else 0.0,
            }})
            print(f"prompt {{i}}: {{results[-1]['tok_s']}} tok/s", file=sys.stderr)

        print("RESULT_JSON:" + json.dumps({{'label': {label!r}, 'results': results}}))
    """)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=timeout,
    )
    out_line = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            out_line = line[len("RESULT_JSON:"):]
    if out_line is None:
        return {
            "label": label, "ok": False,
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-40:]),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:]),
        }
    r = json.loads(out_line)
    r["ok"] = True
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--num-gpus", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--n-lines", type=int, default=200)
    ap.add_argument("--lookup", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    prompts = [_build_prompt(f, t, args.n_lines) for f, t in zip(_FILES, _TASKS)]

    print(f"[T7.1] Prompt lookup decoding -- model={args.model}, "
          f"num_gpus={args.num_gpus}, max_new={args.max_new}, "
          f"lookup_n={args.lookup}, {len(prompts)} prompts")

    results = {"model": args.model, "num_gpus": args.num_gpus,
               "max_new": args.max_new, "lookup_n": args.lookup,
               "n_prompts": len(prompts), "variants": []}

    for label, lookup in [("LOOKUP_OFF", 0), ("LOOKUP_ON", args.lookup)]:
        r = _run_variant(label, lookup, args.model, args.num_gpus, prompts,
                          args.max_new, args.timeout)
        results["variants"].append(r)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))

    off = next(v for v in results["variants"] if v["label"] == "LOOKUP_OFF")
    on = next(v for v in results["variants"] if v["label"] == "LOOKUP_ON")

    lines = [
        "# T7.1 -- Prompt lookup decoding",
        "",
        f"Model: `{args.model}`, num_gpus={args.num_gpus}, "
        f"max_new_tokens={args.max_new}, VRM_PROMPT_LOOKUP={args.lookup}, "
        f"do_sample=False (greedy).",
        "",
        "| Prompt | tok/s OFF | tok/s ON | gain | outputs identical |",
        "|---|---|---|---|---|",
    ]
    if off.get("ok") and on.get("ok"):
        for i, (ro, rn) in enumerate(zip(off["results"], on["results"])):
            gain = round(100 * (rn["tok_s"] / ro["tok_s"] - 1), 1) if ro["tok_s"] else None
            identical = ro["output"] == rn["output"]
            lines.append(f"| {i} ({_FILES[i]}) | {ro['tok_s']} | {rn['tok_s']} | "
                          f"{gain}% | {identical} |")
    else:
        lines.append(f"| FAIL | | | | OFF.ok={off.get('ok')} ON.ok={on.get('ok')} |")
        if not off.get("ok"):
            lines.append("```\n" + off.get("stderr_tail", "") + "\n```")
        if not on.get("ok"):
            lines.append("```\n" + on.get("stderr_tail", "") + "\n```")

    OUT_MD.write_text("\n".join(lines))
    print(f"\nWrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
