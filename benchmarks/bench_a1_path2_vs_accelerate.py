#!/usr/bin/env python3
"""A1 — Parité Path 2 (split manuel VRAMancer + GpuPipeline) vs Path 1 (accelerate).

decision_architecte_7.md §3, palier A1. Objectif : PROUVER, avant de basculer la
prod (A2), que le chemin « split manuel » donne les MÊMES sorties qu'accelerate,
sans perte de débit notable.

Critères de passage A1 (architecte) :
  (a) sorties greedy IDENTIQUES au token près (256 tokens, prompt fixe ci-dessous) ;
  (b) tok/s Path 2 >= tok/s accelerate - 5 %.

Les deux chemins (cf. core/backends.py) :
  Path 1 : device_map="auto" -> hf_device_map -> split_model() garde accelerate,
           blocks=None -> generate() = model.generate() natif.
  Path 2 : chargement SANS device_map="auto" -> split_model(2) retire les hooks
           accelerate, peuple self.blocks (>1) -> forward manuel infer()
           (embed -> blocs -> norm -> head, transferts hidden_states.to(block_dev)).

⚠️ STATUT : SCAFFOLD NON TESTÉ — à exécuter sur le desktop 2 GPU (3090+5070Ti),
   serveur Qwen3.6 mis en pause (contention VRAM). L'invocation exacte de Path 2
   (load CPU + split_model) reste à confirmer sur matériel — voir
   QUESTION_DEEPSEEK_A1_PATH2.md (Q-A1.1/.2/.4). Le garde-fou _assert_path ci-dessous
   échoue FORT si un chemin n'est pas réellement emprunté (risque #1 : benchmarker
   deux fois Path 1 sans le voir).

Usage :
    # libérer les GPU d'abord (arrêter le serveur Qwen3.6)
    python benchmarks/bench_a1_path2_vs_accelerate.py --model Qwen/Qwen2.5-14B-Instruct
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path

# Prompt fixe du protocole Phase 7 (7.md, règle 3 de non-régression qualité).
FIXED_PROMPT = "Write a Python function that parses a CSV file and returns a dict."

DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"
OUT_JSON = Path("benchmarks/results/phase7/A1_path2_vs_accelerate.json")
OUT_MD = Path("benchmarks/results/phase7/A1_path2_vs_accelerate.md")

# Préambule commun à chaque worker : env propre, GPU visibles, alloc.
_PREAMBLE = """
import os, sys, json, time
os.environ.pop('CUDA_VISIBLE_DEVICES', None)
os.environ.pop('VRM_MINIMAL_TEST', None)
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['VRM_VRAM_LENDING'] = '0'
os.environ['VRM_DISABLE_TURBO'] = '1'   # mesurer le chemin nu, pas TurboEngine
import torch
"""

# --- Worker Path 1 : accelerate device_map="auto" -------------------------
_WORKER_PATH1 = _PREAMBLE + """
os.environ['VRM_FORCE_MULTI_GPU'] = '1'
from core.inference_pipeline import InferencePipeline, reset_pipeline
reset_pipeline()
pipe = InferencePipeline(backend_name='huggingface', enable_metrics=False,
                         enable_discovery=False, verbose=False)
pipe.load(MODEL, num_gpus=2)

# Garde-fou : on DOIT être sur accelerate (blocks None + hf_device_map présent).
be = getattr(pipe, 'backend', None)
blocks = getattr(be, 'blocks', None)
has_devmap = bool(getattr(getattr(be, 'model', None), 'hf_device_map', None))
path_used = 'accelerate' if (blocks is None and has_devmap) else 'UNKNOWN'

out, toks = _gen(pipe.generate)
print('RESULT_JSON:' + json.dumps({
    'path': 1, 'path_used': path_used, 'has_device_map': has_devmap,
    'n_blocks': (len(blocks) if blocks else 0),
    'output': out, 'tok_s': toks['tok_s'], 'elapsed_s': toks['elapsed_s'],
}))
"""

# --- Worker Path 2 : split manuel VRAMancer -------------------------------
# Invocation confirmée par DeepSeek (Q-A1.1) ET vérifiée dans le code :
# charge le modèle en CPU (device_map=None) pour éviter hf_device_map, puis
# split_model(2) qui retire les hooks accelerate et répartit les blocs
# (assign_blocks_to_gpus déplace réellement CPU->cuda).
#
# Parité rotary — crainte DeepSeek Q-A1.2 VÉRIFIÉE FAUSSE pour Qwen2.5 :
# _POS_EMBED_PATTERNS ne matche que GPT-2 (`transformer.wpe`), absent de Qwen ->
# comp["pos_embed"] = None -> le bloc additif `hidden_states + pos_emb` (1642)
# est sauté. Le rotary est appliqué UNE fois via comp["rotary_emb"] passé en
# position_embeddings à chaque bloc (1764+). Pas de double application ici.
# (Sur un modèle GPT-2 ce serait à revérifier.)
_WORKER_PATH2 = _PREAMBLE + """
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.backends import HuggingFaceBackend

be = HuggingFaceBackend(MODEL)
# Chargement CPU/RAM (~28 Go bf16) SANS device_map auto -> pas de hf_device_map.
be.model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map=None, low_cpu_mem_usage=True)
be.tokenizer = AutoTokenizer.from_pretrained(MODEL)

# split_model(2) : retire hooks accelerate, _extract_layers + _split_by_vram +
# assign_blocks_to_gpus -> peuple be.blocks (>1) et be.block_devices.
blocks = be.split_model(2)

# Garde-fou CRITIQUE : Path 2 doit réellement être emprunté.
assert be.blocks is not None and len(be.blocks) > 1, (
    'Path 2 NON emprunté (blocks=%r) — on benchmarkerait Path 1 par erreur !'
    % (None if be.blocks is None else len(be.blocks)))

out, toks = _gen(be.generate)
print('RESULT_JSON:' + json.dumps({
    'path': 2, 'path_used': 'manual_split', 'n_blocks': len(be.blocks),
    'block_devices': [str(d) for d in (be.block_devices or [])],
    'output': out, 'tok_s': toks['tok_s'], 'elapsed_s': toks['elapsed_s'],
}))
"""

# Helper de génération injecté dans chaque worker (greedy, 256 tokens, timing).
_GEN_HELPER = """
MODEL = {model!r}
PROMPT = {prompt!r}
MAX_NEW = {max_new}

def _gen(generate_fn):
    import statistics
    # Warmup 50 tokens (rampe PCIe Gen1->Gen4, caches CUDA) -- non mesuré.
    # (DeepSeek Q-A1.3 : warmup plus long que 8.)
    _ = generate_fn(PROMPT, max_new_tokens=50, do_sample=False)
    torch.cuda.synchronize()
    samples, out = [], None
    for _r in range(3):                          # médiane de 3 (DeepSeek Q-A1.3)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = generate_fn(PROMPT, max_new_tokens=MAX_NEW, do_sample=False)
        torch.cuda.synchronize()                 # sync OBLIGATOIRE (Q-A1.4 piège 5)
        samples.append(time.perf_counter() - t0)
    dt = statistics.median(samples)
    # Prompt fixe court (~10 tokens) => prefill négligeable vs 256 decode =>
    # tok/s end-to-end ~ tok/s decode (decode-only pas nécessaire ici).
    return out, {{'elapsed_s': round(dt, 3), 'samples_s': [round(s, 3) for s in samples],
                 'tok_s': round(MAX_NEW / dt, 2) if dt > 0 else 0.0}}
"""


def _run_worker(path: int, model: str, max_new: int, timeout: int) -> dict:
    body = _WORKER_PATH1 if path == 1 else _WORKER_PATH2
    script = (_GEN_HELPER.format(model=model, prompt=FIXED_PROMPT, max_new=max_new)
              + textwrap.dedent(body))
    print(f"\n=== Path {path} ===")
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=timeout)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            r = json.loads(line[len("RESULT_JSON:"):])
            r["ok"] = True
            return r
    return {"path": path, "ok": False,
            "stdout_tail": "\n".join(proc.stdout.splitlines()[-30:]),
            "stderr_tail": "\n".join(proc.stderr.splitlines()[-40:])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--timeout", type=int, default=2400)
    args = ap.parse_args()

    print(f"[A1] Parité Path 2 vs Path 1 — model={args.model}, max_new={args.max_new}")
    print("⚠️  scaffold non testé — confirmer l'invocation Path 2 sur GPU (cf. "
          "QUESTION_DEEPSEEK_A1_PATH2.md). Serveur Qwen3.6 à mettre en pause.")

    # Process séparé par chemin : un seul 14B chargé à la fois (pas de contention).
    p1 = _run_worker(1, args.model, args.max_new, args.timeout)
    p2 = _run_worker(2, args.model, args.max_new, args.timeout)

    verdict = {"identical": None, "tok_s_ok": None, "pass": None, "notes": []}
    if p1.get("ok") and p2.get("ok"):
        identical = p1["output"] == p2["output"]          # proxy token-pour-token
        tok_ok = p2["tok_s"] >= p1["tok_s"] * 0.95
        verdict.update(identical=identical, tok_s_ok=tok_ok,
                       **{"pass": bool(identical and tok_ok)})
        # garde-fou : les deux chemins doivent être DIFFÉRENTS
        if p1.get("path_used") == "accelerate" and p2.get("path_used") == "manual_split":
            verdict["notes"].append("OK: chemins distincts confirmés "
                                    "(accelerate vs manual_split).")
        else:
            verdict["notes"].append(f"⚠️ chemins suspects: p1={p1.get('path_used')} "
                                    f"p2={p2.get('path_used')} — résultat NON fiable.")
    else:
        verdict["notes"].append("Un worker a échoué — voir stderr_tail dans le JSON.")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"model": args.model, "max_new": args.max_new, "prompt": FIXED_PROMPT,
         "path1": p1, "path2": p2, "verdict": verdict}, indent=2))
    _write_md(args, p1, p2, verdict)
    print(f"\nÉcrit {OUT_JSON} et {OUT_MD}")
    print(f"VERDICT A1: pass={verdict['pass']} identical={verdict['identical']} "
          f"tok_s_ok={verdict['tok_s_ok']}")


def _write_md(args, p1, p2, verdict):
    def tok(p): return f"{p.get('tok_s','?')} tok/s" if p.get("ok") else "ÉCHEC"
    md = textwrap.dedent(f"""\
        # A1 — Parité Path 2 (split manuel) vs Path 1 (accelerate)

        Modèle : `{args.model}` · bf16 2-GPU · greedy · {args.max_new} tokens ·
        prompt fixe : « {FIXED_PROMPT} »

        > ⚠️ Généré par un scaffold non encore validé sur matériel. Invocation
        > Path 2 à confirmer (QUESTION_DEEPSEEK_A1_PATH2.md).

        | Chemin | Emprunté (vérifié) | tok/s | sortie |
        |---|---|---|---|
        | Path 1 accelerate | {p1.get('path_used','?')} | {tok(p1)} | {len(p1.get('output','')) if p1.get('ok') else '-'} car |
        | Path 2 manuel | {p2.get('path_used','?')} ({p2.get('n_blocks','?')} blocs) | {tok(p2)} | {len(p2.get('output','')) if p2.get('ok') else '-'} car |

        ## Critères A1
        - (a) sorties identiques (token-proxy) : **{verdict['identical']}**
        - (b) tok/s Path 2 ≥ accelerate − 5 % : **{verdict['tok_s_ok']}**
        - **VERDICT : {'PASS' if verdict['pass'] else 'FAIL/À REVOIR'}**

        Notes : {' '.join(verdict['notes'])}

        ## Méthodo (après revue DeepSeek Q-A1.3/.4)
        - tok/s = **médiane de 3 runs**, warmup 50 tokens, `cuda.synchronize()`
          avant/après mesure. Prompt court => prefill négligeable => end-to-end ≈ decode.
        - Invocation Path 2 (load CPU + `split_model`) **confirmée** (Q-A1.1).
        - Rotary : crainte de double-application **vérifiée fausse** sur Qwen2.5
          (`pos_embed` None ; rotary appliqué une fois). Voir commentaire du worker.

        ## Limites restantes
        - Comparaison de sortie = chaînes décodées exactes (proxy token-pour-token) ;
          pour une vraie égalité d'IDs, exposer `out_ids` côté backend.
        - Pièges Path 2 à surveiller au run (DeepSeek Q-A1.4) : sync des streams de
          transfert inter-GPU (le code fait `wait_stream` à 1738), propagation du
          `DynamicCache` entre étapes (sinon decode recalcule tout => tok/s effondré).
        """)
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(md)


if __name__ == "__main__":
    main()
