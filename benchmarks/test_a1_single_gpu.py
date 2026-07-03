#!/usr/bin/env python3
"""Test décisif A1 — le forward manuel `infer()` est-il correct sur 1 SEUL GPU ?

Isole : bug du forward LOGIQUE  vs  bug MULTI-GPU (placement/ordre des blocs).

Méthode : on charge un modèle, on fait le split manuel VRAMancer en 2 blocs
(comme en prod Path 2), MAIS on force **tout sur cuda:0** (blocs + composants),
donc AUCUN transfert cross-GPU. On compare la sortie greedy à la génération
native HF (référence connue-bonne) sur le même GPU.

- manuel cohérent (≈ réf) -> le forward LOGIQUE est OK ; le bug A1 était
  multi-GPU (transfert / ordre des blocs [1,0]).
- manuel dégénéré ("The following is...") -> le forward LUI-MÊME est cassé,
  indépendamment du nombre de GPU (mask/KV/composant).

Usage: python benchmarks/test_a1_single_gpu.py [modele]
"""
import os, sys, json
os.environ["VRM_DISABLE_TURBO"] = "1"
os.environ["VRM_VRAM_LENDING"] = "0"
os.environ.pop("VRM_MINIMAL_TEST", None)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Write a Python function that parses a CSV file and returns a dict."
MAXNEW = 48
DEV = "cuda:0"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

# --- 1. Référence : generate() natif HF sur cuda:0 (connu-bon) ---
print("[ref] chargement natif sur cuda:0 ...", flush=True)
m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                         device_map={"": 0})
ids = tok(PROMPT, return_tensors="pt").input_ids.to(DEV)
ref_ids = m.generate(ids, max_new_tokens=MAXNEW, do_sample=False,
                     pad_token_id=tok.pad_token_id)
ref = tok.decode(ref_ids[0][ids.shape[1]:], skip_special_tokens=True)
del m
torch.cuda.empty_cache()

# --- 2. Forward manuel VRAMancer, 2 blocs, TOUT forcé sur cuda:0 ---
print("[manuel] split + force cuda:0 ...", flush=True)
from core.backends import HuggingFaceBackend
be = HuggingFaceBackend(MODEL)
be.model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                device_map=None)  # CPU
be.tokenizer = tok
be.split_model(2)  # split réel en blocs

# Force monogpu : blocs + composants -> cuda:0, reset caches device
for blk in (be.blocks or []):
    blk.to(DEV)
if getattr(be, "_components", None):
    for k, v in list(be._components.items()):
        if v is not None and hasattr(v, "to"):
            try:
                be._components[k] = v.to(DEV)
            except Exception as e:
                print(f"  warn: move {k}: {e}")
be.block_devices = [0] * len(be.blocks or [])
be._comp_devices = {}  # re-résolution -> cuda:0

nb = len(be.blocks) if be.blocks else 0
man = be.generate(PROMPT, max_new_tokens=MAXNEW, do_sample=False)

res = {
    "model": MODEL, "n_blocks": nb,
    "block_devices_forced": [0] * nb,
    "identical": ref.strip() == man.strip(),
    "ref_native": ref[:400],
    "manual_1gpu": man[:400],
}
print("RESULT_JSON:" + json.dumps(res))
print("\n===== REF (native, connu-bon) =====\n" + repr(ref[:280]))
print("\n===== MANUEL (1 GPU, infer(), " + str(nb) + " blocs) =====\n" + repr(man[:280]))
print("\nidentical:", res["identical"])
