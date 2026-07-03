#!/usr/bin/env python3
"""Test 2 — voie sûre accelerate : 14B multi-GPU avec fix OOM -> baseline.

A1 Path 1 (accelerate) avait OOM au chargement (_initialize_missing_keys ->
init.normal_(weight.float()) upcast fp32 sur GPU plein). DeepSeek : max_memory
par GPU + expandable_segments. On vérifie que le chemin accelerate (celui qui
sous-tend toute la voie B du tiering) :
  1. charge le 14B sans OOM,
  2. produit une sortie CORRECTE (contrairement au forward manuel cassé),
  3. donne le tok/s baseline (que A1 n'avait pas).
"""
import os, time, json, collections
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VRM_DISABLE_TURBO"] = "1"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-14B-Instruct"
PROMPT = "Write a Python function that parses a CSV file and returns a dict."
MAXNEW = 64

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id

print("[accelerate] chargement 14B device_map=auto + max_memory ...", flush=True)
m = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto",
    max_memory={0: "22GiB", 1: "14GiB", "cpu": "32GiB"},
    low_cpu_mem_usage=True,
)
split = dict(collections.Counter(str(v) for v in m.hf_device_map.values()))
print("répartition couches -> device:", split, flush=True)

ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
_ = m.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
torch.cuda.synchronize()
t0 = time.perf_counter()
out = m.generate(ids, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

res = {"model": MODEL, "device_split": split,
       "tok_s": round(MAXNEW / dt, 2), "output": text[:400]}
print("RESULT_JSON:" + json.dumps(res))
print("\n===== ACCELERATE 14B multi-GPU =====\n" + repr(text[:300]))
print("\ntok/s:", res["tok_s"], "| répartition:", split)
