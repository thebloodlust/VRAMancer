#!/usr/bin/env python3
"""Tiering v0.2 — swap-in via GpuPipeline (Rust, ~25 GB/s pinned) au lieu de torch.copy_.

v0.1 plafonnait à ~73% car torch.copy_ ~10 GB/s. GpuPipeline (triple-buffer pinned,
25 GB/s mesuré) transfère ~2.5x plus vite -> le swap coûte moins -> tok/s plus haut.

GpuPipeline.transfer est SYNCHRONE -> pas de prefetch (inutile : transfert rapide).
Structure = POC un-sens : 1 buffer partagé (sync, 1 couche active à la fois) +
post_hook repoint-master (anti-aliasing, libère). VRAM-efficient.

NB : nécessite torch CUDA initialisé sur les 2 GPU AVANT de construire GpuPipeline
(sinon cuDevicePrimaryCtxRetain échoue).

RÉSULTAT MESURÉ (1.5B, 4 couches) — CONTRE-INTUITIF :
- ✅ correct + VRAM-efficient, MAIS tok/s = 61.2% << v0.1 torch.copy_ (73.1%).
- POURQUOI : is_p2p=False -> GpuPipeline CPU-stage (2 sauts) + overhead de setup
  par appel. On transfère PAR PARAMÈTRE (petits ~MB sur un 1.5B) -> l'overhead
  domine. Le 25 GB/s ne vaut que pour de GROS transferts (256 MB), pas des petits.
- LEÇON (mesure > intuition) : GpuPipeline n'aide PAS le tiering de petites couches.
  Il ne brillerait que sur de gros transferts (couches 14B ~0.5 Go), non testable
  ici (14B ne tient pas sur 1 GPU pour la réf).
- => Pour ce setup, v0.1 (torch.copy_, 73.1%) reste le meilleur. Le tiering a un
  coût transfert-bound (~27%) qui ne se règle pas par GpuPipeline sur petites couches.
  La vraie valeur = MoE (on ne streame que les experts actifs = peu de transfert).
"""
import os, sys, time, json
os.environ["VRM_DISABLE_TURBO"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import vramancer_rust as vr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Write a Python function that parses a CSV file and returns a dict."
MAXNEW = 50
COLD = [0, 1, 2, 3]
COMPUTE, STORAGE = "cuda:0", "cuda:1"

# Init contextes CUDA des 2 GPU AVANT GpuPipeline
torch.zeros(1, device=COMPUTE); torch.zeros(1, device=STORAGE)
torch.cuda.synchronize(0); torch.cuda.synchronize(1)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
ids = tok(PROMPT, return_tensors="pt").input_ids.to(COMPUTE)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map={"": 0})
layers = model.model.layers


def gen_timed():
    _ = model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True), round(MAXNEW/(time.perf_counter()-t0), 2)


ref_text, ref_toks = gen_timed()
vram_full = torch.cuda.memory_allocated(0)

masters = {}
for idx in COLD:
    for name, p in layers[idx].named_parameters():
        p.data = p.data.to(STORAGE).contiguous()
        masters[(idx, name)] = p.data
torch.cuda.synchronize()
vram_off = torch.cuda.memory_allocated(0)

gp = vr.GpuPipeline(1, 0, 4)   # storage=1 -> compute=0, chunk 4 MB
print("GpuPipeline is_p2p =", gp.is_p2p(), flush=True)

# 1 buffer partagé (sync -> 1 couche à la fois) + repoint master
bufset = {name: torch.empty_like(p, device=COMPUTE).contiguous()
          for name, p in layers[COLD[0]].named_parameters()}

def make_pre(idx):
    def pre(module, inp):
        for name, p in module.named_parameters():
            m = masters[(idx, name)]
            buf = bufset[name]
            gp.transfer(m.data_ptr(), buf.data_ptr(), m.numel() * m.element_size())
            p.data = buf
        return None
    return pre

def make_post(idx):
    def post(module, inp, out):
        for name, p in module.named_parameters():
            p.data = masters[(idx, name)]
        return None
    return post

handles = []
for idx in COLD:
    handles.append(layers[idx].register_forward_pre_hook(make_pre(idx)))
    handles.append(layers[idx].register_forward_hook(make_post(idx)))

off_text, off_toks = gen_timed()

res = {"model": MODEL, "cold_layers": COLD, "is_p2p": gp.is_p2p(),
       "identical": ref_text.strip() == off_text.strip(),
       "vram_saved_mb": round((vram_full - vram_off) / 1024**2, 1),
       "tok_s_ref": ref_toks, "tok_s_v0_2": off_toks,
       "tok_s_ratio_pct": round(100 * off_toks / ref_toks, 1) if ref_toks else 0,
       "off": off_text[:140]}
print("RESULT_JSON:" + json.dumps(res))
print(f"1. identique : {res['identical']}")
print(f"2. VRAM éco  : {res['vram_saved_mb']} MB (1 buffer, GpuPipeline)")
print(f"3. tok/s ref={ref_toks} v0.2={off_toks} ({res['tok_s_ratio_pct']}%)  "
      f"[POC 71.7 | v0.1 2-buf 73.1]")
print("OFF:", repr(off_text[:120]))
