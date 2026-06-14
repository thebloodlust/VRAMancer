#!/usr/bin/env python3
"""Tiering v0.1 — prefetch DOUBLE-BUFFER (2 buffers partagés, VRAM-efficient).

Corrige la race du v0 : le prefetch écrivait un buffer encore lu par une couche
précédente. Fix robuste = `prefetch_stream.wait_stream(default)` AVANT chaque
copie -> le prefetch attend que tout le calcul déjà enqueué (dont l'ancien
utilisateur du buffer) soit fini, puis écrit. Et `default.wait_event(prefetch)`
avant d'utiliser un buffer (poids prêts).

2 buffers seulement (slot pos%2) -> économie VRAM réelle même avec N couches froides
(contrairement au v0 buffer-par-couche).

RÉSULTAT MESURÉ (1.5B, 4 couches froides) :
- ✅ CORRECT + VRAM-efficient (2 buffers, 357 MB économisés), tok/s = 73.1%.
- L'aliasing (2 couches dont les params pointent le même buffer réécrit) était LE
  bug ; le post_hook qui repointe sur le master cuda:1 avant la réutilisation du
  buffer le règle. (wait_stream gère la libération du buffer.)
- MAIS gain marginal vs POC (71.7%) : `torch.copy_` plafonne à ~10 GB/s, donc le
  transfert reste lourd et la synchro mange l'overlap. Comparaison :
    POC un-sens 71.7 | v0 per-layer 78.3 (pas VRAM-eff) | v0.1 fresh 67.8 (alloc) |
    v0.1 2-buf+repoint 73.1 (cette version).
- CONCLUSION : le vrai levier = la BANDE PASSANTE du transfert. -> v0.2 = remplacer
  torch.copy_ par GpuPipeline (25 GB/s, ~2.5x) -> bien moins à cacher -> vrai gain.
"""
import os, sys, time, json
os.environ["VRM_DISABLE_TURBO"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Write a Python function that parses a CSV file and returns a dict."
MAXNEW = 50
COLD = [0, 1, 2, 3]
COMPUTE, STORAGE = "cuda:0", "cuda:1"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
ids = tok(PROMPT, return_tensors="pt").input_ids.to(COMPUTE)

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map={"": 0})
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
        p.data = p.data.to(STORAGE)
        masters[(idx, name)] = p.data
torch.cuda.synchronize()
vram_off = torch.cuda.memory_allocated(0)

# 2 buffers PARTAGÉS (rapide, VRAM-efficient) + repoint-master en post (anti-aliasing).
def make_bufset(layer):
    return {name: torch.empty_like(p, device=COMPUTE) for name, p in layer.named_parameters()}
bufsets = [make_bufset(layers[COLD[0]]), make_bufset(layers[COLD[0]])]

pstream = torch.cuda.Stream(device=0)
pdone = {}    # idx -> event "prefetch terminé"

def prefetch(idx, slot):
    pstream.wait_stream(torch.cuda.current_stream(0))   # buffer-free (ancien user fini)
    with torch.cuda.stream(pstream):
        for name, _ in layers[idx].named_parameters():
            bufsets[slot][name].copy_(masters[(idx, name)], non_blocking=True)
    ev = torch.cuda.Event(); ev.record(pstream)
    pdone[idx] = ev

def make_pre(idx, pos):
    slot = pos % 2
    def pre(module, inp):
        if idx not in pdone:
            prefetch(idx, slot)
        torch.cuda.current_stream(0).wait_event(pdone[idx])
        for name, p in module.named_parameters():
            p.data = bufsets[slot][name]
        if pos + 1 < len(COLD):
            prefetch(COLD[pos + 1], (pos + 1) % 2)
        return None
    return pre

def make_post(idx):
    def post(module, inp, out):
        for name, p in module.named_parameters():
            p.data = masters[(idx, name)]     # retire l'alias avant réutilisation du buffer
        pdone.pop(idx, None)
        return None
    return post

handles = []
for pos, idx in enumerate(COLD):
    handles.append(layers[idx].register_forward_pre_hook(make_pre(idx, pos)))
    handles.append(layers[idx].register_forward_hook(make_post(idx)))

off_text, off_toks = gen_timed()

res = {"model": MODEL, "cold_layers": COLD, "n_buffers": 2,
       "identical": ref_text.strip() == off_text.strip(),
       "vram_saved_mb": round((vram_full - vram_off) / 1024**2, 1),
       "tok_s_ref": ref_toks, "tok_s_v0_1": off_toks,
       "tok_s_ratio_pct": round(100 * off_toks / ref_toks, 1) if ref_toks else 0,
       "off": off_text[:140]}
print("RESULT_JSON:" + json.dumps(res))
print(f"1. identique : {res['identical']}")
print(f"2. VRAM éco  : {res['vram_saved_mb']} MB (2 buffers partagés)")
print(f"3. tok/s ref={ref_toks} v0.1={off_toks} ({res['tok_s_ratio_pct']}%)  [POC 71.7, v0 78.3]")
print("OFF:", repr(off_text[:120]))
