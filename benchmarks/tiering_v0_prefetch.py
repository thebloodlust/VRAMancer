#!/usr/bin/env python3
"""Tiering v0 — prefetch double-buffer : cacher le swap GPU1->GPU0 derrière le calcul.

Étend le POC (`poc_tiering_offload_gpu1.py`). Couches froides consécutives sur
cuda:1 (master). Pendant que la couche N calcule sur cuda:0, on précharge les
poids de N+1 sur un **stream dédié** dans le buffer alterné (double-buffer).
La couche N+1 attend juste l'event de son prefetch (déjà ~fini) au lieu de copier.

Gate : sortie greedy IDENTIQUE au ref (la synchro stream/event est correcte),
puis tok/s vs POC un-sens (25.32 / 71.7% sur 1.5B / 4 couches).

RÉSULTAT MESURÉ (1.5B, 4 couches) :
- Prefetch buffer DÉDIÉ/couche (cette version) : ✅ correct, 78.3% (vs 71.7% POC).
  Concept validé : le prefetch cache une partie du transfert -> +6.6 pts.
  LIMITE : 1 buffer/couche -> pas d'économie VRAM à l'échelle (toutes les couches
  froides finiraient sur GPU0). OK pour la démo, pas pour la prod.
- Prefetch DOUBLE-BUFFER (2 buffers partagés, VRAM-efficient) : a donné 86.4% MAIS
  sortie CASSÉE -> race de synchro double-buffer (hand-roll torch, non résolue).
  -> La version prod (VRAM-efficient + correcte) doit passer par GpuPipeline (Rust,
  triple-buffer pinned, conçu pour ça) = v0.1. Le hand-roll torch n'est pas le bon outil.
"""
import os, sys, time, json
os.environ["VRM_DISABLE_TURBO"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Write a Python function that parses a CSV file and returns a dict."
MAXNEW = 50
COLD = [0, 1, 2, 3]          # couches froides consécutives
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

# Masters sur cuda:1
masters = {}
for idx in COLD:
    for name, p in layers[idx].named_parameters():
        p.data = p.data.to(STORAGE)
        masters[(idx, name)] = p.data
torch.cuda.synchronize()
vram_off = torch.cuda.memory_allocated(0)

# DIAGNOSTIC : un buffer DÉDIÉ par couche froide (pas de partage -> pas de race buffer)
def make_bufset(layer):
    return {name: torch.empty_like(p, device=COMPUTE) for name, p in layer.named_parameters()}
bufs = {idx: make_bufset(layers[idx]) for idx in COLD}

pstream = torch.cuda.Stream(device=0)
events = {}                 # idx -> event "prefetch de idx terminé" (sur pstream)

def prefetch(idx):
    with torch.cuda.stream(pstream):
        for name, _ in layers[idx].named_parameters():
            bufs[idx][name].copy_(masters[(idx, name)], non_blocking=True)
    ev = torch.cuda.Event(); ev.record(pstream)
    events[idx] = ev

def make_pre(idx, pos):
    def pre(module, inp):
        if idx not in events:                 # 1re couche froide : prefetch
            prefetch(idx)
        torch.cuda.current_stream(0).wait_event(events[idx])  # poids prêts sur cuda:0
        for name, p in module.named_parameters():
            p.data = bufs[idx][name]
        if pos + 1 < len(COLD):               # précharge la suivante pendant le calcul
            prefetch(COLD[pos + 1])
        return None
    return pre

handles = []
for pos, idx in enumerate(COLD):
    handles.append(layers[idx].register_forward_pre_hook(make_pre(idx, pos)))

off_text, off_toks = gen_timed()

res = {"model": MODEL, "cold_layers": COLD,
       "identical": ref_text.strip() == off_text.strip(),
       "vram_saved_mb": round((vram_full - vram_off) / 1024**2, 1),
       "tok_s_ref": ref_toks, "tok_s_prefetch": off_toks,
       "tok_s_ratio_pct": round(100 * off_toks / ref_toks, 1) if ref_toks else 0,
       "ref": ref_text[:160], "off": off_text[:160]}
print("RESULT_JSON:" + json.dumps(res))
print(f"\n1. identique     : {res['identical']}")
print(f"2. VRAM économisée: {res['vram_saved_mb']} MB")
print(f"3. tok/s ref={ref_toks} prefetch={off_toks} ({res['tok_s_ratio_pct']}%)")
print(f"   (rappel POC un-sens : 71.7%)")
print("REF:", repr(ref_text[:140]))
print("OFF:", repr(off_text[:140]))
