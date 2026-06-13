#!/usr/bin/env python3
"""POC tiering — GPU1 comme magasin de poids, GPU0 calcule (voie accelerate, voie B).

Valide le MÉCANISME (pas encore la valeur) du design convergé Opus↔DeepSeek :
- modèle chargé entièrement sur cuda:0 (compute) ;
- 2 couches "froides" déplacées sur cuda:1 (magasin) APRÈS chargement ;
- pre_forward_hook : ramène les poids cuda:1 -> cuda:0 juste avant le forward ;
- post_forward_hook : renvoie les poids sur cuda:1 (libère cuda:0).
Le forward reste natif (model.generate). On ne touche PAS la logique d'inférence.

3 critères de succès :
  1. sortie greedy IDENTIQUE au run sans offload (cohérence device OK) ;
  2. VRAM cuda:0 réellement économisée (= taille des couches offloadées) ;
  3. tok/s avec offload >= sans offload - 30% (swap torch.to, GpuPipeline plus tard).

NB : swap via torch `.to()` (simple, valide la correction). GpuPipeline = optim
de transport, étape suivante. POC en BF16 (pas de sous-classe FP4).
"""
import os, sys, time, json
os.environ["VRM_DISABLE_TURBO"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
PROMPT = "Write a Python function that parses a CSV file and returns a dict."
MAXNEW = 50
COLD = [0, 1, 2, 3]   # couches déplacées sur cuda:1
COMPUTE, STORAGE = "cuda:0", "cuda:1"

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
ids = tok(PROMPT, return_tensors="pt").input_ids.to(COMPUTE)

print("[load] tout sur cuda:0 ...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map={"": 0})


def gen_timed():
    _ = model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True), round(MAXNEW / dt, 2)


# --- 1. Référence : sans offload ---
ref_text, ref_toks = gen_timed()
vram_full = torch.cuda.memory_allocated(0)

# --- 2. Déplacer les couches froides sur cuda:1 (magasin) ---
def layers_of(m):
    return m.model.layers

for idx in COLD:
    for p in layers_of(model)[idx].parameters():
        p.data = p.data.to(STORAGE)   # libère cuda:0
torch.cuda.synchronize()
vram_offloaded = torch.cuda.memory_allocated(0)

# --- 3. Hooks : swap-in UN SEUL SENS (poids read-only en inference) ---
# Master des poids gardé sur cuda:1 ; on copie vers cuda:0 juste pour le calcul,
# puis on re-pointe sur le master (la copie cuda:0 est libérée par le GC).
# -> 1 seul transfert cuda:1->cuda:0 par couche/token (pas de retour inutile).
_masters = {}
for idx in COLD:
    for p in layers_of(model)[idx].parameters():
        _masters[id(p)] = p.data  # déjà sur cuda:1

def make_pre(layer):
    def pre(module, inp):
        for p in layer.parameters():
            p.data = _masters[id(p)].to(COMPUTE, non_blocking=False)
        return None
    return pre

def make_post(layer):
    def post(module, inp, out):
        for p in layer.parameters():
            p.data = _masters[id(p)]  # repointe sur cuda:1 (0 transfert)
        return None
    return post

handles = []
for idx in COLD:
    L = layers_of(model)[idx]
    handles.append(L.register_forward_pre_hook(make_pre(L)))
    handles.append(L.register_forward_hook(make_post(L)))

# --- 4. Run avec offload ---
off_text, off_toks = gen_timed()

res = {
    "model": MODEL, "cold_layers": COLD,
    "identical": ref_text.strip() == off_text.strip(),
    "vram_full_mb": round(vram_full / 1024**2, 1),
    "vram_offloaded_mb": round(vram_offloaded / 1024**2, 1),
    "vram_saved_mb": round((vram_full - vram_offloaded) / 1024**2, 1),
    "tok_s_ref": ref_toks, "tok_s_offload": off_toks,
    "tok_s_ratio_pct": round(100 * off_toks / ref_toks, 1) if ref_toks else 0,
    "ref_text": ref_text[:200], "offload_text": off_text[:200],
}
print("RESULT_JSON:" + json.dumps(res))
print("\n=== CRITÈRES ===")
print(f"1. sortie identique     : {res['identical']}")
print(f"2. VRAM cuda:0 économisée: {res['vram_saved_mb']} MB ({len(COLD)} couches)")
print(f"3. tok/s ref={res['tok_s_ref']} offload={res['tok_s_offload']} "
      f"({res['tok_s_ratio_pct']}% du ref)")
print("\nREF    :", repr(ref_text[:160]))
print("OFFLOAD:", repr(off_text[:160]))
