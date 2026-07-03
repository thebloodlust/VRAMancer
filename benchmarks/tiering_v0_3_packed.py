#!/usr/bin/env python3
"""Tiering v0.3 — transfert PACKÉ via GpuPipeline (1 gros transfert/couche).

v0.2 : GpuPipeline par-paramètre = lent (overhead par appel sur petits transferts).
v0.3 : packer tous les params d'une couche en UN buffer contigu sur GPU1, faire
UN SEUL gp.transfer (gros -> la BW 25 GB/s s'exprime), puis re-viewer le dst.
Hypothèse DeepSeek : devrait dépasser torch (73.1%), viser ~80-85%.

1 buffer GPU0 partagé (sync) + repoint sur le master packé cuda:1 (anti-aliasing).

RÉSULTAT MESURÉ (1.5B, couche packée 89 MB) — HYPOTHÈSE RÉFUTÉE :
- ✅ correct + VRAM-eff, mais tok/s = 64.0% << torch (73.1%). Le packing aide à peine
  (v0.2 par-param 61.2% -> v0.3 packé 64.0%) mais torch.copy_ reste DEVANT.
- POURQUOI : le 25 GB/s de GpuPipeline vaut pour des transferts ISOLÉS (P2.10, 256MB).
  Dans le tiering (transfert ENTRELACÉ avec le calcul + overhead FFI + son stream
  concurrence le stream de calcul), torch.copy_ s'intègre mieux. Le contexte compte.
- CONCLUSION FERME (3 mesures GpuPipeline : par-param 61, packé 64 < torch 73) :
  pour le swap-in du tiering, **torch.copy_ est le meilleur outil**. GpuPipeline
  n'aide pas. Le coût dense (~27%) est transfert-bound et irréductible ici.
  => Meilleur tiering = v0.1 (torch double-buffer, 73.1%). La vraie valeur reste
  le MoE (peu de transfert), pas la vitesse de transfert.
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

# Packer chaque couche froide en UN buffer contigu uint8 sur cuda:1
packed = {}   # idx -> uint8 tensor (cuda:1) = master packé
vmap = {}     # idx -> {name: (offset, nbytes, shape, dtype)}
for idx in COLD:
    L = layers[idx]
    total = sum(p.numel() * p.element_size() for p in L.parameters())
    buf = torch.empty(total, dtype=torch.uint8, device=STORAGE)
    off, vm = 0, {}
    for name, p in L.named_parameters():
        nb = p.numel() * p.element_size()
        buf[off:off + nb].copy_(p.data.contiguous().view(torch.uint8).reshape(-1))
        vm[name] = (off, nb, tuple(p.shape), p.dtype)
        off += nb
    packed[idx], vmap[idx] = buf, vm
    # repoint params sur le packé cuda:1 (libère les tenseurs originaux)
    for name, p in L.named_parameters():
        o, nb, shp, dt = vm[name]
        p.data = buf[o:o + nb].view(dt).reshape(shp)
torch.cuda.synchronize()
vram_off = torch.cuda.memory_allocated(0)

gp = vr.GpuPipeline(1, 0, 4)
print("is_p2p =", gp.is_p2p(), flush=True)

# 1 buffer GPU0 partagé (taille de la plus grosse couche packée)
maxb = max(packed[i].numel() for i in COLD)
dst = torch.empty(maxb, dtype=torch.uint8, device=COMPUTE)

def make_pre(idx):
    def pre(module, inp):
        pm = packed[idx]
        gp.transfer(pm.data_ptr(), dst.data_ptr(), pm.numel())   # UN gros transfert
        for name, p in module.named_parameters():
            o, nb, shp, dt = vmap[idx][name]
            p.data = dst[o:o + nb].view(dt).reshape(shp)
        return None
    return pre

def make_post(idx):
    def post(module, inp, out):
        pm = packed[idx]
        for name, p in module.named_parameters():
            o, nb, shp, dt = vmap[idx][name]
            p.data = pm[o:o + nb].view(dt).reshape(shp)   # retour cuda:1 (anti-aliasing)
        return None
    return post

handles = []
for idx in COLD:
    handles.append(layers[idx].register_forward_pre_hook(make_pre(idx)))
    handles.append(layers[idx].register_forward_hook(make_post(idx)))

off_text, off_toks = gen_timed()

res = {"model": MODEL, "cold_layers": COLD,
       "packed_mb_per_layer": round(packed[COLD[0]].numel() / 1024**2, 1),
       "identical": ref_text.strip() == off_text.strip(),
       "vram_saved_mb": round((vram_full - vram_off) / 1024**2, 1),
       "tok_s_ref": ref_toks, "tok_s_v0_3": off_toks,
       "tok_s_ratio_pct": round(100 * off_toks / ref_toks, 1) if ref_toks else 0,
       "off": off_text[:140]}
print("RESULT_JSON:" + json.dumps(res))
print(f"1. identique : {res['identical']}")
print(f"2. VRAM éco  : {res['vram_saved_mb']} MB | couche packée: {res['packed_mb_per_layer']} MB")
print(f"3. tok/s ref={ref_toks} v0.3={off_toks} ({res['tok_s_ratio_pct']}%)  "
      f"[POC 71.7 | v0.1 73.1 | v0.2 GpuPipe/param 61.2]")
print("OFF:", repr(off_text[:120]))
