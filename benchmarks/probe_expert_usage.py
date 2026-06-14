#!/usr/bin/env python3
"""Probe T7.11 — distribution de fréquence d'activation des experts MoE.

LA mesure qui décide si le MoE-tiering a une carte :
- distribution PIQUÉE (top-k experts couvrent l'essentiel) -> chauds résidents /
  froids streamés -> peu de transfert -> tiering GAGNE.
- distribution PLATE (experts ~équiprobables) -> il faut tout streamer -> tiering PERD.

Méthode : post_hook sur chaque `mlp.gate` (nn.Linear -> router_logits), on calcule
le top-k et on compte les activations par (couche, expert). On distingue le PREFILL
(forward du prompt, seq>1, union d'experts) du DÉCODE (seq==1, top-k par token).

Usage: python benchmarks/probe_expert_usage.py [modele]
"""
import os, sys, json, collections
os.environ["VRM_DISABLE_TURBO"] = "1"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen1.5-MoE-A2.7B"
PROMPTS = [
    "Write a Python function that parses a CSV file and returns a dict.",
    "Implement a binary search tree in Python with insert and search.",
    "Refactor this loop into a list comprehension: result=[]\nfor x in data:\n  if x>0: result.append(x*2)",
]
MAXNEW = 80

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
print("[load] device_map=auto ...", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")

cfg = model.config
NUM_EXPERTS = getattr(cfg, "num_experts", getattr(cfg, "n_routed_experts", None))
TOPK = getattr(cfg, "num_experts_per_tok", getattr(cfg, "moe_topk", 8))
print(f"num_experts={NUM_EXPERTS} top_k={TOPK}", flush=True)

# Compteurs
counts = collections.defaultdict(collections.Counter)   # layer -> Counter(expert -> activations)
prefill_unique = {}                                     # layer -> nb experts uniques au prefill
decode_tokens = collections.Counter()                   # layer -> nb tokens décode vus

def make_hook(lidx):
    def hook(module, inp, out):
        logits = out[0] if isinstance(out, tuple) else out
        rl = logits.reshape(-1, logits.shape[-1])        # [tokens, num_experts]
        seq = rl.shape[0]
        topk = torch.topk(rl, k=TOPK, dim=-1).indices    # [tokens, top_k]
        flat = topk.reshape(-1).tolist()
        counts[lidx].update(flat)
        if seq > 1:
            prefill_unique[lidx] = len(set(flat))
        else:
            decode_tokens[lidx] += 1
    return hook

# Hook chaque couche dont le mlp a un `gate`
n_moe = 0
for i, layer in enumerate(model.model.layers):
    mlp = getattr(layer, "mlp", None)
    gate = getattr(mlp, "gate", None)
    if gate is not None and isinstance(gate, torch.nn.Linear):
        gate.register_forward_hook(make_hook(i))
        n_moe += 1
print(f"{n_moe} couches MoE hookées", flush=True)

for p in PROMPTS:
    ids = tok(p, return_tensors="pt").input_ids.to(model.device if hasattr(model, "device") else "cuda:0")
    model.generate(ids, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)

# Agrégat sur toutes les couches
agg = collections.Counter()
for lidx, c in counts.items():
    agg.update(c)
total = sum(agg.values())
ranked = agg.most_common()
def coverage(k):
    return round(100 * sum(v for _, v in ranked[:k]) / total, 1) if total else 0

res = {
    "model": MODEL, "num_experts": NUM_EXPERTS, "top_k": TOPK,
    "n_moe_layers": n_moe, "total_activations": total,
    "coverage_top4_pct": coverage(4), "coverage_top8_pct": coverage(8),
    "coverage_top16_pct": coverage(16),
    "coverage_top25pct_pct": coverage(max(1, NUM_EXPERTS // 4)) if NUM_EXPERTS else None,
    "uniform_expectation_top8_pct": round(100 * 8 / NUM_EXPERTS, 1) if NUM_EXPERTS else None,
    "prefill_unique_avg": round(sum(prefill_unique.values()) / max(1, len(prefill_unique)), 1),
}
print("RESULT_JSON:" + json.dumps(res))
print("\n=== Distribution des experts (agrégé toutes couches) ===")
print(f"num_experts={NUM_EXPERTS}, top_k/token={TOPK}, couches MoE={n_moe}")
print(f"Couverture top4  : {res['coverage_top4_pct']}%")
print(f"Couverture top8  : {res['coverage_top8_pct']}%  (uniforme attendrait {res['uniform_expectation_top8_pct']}%)")
print(f"Couverture top16 : {res['coverage_top16_pct']}%")
print(f"Couverture top 25% des experts : {res['coverage_top25pct_pct']}%")
print(f"Experts uniques au PREFILL (moy/couche) : {res['prefill_unique_avg']} / {NUM_EXPERTS}")
print("\n-> Si top8 >> uniforme et top25% > ~80% : PIQUÉE -> tiering MoE gagne.")
print("-> Si proche de l'uniforme : PLATE -> tiering MoE perd (à documenter).")
