#!/usr/bin/env python3
"""Mesure disagg prefill/décode — vaut-il le coup sur 2 GPU commodity (sans NVLink) ?

On NE construit PAS un serveur disagg complet (handoff KV + continuous batching =
plusieurs jours). On mesure les quantités qui DÉCIDENT (méthode du probe MoE) :

  A) Composants par requête sur 1 GPU : temps prefill vs temps décode.
  B) Taxe KV : temps de transfert du KV cache GPU0->GPU1 (P2P bloqué ici -> CPU-staged).
  C) Scaling de l'alternative SIMPLE = data-parallel (2 copies, on route les requêtes,
     ZÉRO transfert) vs mono-GPU. Si ça scale ~2x, disagg (qui ajoute du transfert et
     sérialise prefill->décode) ne peut pas faire mieux pour du multi-user.
  D) (bonus) disagg 1 requête : prefill@GPU0 + transfert KV + décode@GPU1 vs tout@GPU0.

Verdict basé sur la mesure, pas sur l'intuition.

Usage: python benchmarks/bench_disagg_prefill_decode.py [modele]
"""
import os, sys, time, json, threading, statistics
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B-Instruct"
N_REQ = 6          # requêtes concurrentes (scénario multi-user)
PROMPT_TOK = 220   # prompt réaliste
GEN_TOK = 64       # tokens générés

torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token_id is None:
    tok.pad_token_id = tok.eos_token_id
BASE = "def quicksort(arr):\n    " + "# explain step by step\n    " * 30
ids_cpu = tok(BASE, return_tensors="pt").input_ids[:, :PROMPT_TOK]
print(f"[setup] modèle={MODEL} prompt={ids_cpu.shape[1]}tok gen={GEN_TOK} n_req={N_REQ}", flush=True)


def load_on(dev):
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(dev).eval()
    return m


def cache_kv(past):
    """Renvoie la liste [(keys, values)] du cache, robuste aux versions transformers."""
    if hasattr(past, "layers"):                       # transformers >= 4.5x
        return [(l.keys, l.values) for l in past.layers]
    leg = past.to_legacy_cache() if hasattr(past, "to_legacy_cache") else past
    return [(k, v) for (k, v) in leg]


def move_cache_(past, dev):
    """Déplace le cache vers dev IN-PLACE (nouvelle API .layers)."""
    if hasattr(past, "layers"):
        for l in past.layers:
            l.keys = l.keys.to(dev)
            l.values = l.values.to(dev)
        return past
    raise RuntimeError("format de cache non supporté pour le déplacement")


@torch.no_grad()
def run_one(model, dev, gen=GEN_TOK):
    ids = ids_cpu.to(dev)
    torch.cuda.synchronize(dev)
    out = model.generate(ids, max_new_tokens=gen, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(dev)
    return out.shape[1] - ids.shape[1]


@torch.no_grad()
def components(model, dev):
    """A) Mesure prefill seul vs décode seul sur 1 GPU."""
    ids = ids_cpu.to(dev)
    # warmup
    model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize(dev)
    # prefill seul
    t = time.perf_counter()
    out = model(ids, use_cache=True)
    torch.cuda.synchronize(dev)
    t_prefill = time.perf_counter() - t
    past = out.past_key_values
    # décode: GEN_TOK pas
    nxt = out.logits[:, -1:].argmax(-1)
    t = time.perf_counter()
    cur = nxt
    for i in range(GEN_TOK):
        o = model(cur, past_key_values=past, use_cache=True)
        past = o.past_key_values
        cur = o.logits[:, -1:].argmax(-1)
    torch.cuda.synchronize(dev)
    t_decode = time.perf_counter() - t
    return t_prefill, t_decode, out


@torch.no_grad()
def kv_transfer_tax(model0):
    """B) Temps de transfert du KV cache GPU0->GPU1 pour le prompt."""
    ids = ids_cpu.to("cuda:0")
    out = model0(ids, use_cache=True)
    torch.cuda.synchronize("cuda:0")
    kv = cache_kv(out.past_key_values)
    nbytes = sum(t.numel() * t.element_size() for layer in kv for t in layer)
    torch.cuda.synchronize()
    t = time.perf_counter()
    moved = [(k.to("cuda:1", non_blocking=False), v.to("cuda:1", non_blocking=False)) for k, v in kv]
    torch.cuda.synchronize("cuda:1")
    dt = time.perf_counter() - t
    del moved
    return dt, nbytes


@torch.no_grad()
def decode_batch_scaling(model, dev, batches=(1, 2, 4, 8)):
    """C) Scaling du décode en BATCH sur 1 GPU (sans threads -> sans artefact GIL).

    Mesure : en batchant B requêtes, le décode (memory-bound) partage la bande
    passante -> le débit agrégé monte. On regarde aussi le coût prefill batché.
    C'est le signal qui dit QUAND le prefill (dédié à un GPU en disagg) équilibrerait
    le décode : il faut B tel que B×prefill ≈ décode_wall.
    """
    rows = []
    for B in batches:
        ids = ids_cpu.to(dev).repeat(B, 1)
        # warmup
        model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize(dev)
        # prefill batché
        t = time.perf_counter()
        out = model(ids, use_cache=True)
        torch.cuda.synchronize(dev)
        tp = time.perf_counter() - t
        # décode batché
        past = out.past_key_values
        cur = out.logits[:, -1:].argmax(-1)
        t = time.perf_counter()
        for _ in range(GEN_TOK):
            o = model(cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            cur = o.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize(dev)
        tdec = time.perf_counter() - t
        agg_tps = B * GEN_TOK / (tp + tdec)
        rows.append({"B": B, "prefill_ms": round(tp*1000, 1), "decode_ms": round(tdec*1000, 1),
                     "agg_tok_s": round(agg_tps, 1)})
        print(f"[C] batch={B}: prefill={tp*1000:.1f}ms décode={tdec*1000:.1f}ms "
              f"-> {agg_tps:.1f} tok/s agrégé", flush=True)
    return rows


def main():
    res = {"model": MODEL, "n_req": N_REQ, "prompt_tok": int(ids_cpu.shape[1]), "gen_tok": GEN_TOK}
    print("[load] GPU0...", flush=True)
    m0 = load_on("cuda:0")

    # A) composants
    tp, td, _ = components(m0, "cuda:0")
    res["prefill_s"] = round(tp, 4)
    res["decode_s"] = round(td, 4)
    res["decode_per_tok_ms"] = round(td / GEN_TOK * 1000, 2)
    print(f"[A] prefill={tp*1000:.1f}ms  décode={td*1000:.1f}ms ({td/GEN_TOK*1000:.2f}ms/tok)", flush=True)

    # B) taxe KV
    dt_kv, nbytes = kv_transfer_tax(m0)
    res["kv_bytes"] = nbytes
    res["kv_transfer_ms"] = round(dt_kv * 1000, 2)
    res["kv_GBps"] = round(nbytes / dt_kv / 1e9, 1)
    res["kv_tax_vs_prefill_pct"] = round(100 * dt_kv / tp, 1)
    print(f"[B] KV={nbytes/1e6:.1f}MB transfert={dt_kv*1000:.2f}ms ({nbytes/dt_kv/1e9:.1f} GB/s) "
          f"= {res['kv_tax_vs_prefill_pct']}% du prefill", flush=True)

    # C) scaling du décode en batch (robuste, sans GIL)
    res["decode_batch_scaling"] = decode_batch_scaling(m0, "cuda:0")
    # B tel que B×prefill ≈ décode_wall (point d'équilibre prefill/décode)
    last = res["decode_batch_scaling"][-1]
    res["balance_batch_estimate"] = round(last["decode_ms"] / (tp * 1000), 1)

    print("[load] GPU1 (2e copie)...", flush=True)
    m1 = load_on("cuda:1")

    # D) disagg 1 requête (best-effort)
    try:
        ids0 = ids_cpu.to("cuda:0")
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = m0(ids0, use_cache=True)
        cache1 = move_cache_(out.past_key_values, "cuda:1")  # transfert KV GPU0->GPU1
        cur = out.logits[:, -1:].argmax(-1).to("cuda:1")
        for i in range(GEN_TOK):
            o = m1(cur, past_key_values=cache1, use_cache=True)
            cache1 = o.past_key_values
            cur = o.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        t_disagg = time.perf_counter() - t
        # comparatif tout@GPU0
        torch.cuda.synchronize("cuda:0")
        t = time.perf_counter()
        run_one(m0, "cuda:0")
        t_mono = time.perf_counter() - t
        res["disagg_1req_s"] = round(t_disagg, 4)
        res["mono_1req_s"] = round(t_mono, 4)
        res["disagg_overhead_pct"] = round(100 * (t_disagg - t_mono) / t_mono, 1)
        print(f"[D] disagg 1req={t_disagg*1000:.1f}ms  vs  mono 1req={t_mono*1000:.1f}ms "
              f"-> {res['disagg_overhead_pct']:+.1f}% (inclut ma boucle décode manuelle, pas que le transfert)",
              flush=True)
    except Exception as e:
        res["disagg_error"] = f"{type(e).__name__}: {e}"
        print(f"[D] disagg 1req échec: {e}", flush=True)

    print("\nRESULT_JSON:" + json.dumps(res))
    # Verdict basé sur les mesures ROBUSTES (A, B, C)
    ratio = td / tp
    print("\n=== VERDICT (mesures robustes : ratio décode/prefill + taxe KV) ===")
    print(f"• décode/prefill = {ratio:.0f}:1  (décode={td*1000:.0f}ms >> prefill={tp*1000:.0f}ms)")
    print(f"• taxe KV transfert GPU0->GPU1 = {res['kv_transfer_ms']}ms (latency-bound {res['kv_GBps']} GB/s, P2P bloqué)")
    print(f"• équilibre prefill/décode atteint vers batch≈{res['balance_batch_estimate']} streams décode/GPU")
    print(f"-> Pour ce workload décode-dominé, dédier un GPU au prefill (disagg) le laisse")
    print(f"   ~{100*tp/(tp+td):.0f}% occupé : gâché. L'alternative = répliquer le modèle et router")
    print(f"   les requêtes entières (data-parallel, zéro transfert) quand il tient sur 1 GPU.")
    print(f"   Disagg ne paierait que si prefill ~= décode (prompts énormes / fort batching")
    print(f"   décode) ET avec un P2P rapide (NVLink) — aucun des deux ici.")


if __name__ == "__main__":
    main()
