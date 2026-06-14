#!/usr/bin/env python3
"""S1 — test comparatif des 2 variantes de vramancer.patch() (léger vs lourd).

La méthode du projet : on teste les deux, on garde celle qui a des résultats.
Mesure sur TinyLlama-1.1B-Chat (en cache, petit modèle réel décodeur).

Pour chaque variante : ça charge ? ça génère du texte cohérent ? coût de chargement,
injection correcte (device_map/max_memory), API (HF-native vs custom), machinerie.

Usage: python benchmarks/test_s1_patch.py [modele]
"""
import os, sys, time, json, traceback
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "Write a Python function to compute the factorial of n."
MAXNEW = 40


def _gen_light(model, tok):
    import torch
    ids = tok(PROMPT, return_tensors="pt").input_ids.to(model.device)
    t = time.perf_counter()
    out = model.generate(ids, max_new_tokens=MAXNEW, do_sample=False)
    dt = time.perf_counter() - t
    txt = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return txt, dt


def run_light():
    import vramancer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    res = {"variant": "light", "ok": False}
    captured = {}
    # Espionne les kwargs injectés (le module est vramancer.dropin)
    import vramancer.dropin as vp
    orig_cmm = vp.compute_max_memory
    vp.compute_max_memory = lambda *a, **k: (captured.__setitem__("mm", orig_cmm(*a, **k)) or captured["mm"])
    try:
        vramancer.patch("light")
        tok = AutoTokenizer.from_pretrained(MODEL)
        t = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto")
        res["load_s"] = round(time.perf_counter() - t, 2)
        res["device_map_injected"] = "auto" if hasattr(model, "hf_device_map") else "single"
        res["max_memory"] = captured.get("mm")
        res["generate_wrapped"] = getattr(model, "_vrm_generate_wrapped", False)
        res["is_hf_module"] = hasattr(model, "forward") and hasattr(model, "config")
        txt, gdt = _gen_light(model, tok)
        res["gen_s"] = round(gdt, 2)
        res["output"] = txt.strip()[:120]
        res["ok"] = len(txt.strip()) > 0
        del model
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc().splitlines()[-3:]
    finally:
        vp.compute_max_memory = orig_cmm
        vramancer.unpatch()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
    return res


def run_heavy():
    import vramancer
    from transformers import AutoModelForCausalLM
    res = {"variant": "heavy", "ok": False}
    try:
        vramancer.patch("heavy")
        t = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(MODEL)
        res["load_s"] = round(time.perf_counter() - t, 2)
        res["is_hf_module"] = hasattr(model, "config")  # attendu: False (objet custom)
        res["api"] = "custom .generate(prompt:str)->str"
        t = time.perf_counter()
        txt = model.generate(PROMPT, max_new_tokens=MAXNEW)
        res["gen_s"] = round(time.perf_counter() - t, 2)
        res["output"] = (txt or "").strip()[:120]
        res["ok"] = bool(txt and txt.strip())
        try:
            model.pipeline.shutdown()
        except Exception:
            pass
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
        res["trace"] = traceback.format_exc().splitlines()[-4:]
    finally:
        vramancer.unpatch()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass
    return res


if __name__ == "__main__":
    print(f"[S1 test] modèle={MODEL}\n", flush=True)
    light = run_light()
    print("LIGHT:", json.dumps(light, ensure_ascii=False, indent=2), flush=True)
    heavy = run_heavy()
    print("\nHEAVY:", json.dumps(heavy, ensure_ascii=False, indent=2), flush=True)
    print("\nRESULT_JSON:" + json.dumps({"light": light, "heavy": heavy}, ensure_ascii=False))
