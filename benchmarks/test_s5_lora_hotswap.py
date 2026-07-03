#!/usr/bin/env python3
"""S5 — LoRA hot-swap : charger/switcher des adaptateurs en < 1s sans recharger la base.

Valide la mécanique (API PEFT intégrée à transformers) + chronomètre le switch.
Teste les deux chemins : add_adapter (en mémoire) ET load_adapter (depuis disque,
le cas multi-tenant réel). Modèle: Qwen2.5-0.5B (cache, petit).

Usage: python benchmarks/test_s5_lora_hotswap.py
"""
import os, sys, time, tempfile, json
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig

    # Cet env a des libs quant cassées (gptqmodel 7.0.0 API, bitsandbytes cu13) que
    # peft sonde à l'injection LoRA. Le modèle est fp16 → pas besoin de LoRA quantifié.
    # On neutralise ces dispatchers POUR LE TEST (la feature S5 utilise l'API peft
    # standard ; en env sain ces backends marchent).
    import peft.tuners.lora.model as _lm
    _lm.is_bnb_available = lambda: False
    _lm.is_bnb_4bit_available = lambda: False
    _lm.dispatch_gptq = lambda *a, **k: None
    _lm.dispatch_awq = lambda *a, **k: None

    res = {"model": MODEL}
    tok = AutoTokenizer.from_pretrained(MODEL)
    t = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda:0").eval()
    res["base_load_s"] = round(time.perf_counter() - t, 2)
    print(f"[base] chargée en {res['base_load_s']}s", flush=True)

    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0)

    # 1) add_adapter en mémoire (2 adaptateurs)
    model.add_adapter(cfg, adapter_name="tenantA")
    model.add_adapter(cfg, adapter_name="tenantB")
    print("[add] tenantA + tenantB injectés", flush=True)

    # 2) sauver tenantA sur disque puis le charger sous un nouveau nom (chemin multi-tenant réel)
    tmp = tempfile.mkdtemp(prefix="vrm_lora_")
    model.set_adapter("tenantA")
    model.save_pretrained(tmp)  # sauve l'adaptateur actif
    t = time.perf_counter()
    model.load_adapter(tmp, adapter_name="fromdisk")
    res["load_from_disk_s"] = round(time.perf_counter() - t, 3)
    print(f"[load] adaptateur disque chargé en {res['load_from_disk_s']}s", flush=True)

    # 3) chronométrer les SWITCH (le cœur du hot-swap)
    names = ["tenantA", "tenantB", "fromdisk"]
    # warmup
    model.set_adapter("tenantA")
    torch.cuda.synchronize()
    times = []
    for n in names * 3:
        t = time.perf_counter()
        model.set_adapter(n)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    res["switch_ms_avg"] = round(1000 * sum(times) / len(times), 2)
    res["switch_ms_max"] = round(1000 * max(times), 2)
    print(f"[switch] moyenne {res['switch_ms_avg']}ms, max {res['switch_ms_max']}ms", flush=True)

    # 4) désactiver (revenir à la base) + génération de contrôle
    ids = tok("def add(a,b):", return_tensors="pt").input_ids.to("cuda:0")
    model.set_adapter("tenantB")
    _ = model.generate(ids, max_new_tokens=8, do_sample=False)
    model.disable_adapters()
    _ = model.generate(ids, max_new_tokens=8, do_sample=False)
    model.enable_adapters()
    res["active_adapters"] = list(model.active_adapters()) if hasattr(model, "active_adapters") else None
    print(f"[ctrl] disable/enable OK, actifs={res['active_adapters']}", flush=True)

    # 5) unload
    model.delete_adapter("fromdisk")
    res["after_unload_ok"] = "fromdisk" not in (model.peft_config or {})
    print(f"[unload] fromdisk supprimé: {res['after_unload_ok']}", flush=True)

    res["hotswap_under_1s"] = res["switch_ms_max"] < 1000 and res["load_from_disk_s"] < 1.0
    print("\nRESULT_JSON:" + json.dumps(res))
    print("\n=== VERDICT ===")
    print(f"switch max {res['switch_ms_max']}ms, load-disque {res['load_from_disk_s']}s -> "
          f"hot-swap <1s : {'OUI' if res['hotswap_under_1s'] else 'NON'}")
    return 0 if res["hotswap_under_1s"] else 1


if __name__ == "__main__":
    sys.exit(main())
