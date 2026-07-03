#!/usr/bin/env python3
"""Test health-check ClusterRouter : un worker tué doit être relancé automatiquement."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    import torch
    if torch.cuda.device_count() < 2:
        print("Besoin de 2 GPU."); return 0
    # Capture les alertes M4 (le monitor fait `from core.alerts import notify` à l'exécution)
    import core.alerts as _alerts_mod
    captured_alerts = []
    _alerts_mod.notify = lambda msg, level="info", **k: (captured_alerts.append((level, msg)) or {"ok": True})

    from core.cluster_router import ClusterRouter
    r = ClusterRouter(MODEL, gpu_ids=[0, 1])
    r.start()
    st = r.status()
    print(f"[start] {st}", flush=True)
    assert st["alive"] == 2 and st["restarts"] == 0

    # Tuer brutalement un worker
    victim = r._workers[0]["proc"]
    print(f"[kill] worker GPU0 (pid {victim.pid})", flush=True)
    victim.terminate()

    # Attendre le monitor (tick 3s) + respawn + reload modèle
    ok = False
    for i in range(20):
        time.sleep(2)
        st = r.status()
        if st["alive"] == 2 and st["restarts"] >= 1:
            ok = True; break
    print(f"[after] {st}  (attendu alive=2, restarts>=1)", flush=True)

    # Le cluster sert encore ?
    res = r.submit("Write a function", max_tokens=8)
    print(f"[submit après restart] ok={res.get('ok')} gpu={res.get('gpu_id')}", flush=True)
    r.shutdown()

    alert_ok = any("mort" in m for _, m in captured_alerts)
    print(f"[M4] alerte émise: {captured_alerts[0] if captured_alerts else None}", flush=True)

    verdict = ok and res.get("ok") and alert_ok
    print(f"\nVERDICT (worker relancé + cluster OK + alerte M4): {'OUI' if verdict else 'NON'}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
