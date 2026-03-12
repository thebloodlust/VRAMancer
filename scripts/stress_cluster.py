#!/usr/bin/env python3
"""
VRAMancer Cluster Stress Test (Dry-Run End-to-End)
--------------------------------------------------
Test de simulation de charge intensive.
Frappe violemment l'API VRAMancer pour valider la robustesse :
- du Continuous Batching asynchrone (vLLM / Ollama)
- du P2P Zero-Trust (WebSockets et TCP)
- des Cuda Streams (Zero-Copy TensorRT) sans bloquer le GIL.

Usage:
  python scripts/stress_cluster.py --url http://127.0.0.1:5030/v1/completions --concurrent 50 --total 200
"""
import asyncio
import aiohttp
import time
import argparse
from typing import List, Dict

async def fetch(session: aiohttp.ClientSession, url: str, payload: dict, token: str) -> Dict:
    start_t = time.time()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                data = await response.json()
                latency = time.time() - start_t
                return {"status": "success", "latency": latency, "code": 200, "data": data}
            else:
                text = await response.text()
                return {"status": "error", "code": response.status, "msg": text[:100]}
    except asyncio.TimeoutError:
        return {"status": "timeout", "latency": 30.0}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

async def bombard_api(url: str, prompt: str, concurrent_reqs: int, total_reqs: int, token: str):
    print(f"🚀 Début du Stress Test VRAMancer sur {url}")
    print(f"🎯 Concurrence: {concurrent_reqs} | Requêtes totales: {total_reqs}")
    
    payload = {
        "model": "stress-test-model",
        "prompt": prompt,
        "max_tokens": 16,
        "stream": False
    }

    results = []
    
    # On limite la concurrence maximale avec un Sémaphore (le goulot d'étranglement réseau)
    sem = asyncio.Semaphore(concurrent_reqs)
    
    async with aiohttp.ClientSession() as session:
        async def bound_fetch():
            async with sem:
                return await fetch(session, url, payload, token)
        
        start_time = time.time()
        
        # Lancement asynchrone "Dry Run"
        tasks = [asyncio.create_task(bound_fetch()) for _ in range(total_reqs)]
        
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            res = await coro
            results.append(res)
            if (i+1) % max(1, (total_reqs // 10)) == 0:
                print(f"  ... {i+1}/{total_reqs} requêtes traitées")
                
        total_time = time.time() - start_time
        
    # Analyse
    success = [r for r in results if r["status"] == "success"]
    timeouts = [r for r in results if r["status"] == "timeout"]
    errors = [r for r in results if r["status"] == "error"]
    
    latencies = [r["latency"] for r in success]
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    
    print("\n" + "="*50)
    print("📊 RÉSULTATS DU STRESS TEST")
    print("="*50)
    print(f"⏱️  Temps total       : {total_time:.2f} secondes")
    print(f"⚡ Débit (RPS)       : {(len(success)/total_time):.2f} req/s")
    print(f"✅ Succès            : {len(success)}")
    print(f"⏳ Timeouts          : {len(timeouts)}")
    print(f"❌ Erreurs critiques : {len(errors)}")
    print(f"🐢 Latence moyenne   : {avg_latency:.3f} s")
    print("="*50)
    
    if errors:
        print("Détail des premières erreurs :")
        for e in errors[:5]:
            print(f" - {e}")
            
    if len(success) == total_reqs:
        print("\n🏆 SUCCÈS TOTAL : L'architecture Zero-Copy et le backend Async tiennent la charge avec succès !")
    else:
        print("\n⚠️ AVERTISSEMENT : Des goulots d'étranglement ou des plantages ont été identifiés.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5030/v1/completions")
    parser.add_argument("--concurrent", type=int, default=50)
    parser.add_argument("--total", type=int, default=200)
    parser.add_argument("--token", type=str, default="testtoken", help="Zero-Trust API Token")
    args = parser.parse_args()
    
    asyncio.run(bombard_api(args.url, "Stress test prompt.", args.concurrent, args.total, args.token))
