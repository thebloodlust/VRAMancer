#!/usr/bin/env python3
"""C4 — mesure TTFT + tok/s décode selon la taille de contexte (agent = gros prompts).

Envoie des prompts de code synthétiques de taille croissante et mesure : TTFT (proxy :
temps pour 1 token) et débit décode (tokens/s sur 64 tokens). Publie le tableau pour
BENCHMARK_RESULTS.md. Les tailles qui dépassent le contexte du serveur sont sautées
(erreur 400/413) — augmente le contexte du serveur pour les mesurer.

Usage: python benchmarks/bench_context_scaling.py [url] [tailles_tok]
  ex:  python benchmarks/bench_context_scaling.py http://localhost:5030 1000,2000,4000,8000,16000,32000
"""
import sys, json, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5030"
SIZES = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2
                          else ["1000", "2000", "4000", "8000", "16000", "32000"])]
# ~1 token ≈ 4 chars. Bout de code répété pour approcher N tokens.
_SNIPPET = "def process(item):\n    result = transform(item.value) + offset  # compute\n    return result\n"


def _prompt_of_tokens(n_tok: int) -> str:
    target_chars = n_tok * 4
    reps = max(1, target_chars // len(_SNIPPET))
    return ("# Codebase excerpt\n" + _SNIPPET * reps +
            "\n# Question: summarize what process() does in one sentence.")


def chat(prompt, max_tokens):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.loads(r.read().decode())
        dt = time.perf_counter() - t0
        if "choices" not in d:
            return None, dt, d
        return dt, d["choices"][0], None
    except urllib.error.HTTPError as e:
        return None, None, e.read().decode()[:120]
    except Exception as e:
        return None, None, str(e)[:120]


def main():
    print(f"[C4] contexte scaling contre {URL}\n")
    print(f"{'prompt~tok':>10} {'TTFT(s)':>9} {'décode tok/s':>13} {'note':>8}")
    rows = []
    for n in SIZES:
        prompt = _prompt_of_tokens(n)
        # TTFT proxy : 1 token
        ttft, c1, err1 = chat(prompt, 1)
        if ttft is None:
            print(f"{n:>10} {'—':>9} {'—':>13}   skip ({err1})")
            continue
        # débit décode : 64 tokens
        dt64, c64, err64 = chat(prompt, 64)
        if dt64 is None:
            print(f"{n:>10} {ttft:>9.2f} {'—':>13}   {err64}")
            continue
        gen = c64.get("usage", {}) if isinstance(c64, dict) else {}
        # décode ≈ (t64 - ttft) pour ~63 tokens de plus
        dec_tok_s = 63 / max(1e-3, (dt64 - ttft))
        print(f"{n:>10} {ttft:>9.2f} {dec_tok_s:>13.1f}   ok")
        rows.append({"prompt_tok": n, "ttft_s": round(ttft, 2), "decode_tok_s": round(dec_tok_s, 1)})
    print("\nRESULT_JSON:" + json.dumps(rows))
    print("\n-> Copier le tableau dans BENCHMARK_RESULTS.md. Les 'skip' = contexte serveur trop petit.")


if __name__ == "__main__":
    main()
