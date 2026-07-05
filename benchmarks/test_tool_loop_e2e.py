#!/usr/bin/env python3
"""C0 — test e2e de la BOUCLE agent complète (tool_call -> tool_result -> réponse finale).

Nécessite le serveur Qwen3.6 lancé (./serve_qwen36.sh) AVEC le code C0 (redémarrer si besoin).
Vérifie : le modèle émet un tool_call ; on renvoie le résultat (role:tool) ; le modèle
produit une réponse texte finale utilisant le résultat ; + un cas d'erreur (city not found).

Usage: python benchmarks/test_tool_loop_e2e.py [url]
"""
import sys, json, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5030"
WEATHER_TOOL = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]


def chat(messages, tools=None, max_tokens=200):
    body = {"messages": messages, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())["choices"][0]


def _round_trip(tool_result_content: str):
    """1 aller-retour : demande météo -> tool_call -> renvoie result -> réponse finale."""
    m1 = [{"role": "user", "content": "What is the weather in Paris? Use the tool."}]
    c1 = chat(m1, WEATHER_TOOL)
    tcs = c1["message"].get("tool_calls")
    assert tcs and tcs[0]["function"]["name"] == "get_weather", f"pas de tool_call: {c1}"
    tc = tcs[0]
    # Tour 2 : on renvoie le résultat de l'outil
    m2 = m1 + [
        {"role": "assistant", "tool_calls": tcs},
        {"role": "tool", "tool_call_id": tc["id"], "content": tool_result_content},
    ]
    c2 = chat(m2, WEATHER_TOOL)
    return c1, c2


def main():
    fails = 0
    # Cas nominal : résultat 22C -> réponse finale texte mentionnant 22
    try:
        c1, c2 = _round_trip('{"temp_c": 22, "condition": "sunny"}')
        final = (c2["message"].get("content") or "")
        assert c2["finish_reason"] != "tool_calls" or not c2["message"].get("tool_calls"), \
            f"a re-appelé l'outil au lieu de répondre: {c2}"
        assert "22" in final, f"réponse finale ne mentionne pas 22C: {final!r}"
        print(f"[OK ] nominal : tour1 tool_call ✓, tour2 réponse finale = {final[:80]!r}")
    except Exception as e:
        fails += 1; print(f"[FAIL] nominal : {e}")

    # Cas erreur : city not found -> le modèle gère gracieusement (pas de crash, réponse texte)
    try:
        c1, c2 = _round_trip('{"error": "city not found"}')
        final = (c2["message"].get("content") or "")
        assert final.strip(), f"pas de réponse texte sur erreur: {c2}"
        print(f"[OK ] erreur  : réponse gérée = {final[:80]!r}")
    except Exception as e:
        fails += 1; print(f"[FAIL] erreur : {e}")

    print("TOUS OK" if not fails else f"{fails} ÉCHECS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
