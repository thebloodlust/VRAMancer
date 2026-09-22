#!/usr/bin/env python3
"""Tool calling sur VRAMancer en 30 lignes — porte d'entrée dev (D2.1).

Utilise le client `openai` OFFICIEL pointé sur le serveur local : rien de spécifique
à VRAMancer côté client, c'est tout l'intérêt (API OpenAI-compatible).

Prérequis — lancer le serveur dans un autre terminal :

    vramancer serve --model Qwen/Qwen3.6-35B-A3B --profile coding --port 5030

Puis :

    python examples/tool_calling_quickstart.py
"""
from __future__ import annotations

import json
import sys

BASE_URL = "http://localhost:5030/v1"
MODEL = "local"  # le serveur sert le modèle chargé, quel que soit le nom demandé

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Renvoie la météo actuelle d'une ville.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "Nom de la ville"}},
            "required": ["city"],
        },
    },
}]


def get_weather(city: str) -> dict:
    """Faux outil : en vrai tu appellerais une API météo ici."""
    return {"city": city, "temp_c": 18, "conditions": "nuageux"}


def main() -> int:
    try:
        from openai import OpenAI
    except ImportError:
        print("Il manque le client openai :  pip install openai", file=sys.stderr)
        return 1

    client = OpenAI(base_url=BASE_URL, api_key="not-needed")
    messages = [{"role": "user", "content": "Quel temps fait-il à Toulouse ?"}]

    try:
        # Tour 1 — le modèle décide d'appeler l'outil
        r1 = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    except Exception as e:  # connexion refusée, timeout, 5xx…
        print(f"Impossible de joindre {BASE_URL} ({type(e).__name__}).", file=sys.stderr)
        print("Lance d'abord :  vramancer serve --model <modele> --profile coding --port 5030",
              file=sys.stderr)
        return 1

    msg = r1.choices[0].message
    if not msg.tool_calls:
        print("Le modèle n'a pas appelé d'outil ; il a répondu directement :")
        print(msg.content)
        return 0

    call = msg.tool_calls[0]
    args = json.loads(call.function.arguments or "{}")
    print(f"→ tool_call : {call.function.name}({args})")
    result = get_weather(**args)
    print(f"→ résultat  : {result}")

    # Tour 2 — on renvoie le résultat, le modèle rédige la réponse finale
    messages += [
        msg.model_dump(exclude_none=True),
        {"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)},
    ]
    r2 = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS)
    print("\nRéponse finale :")
    print(r2.choices[0].message.content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
