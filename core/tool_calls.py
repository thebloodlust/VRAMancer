"""T6.3 — parsing des tool-calls (function calling) pour le endpoint OpenAI chat.

Qwen3-Coder / Qwen2.5 émettent les appels d'outils au format **Hermes** :

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Paris"}}
    </tool_call>

On extrait ces blocs de la sortie du modèle et on les convertit au format OpenAI
`tool_calls` (pour un client `openai` standard). Le texte hors blocs reste le `content`.

NB : le format exact doit être confirmé sur la vraie sortie de Qwen3.6 (le parser est
tolérant : JSON par bloc, plusieurs blocs, arguments objet ou déjà-stringifiés).
"""
from __future__ import annotations
import json
import re
import secrets
from typing import Any, Dict, List, Tuple

# Bloc Hermes <tool_call> ... </tool_call> (non-greedy, multi-ligne)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _new_call_id() -> str:
    return "call_" + secrets.token_hex(12)


def parse_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extrait les tool-calls d'une sortie modèle.

    Renvoie (content_sans_blocs, tool_calls_openai). `tool_calls_openai` est vide s'il
    n'y a pas d'appel d'outil.
    """
    if not text or "<tool_call>" not in text:
        return text, []
    tool_calls: List[Dict[str, Any]] = []
    for raw in _TOOL_CALL_RE.findall(text):
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except Exception:
            continue  # bloc malformé -> ignoré (tolérant)
        name = obj.get("name") or obj.get("function")
        if not name:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        # OpenAI attend arguments = STRING JSON
        if not isinstance(args, str):
            try:
                args = json.dumps(args, ensure_ascii=False)
            except Exception:
                args = "{}"
        tool_calls.append({
            "id": _new_call_id(),
            "type": "function",
            "function": {"name": name, "arguments": args},
        })
    # content = texte hors des blocs tool_call
    content = _TOOL_CALL_RE.sub("", text).strip()
    return content, tool_calls


def build_chat_message(text: str) -> Tuple[Dict[str, Any], str]:
    """Construit le message `assistant` OpenAI + le finish_reason à partir d'une sortie.

    Renvoie (message, finish_reason). finish_reason = 'tool_calls' si des appels sont
    présents, sinon 'stop'.
    """
    content, tool_calls = parse_tool_calls(text)
    msg: Dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
        return msg, "tool_calls"
    return msg, "stop"
