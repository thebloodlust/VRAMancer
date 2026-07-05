"""T6.3 — parsing des tool-calls (function calling) pour le endpoint OpenAI chat.

Qwen3-Coder / Qwen2.5 émettent les appels d'outils au format **Hermes** :

    <tool_call>
    {"name": "get_weather", "arguments": {"city": "Paris"}}
    </tool_call>

On extrait ces blocs de la sortie du modèle et on les convertit au format OpenAI
`tool_calls` (pour un client `openai` standard). Le texte hors blocs reste le `content`.

VALIDÉ sur Qwen3.6-35B-A3B (GGUF, llama.cpp) le 2026-07-03 : requête `tools` -> le modèle
émet un `<tool_call>` -> ce parser produit un `tool_calls` OpenAI correct (finish_reason
`tool_calls`, arguments JSON-string). Le parser reste tolérant (multi-blocs, malformés ignorés).
"""
from __future__ import annotations
import json
import re
import secrets
from typing import Any, Dict, List, Tuple

# Bloc Hermes <tool_call> ... </tool_call> (non-greedy, multi-ligne)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Bloc de raisonnement Qwen (<think>...</think>) — NE doit PAS fuir dans content.
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Bloc <think> non fermé (coupé par max_tokens) : on strippe jusqu'à la fin.
_THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)


def strip_think(text: str) -> Tuple[str, str]:
    """Sépare le raisonnement <think> du reste. Renvoie (texte_sans_think, reasoning)."""
    if not text or "<think>" not in text:
        return text, ""
    reasoning = "\n".join(m.strip() for m in _THINK_RE.findall(text))
    cleaned = _THINK_RE.sub("", text)
    cleaned = _THINK_OPEN_RE.sub("", cleaned)  # <think> non fermé (tronqué)
    return cleaned.strip(), reasoning.strip()


def _new_call_id() -> str:
    return "call_" + secrets.token_hex(12)


def parse_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extrait les tool-calls d'une sortie modèle.

    Renvoie (content_sans_blocs, tool_calls_openai). `tool_calls_openai` est vide s'il
    n'y a pas d'appel d'outil.
    """
    text, _ = strip_think(text)  # le raisonnement ne doit pas polluer le parsing/content
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


_TOOL_PREAMBLE = (
    "/no_think\n"
    "You can call tools. When a tool is needed, respond with ONLY the tool call block "
    "and nothing else — no explanation, no reasoning, no <think>. Emit exactly:\n"
    '<tool_call>\n{"name": <tool_name>, "arguments": <args_object>}\n</tool_call>\n'
    "When you have the tool result, answer the user in plain text.\n"
    "Available tools (JSON schema): {specs}"
)


def build_chat_prompt(messages: List[Dict[str, Any]], tools: Any = None) -> str:
    """Messages OpenAI -> prompt texte, pour la BOUCLE agent complète (C0).

    Gère : system, user, assistant (avec `tool_calls` réinjectés en <tool_call>),
    `role:"tool"` (résultat d'outil en <tool_response>). Injecte les `tools` (préambule
    Hermes) et détecte les boucles (3 tool_calls identiques -> instruction de répondre
    en texte). Renvoie le prompt (se termine par 'Assistant:').
    """
    tools = tools or []
    parts: List[str] = []
    if tools:
        try:
            specs = json.dumps([t.get("function", t) for t in tools], ensure_ascii=False)
            parts.append("System: " + _TOOL_PREAMBLE.format(specs=specs))
        except Exception:
            pass
    # Détection de boucle
    recent = [(tc.get("function", {}).get("name"), tc.get("function", {}).get("arguments"))
              for m in messages if m.get("role") == "assistant"
              for tc in (m.get("tool_calls") or [])]
    loop_break = ""
    if len(recent) >= 3 and len(set(recent[-3:])) == 1:
        loop_break = ("System: The same tool was called 3 times with no progress. "
                      "Answer in plain text now; do NOT call the tool again.")
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            for tc in (msg.get("tool_calls") or []):
                fn = tc.get("function", {})
                a = fn.get("arguments", "{}")
                try:
                    a = json.loads(a) if isinstance(a, str) else a
                except Exception:
                    a = {}
                blk = json.dumps({"name": fn.get("name"), "arguments": a}, ensure_ascii=False)
                parts.append(f"Assistant: <tool_call>\n{blk}\n</tool_call>")
            if content:
                parts.append(f"Assistant: {content}")
        elif role == "tool":
            parts.append(f"User: <tool_response>\n{content}\n</tool_response>")
        else:
            parts.append(f"User: {content}")
    if loop_break:
        parts.append(loop_break)
    parts.append("Assistant:")
    return "\n".join(parts)


def build_chat_message(text: str) -> Tuple[Dict[str, Any], str]:
    """Construit le message `assistant` OpenAI + le finish_reason à partir d'une sortie.

    Renvoie (message, finish_reason). finish_reason = 'tool_calls' si des appels sont
    présents, sinon 'stop'.
    """
    _clean, reasoning = strip_think(text)
    content, tool_calls = parse_tool_calls(_clean)
    msg: Dict[str, Any] = {"role": "assistant", "content": content or None}
    if reasoning:
        msg["reasoning_content"] = reasoning  # transparence, sans polluer content
    if tool_calls:
        msg["tool_calls"] = tool_calls
        return msg, "tool_calls"
    return msg, "stop"
