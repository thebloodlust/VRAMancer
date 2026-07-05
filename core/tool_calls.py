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


def _loads_repair(raw: str):
    """Parse un JSON, avec réparation triviale (virgule finale, quotes simples). Renvoie
    l'objet ou None si irréparable (C5 : on n'émet jamais un tool_call cassé)."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    repaired = re.sub(r",\s*([}\]])", r"\1", raw)          # virgule finale
    repaired = re.sub(r"'([^']*)'\s*:", r'"\1":', repaired)  # 'clé': -> "clé":
    repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)  # : 'val' -> : "val"
    try:
        return json.loads(repaired)
    except Exception:
        return None


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
        obj = _loads_repair(raw)
        if not isinstance(obj, dict):  # irréparable -> jamais de tool_call cassé (C5)
            try:
                import logging
                logging.getLogger("vramancer").warning(
                    "tool_call JSON irréparable, ignoré: %s", raw[:120])
            except Exception:
                pass
            continue
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


# Bloc outils officiel Qwen (ChatML), inséré dans le message system.
_QWEN_TOOLS_BLOCK = """

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

# Assistant pré-rempli avec un bloc <think> VIDE : désactive le raisonnement Qwen3
# (le modèle génère directement après </think>). Validé sur Qwen3.6 (tool_call en 20 tok).
_ASSISTANT_NOTHINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _tc_block(msg: Dict[str, Any]) -> List[str]:
    segs = []
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function", {})
        a = fn.get("arguments", "{}")
        try:
            a = json.loads(a) if isinstance(a, str) else a
        except Exception:
            a = {}
        segs.append("<tool_call>\n" + json.dumps({"name": fn.get("name"), "arguments": a},
                                                  ensure_ascii=False) + "\n</tool_call>")
    return segs


def build_chat_prompt(messages: List[Dict[str, Any]], tools: Any = None,
                      tool_choice: Any = "auto") -> str:
    """Messages OpenAI -> prompt **Qwen ChatML** pour la boucle agent complète (C0).

    Format natif du modèle (`<|im_start|>role...<|im_end|>`) : c'est ce qui rend le
    tool-calling fiable (un prompt générique déclenche le thinking et casse tout).
    - `tools` -> bloc `<tools>` officiel dans le message system.
    - assistant `tool_calls` -> `<tool_call>` ; `role:"tool"` -> `<tool_response>`.
    - fin : assistant + `<think></think>` vide (désactive le raisonnement).
    - détection de boucle (3 tool_calls identiques -> forcer une réponse texte).
    """
    tools = tools or []
    if tool_choice == "none":  # C5 : le client demande d'ignorer les outils
        tools = []
    system = "\n".join((m.get("content") or "") for m in messages if m.get("role") == "system").strip()
    if tools:
        try:
            tools_lines = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
            system = (system + _QWEN_TOOLS_BLOCK.format(tools=tools_lines)).strip()
            if tool_choice == "required" or (isinstance(tool_choice, dict)):
                system += "\nYou MUST call a tool. Respond with a <tool_call> block."
        except Exception:
            pass

    recent = [(tc.get("function", {}).get("name"), tc.get("function", {}).get("arguments"))
              for m in messages if m.get("role") == "assistant"
              for tc in (m.get("tool_calls") or [])]
    loop_break = ("The same tool was called 3 times with no progress. Answer in plain "
                  "text now; do NOT call the tool again.") if (
                      len(recent) >= 3 and len(set(recent[-3:])) == 1) else ""

    parts: List[str] = []
    if system:
        parts.append(f"<|im_start|>system\n{system}<|im_end|>")
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        if role == "system":
            continue
        elif role == "assistant":
            segs = _tc_block(msg)
            if content:
                segs.append(content)
            parts.append(f"<|im_start|>assistant\n{chr(10).join(segs)}<|im_end|>")
        elif role == "tool":
            parts.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response><|im_end|>")
        else:
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
    if loop_break:
        parts.append(f"<|im_start|>user\n{loop_break}<|im_end|>")
    parts.append(_ASSISTANT_NOTHINK)
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
