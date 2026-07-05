#!/usr/bin/env python3
"""C5 — suite de régression du function calling : 10 scénarios sales (sans modèle).

Garantit qu'aucun tool_call syntaxiquement invalide ne sort du serveur, et couvre les
cas que les agents réels envoient. Collectable par pytest.
"""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.tool_calls import parse_tool_calls, build_chat_prompt, strip_think


def test_01_nominal():
    _, calls = parse_tool_calls('<tool_call>{"name":"f","arguments":{"x":1}}</tool_call>')
    assert len(calls) == 1 and calls[0]["function"]["name"] == "f"


def test_02_multi():
    _, calls = parse_tool_calls('<tool_call>{"name":"a","arguments":{}}</tool_call>'
                                '<tool_call>{"name":"b","arguments":{}}</tool_call>')
    assert [c["function"]["name"] for c in calls] == ["a", "b"]


def test_03_malformed_irreparable_no_call():
    content, calls = parse_tool_calls('Plain text <tool_call>totally not json {{{</tool_call>')
    assert calls == []                          # jamais de tool_call cassé
    assert "Plain text" in content              # le texte reste en content


def test_04_malformed_repairable():
    # virgule finale + quotes simples -> réparé
    _, calls = parse_tool_calls("<tool_call>{'name': 'f', 'arguments': {'x': 1,}}</tool_call>")
    assert len(calls) == 1 and json.loads(calls[0]["function"]["arguments"]) == {"x": 1}


def test_05_tool_choice_none_ignores_tools():
    p = build_chat_prompt([{"role": "user", "content": "hi"}],
                          tools=[{"function": {"name": "f"}}], tool_choice="none")
    assert "<tools>" not in p                    # outils non injectés


def test_06_tool_choice_required():
    p = build_chat_prompt([{"role": "user", "content": "hi"}],
                          tools=[{"function": {"name": "f"}}], tool_choice="required")
    assert "MUST call a tool" in p


def test_07_tool_error_result_in_prompt():
    p = build_chat_prompt([
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "tool_calls": [{"function": {"name": "get_weather", "arguments": "{}"}}]},
        {"role": "tool", "content": '{"error": "city not found"}'},
    ], tools=[{"function": {"name": "get_weather"}}])
    assert "<tool_response>" in p and "city not found" in p


def test_08_loop_detected():
    tc = {"function": {"name": "f", "arguments": "{}"}}
    p = build_chat_prompt([{"role": "assistant", "tool_calls": [tc]} for _ in range(3)])
    assert "3 times" in p


def test_09_unicode_and_empty_args():
    _, calls = parse_tool_calls('<tool_call>{"name":"f","arguments":{"note":"café ☕","empty":{}}}</tool_call>')
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["note"] == "café ☕" and args["empty"] == {}


def test_10_think_stripped_before_toolcall():
    content, calls = parse_tool_calls(
        '<think>Let me reason a lot...</think><tool_call>{"name":"f","arguments":{}}</tool_call>')
    assert len(calls) == 1 and "reason" not in (content or "")


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    fails = 0
    for fn in fns:
        try:
            fn(); print(f"[OK ] {fn.__name__}")
        except AssertionError as e:
            fails += 1; print(f"[FAIL] {fn.__name__}: {e}")
    print(f"{len(fns) - fails}/{len(fns)} scénarios OK")
    return fails


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
