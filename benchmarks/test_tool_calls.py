#!/usr/bin/env python3
"""T6.3 — tests du parser de tool-calls (function calling), sans modèle (pytest-safe)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.tool_calls import parse_tool_calls, build_chat_message, build_chat_prompt, strip_think


def test_strip_think():
    # bloc fermé
    c, r = strip_think("<think>reasoning here</think>\nThe answer is 42.")
    assert c == "The answer is 42." and "reasoning" in r
    # bloc non fermé (tronqué par max_tokens) -> tout ce qui suit <think> est retiré
    c2, _ = strip_think("Hi.\n<think>I am thinking and got cut off")
    assert c2 == "Hi."
    # think + tool_call : parse ne doit garder que le tool_call
    _, calls = parse_tool_calls('<think>should I?</think><tool_call>{"name":"f","arguments":{}}</tool_call>')
    assert len(calls) == 1 and calls[0]["function"]["name"] == "f"


def test_prompt_tool_role_and_assistant_tool_calls():
    # C0 : boucle agent — assistant tool_call réinjecté + role:tool en <tool_response>
    messages = [
        {"role": "user", "content": "weather in Paris?"},
        {"role": "assistant", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": '{"temp": "22C"}'},
    ]
    p = build_chat_prompt(messages, tools=[{"function": {"name": "get_weather"}}])
    assert "<tool_call>" in p and "get_weather" in p          # tool_call réinjecté
    assert "<tool_response>" in p and "22C" in p              # résultat d'outil injecté
    assert p.rstrip().endswith("Assistant:")


def test_prompt_loop_detection():
    tc = {"function": {"name": "f", "arguments": "{}"}}
    messages = [{"role": "assistant", "tool_calls": [tc]} for _ in range(3)]
    p = build_chat_prompt(messages)
    assert "3 times" in p and "plain text" in p               # instruction anti-boucle
    # < 3 identiques -> pas d'instruction
    p2 = build_chat_prompt(messages[:2])
    assert "3 times" not in p2


def test_no_tool_call():
    content, calls = parse_tool_calls("Just a normal answer.")
    assert content == "Just a normal answer." and calls == []


def test_single_tool_call():
    text = 'Sure.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'
    content, calls = parse_tool_calls(text)
    assert content == "Sure."
    assert len(calls) == 1
    c = calls[0]
    assert c["type"] == "function" and c["function"]["name"] == "get_weather"
    assert json.loads(c["function"]["arguments"]) == {"city": "Paris"}
    assert c["id"].startswith("call_")


def test_multiple_tool_calls():
    text = ('<tool_call>{"name":"a","arguments":{"x":1}}</tool_call>'
            '<tool_call>{"name":"b","arguments":{"y":2}}</tool_call>')
    content, calls = parse_tool_calls(text)
    assert content == "" and len(calls) == 2
    assert [c["function"]["name"] for c in calls] == ["a", "b"]


def test_malformed_ignored():
    text = '<tool_call>not json</tool_call><tool_call>{"name":"ok","arguments":{}}</tool_call>'
    _, calls = parse_tool_calls(text)
    assert len(calls) == 1 and calls[0]["function"]["name"] == "ok"


def test_build_chat_message_finish_reason():
    msg, fr = build_chat_message('<tool_call>{"name":"f","arguments":{}}</tool_call>')
    assert fr == "tool_calls" and msg["tool_calls"][0]["function"]["name"] == "f"
    msg2, fr2 = build_chat_message("hello")
    assert fr2 == "stop" and msg2["content"] == "hello" and "tool_calls" not in msg2


def _run():
    fails = 0
    for fn in (test_no_tool_call, test_single_tool_call, test_multiple_tool_calls,
               test_malformed_ignored, test_build_chat_message_finish_reason,
               test_prompt_tool_role_and_assistant_tool_calls, test_prompt_loop_detection, test_strip_think):
        try:
            fn(); print(f"[OK ] {fn.__name__}")
        except AssertionError as e:
            fails += 1; print(f"[FAIL] {fn.__name__}: {e}")
    print("TOUS OK" if not fails else f"{fails} ÉCHECS")
    return fails


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
