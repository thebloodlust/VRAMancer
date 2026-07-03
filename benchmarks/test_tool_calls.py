#!/usr/bin/env python3
"""T6.3 — tests du parser de tool-calls (function calling), sans modèle (pytest-safe)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.tool_calls import parse_tool_calls, build_chat_message


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
               test_malformed_ignored, test_build_chat_message_finish_reason):
        try:
            fn(); print(f"[OK ] {fn.__name__}")
        except AssertionError as e:
            fails += 1; print(f"[FAIL] {fn.__name__}: {e}")
    print("TOUS OK" if not fails else f"{fails} ÉCHECS")
    return fails


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
