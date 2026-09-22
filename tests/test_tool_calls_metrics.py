#!/usr/bin/env python3
"""D2.2 — les compteurs /metrics des tool-calls suivent bien le parser.

Réutilise les cas sales de test_tool_calls_regression.py : un JSON réparé doit
incrémenter MALFORMED, un JSON irréparable FAILED, et chaque call émis EMITTED.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.tool_calls import parse_tool_calls

metrics = pytest.importorskip("core.metrics")


def _val(counter):
    """Valeur courante d'un Counter prometheus_client (sans labels)."""
    return counter._value.get()


def test_emitted_counter_increments():
    before = _val(metrics.VRM_TOOL_CALLS_EMITTED)
    parse_tool_calls('<tool_call>{"name":"a","arguments":{}}</tool_call>'
                     '<tool_call>{"name":"b","arguments":{}}</tool_call>')
    assert _val(metrics.VRM_TOOL_CALLS_EMITTED) == before + 2


def test_malformed_counter_increments_when_repaired():
    before_m = _val(metrics.VRM_TOOL_CALLS_MALFORMED)
    before_f = _val(metrics.VRM_TOOL_CALLS_FAILED)
    # virgule finale + quotes simples -> réparé (cf. test_04 de la régression C5)
    _, calls = parse_tool_calls("<tool_call>{'name': 'f', 'arguments': {'x': 1,}}</tool_call>")
    assert len(calls) == 1                                   # réparé, donc émis
    assert _val(metrics.VRM_TOOL_CALLS_MALFORMED) == before_m + 1
    assert _val(metrics.VRM_TOOL_CALLS_FAILED) == before_f    # pas un échec


def test_failed_counter_increments_when_irreparable():
    before_f = _val(metrics.VRM_TOOL_CALLS_FAILED)
    before_e = _val(metrics.VRM_TOOL_CALLS_EMITTED)
    _, calls = parse_tool_calls('Plain text <tool_call>totally not json {{{</tool_call>')
    assert calls == []
    assert _val(metrics.VRM_TOOL_CALLS_FAILED) == before_f + 1
    assert _val(metrics.VRM_TOOL_CALLS_EMITTED) == before_e   # rien d'émis


def test_valid_json_without_name_counts_as_failed():
    before_f = _val(metrics.VRM_TOOL_CALLS_FAILED)
    _, calls = parse_tool_calls('<tool_call>{"arguments":{"x":1}}</tool_call>')
    assert calls == []
    assert _val(metrics.VRM_TOOL_CALLS_FAILED) == before_f + 1


def test_no_tool_call_no_metric_move():
    before = (_val(metrics.VRM_TOOL_CALLS_EMITTED),
              _val(metrics.VRM_TOOL_CALLS_MALFORMED),
              _val(metrics.VRM_TOOL_CALLS_FAILED))
    parse_tool_calls("juste du texte, aucun outil")
    assert (_val(metrics.VRM_TOOL_CALLS_EMITTED),
            _val(metrics.VRM_TOOL_CALLS_MALFORMED),
            _val(metrics.VRM_TOOL_CALLS_FAILED)) == before
