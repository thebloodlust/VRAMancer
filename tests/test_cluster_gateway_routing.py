#!/usr/bin/env python3
"""Routage de la passerelle cluster (core/cluster_gateway.NodePool).

Défauts mesurés le 2026-09-23 sur 3090 + 7900 XT, corrigés ici :
  1. un nœud lent recevait autant de travail qu'un rapide → débit −31 % ;
  2. un nœud mort aspirait le trafic (« trou noir ») → 12 requêtes sur 16 perdues.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.cluster_gateway import NodePool

FAST, SLOW = "http://fast:1", "http://slow:1"


def _warm(pool, fast_tok_s=38.0, slow_tok_s=16.0):
    """Une réponse mesurée par nœud (vitesses observées en test réel à 4 en parallèle)."""
    for url, speed in ((FAST, fast_tok_s), (SLOW, slow_tok_s)):
        n = next(x for x in pool.nodes if x["url"] == url)
        n["inflight"] += 1
        pool.done(n, True, tokens=int(speed * 5), seconds=5.0)


def test_unmeasured_nodes_are_each_tried_once():
    pool = NodePool([FAST, SLOW])
    a, b = pool.pick(), pool.pick()
    assert {a["url"], b["url"]} == {FAST, SLOW}


def test_fast_node_gets_more_concurrent_work():
    pool = NodePool([FAST, SLOW])
    _warm(pool)
    picks = [pool.pick()["url"] for _ in range(4)]      # 4 requêtes en vol
    assert picks.count(FAST) >= 3, picks                  # l'ancien routage : 2/2


def test_slow_node_still_used_when_fast_is_saturated():
    pool = NodePool([FAST, SLOW])
    _warm(pool)
    picks = [pool.pick()["url"] for _ in range(12)]
    assert SLOW in picks                                   # la capacité en plus sert


def test_failed_node_is_removed_immediately():
    """Trou noir : un nœud qui échoue ne doit plus être choisi avant le health-check."""
    pool = NodePool([FAST, SLOW])
    _warm(pool)
    dead = next(n for n in pool.nodes if n["url"] == SLOW)
    dead["inflight"] += 1
    pool.done(dead, False)
    assert dead["ok"] is False
    assert all(pool.pick()["url"] == FAST for _ in range(6))


def test_retry_excludes_the_node_that_just_failed():
    pool = NodePool([FAST, SLOW])
    _warm(pool)
    assert pool.pick(exclude=[FAST])["url"] == SLOW


def test_errors_do_not_count_as_speed():
    pool = NodePool([FAST])
    n = pool.nodes[0]
    n["inflight"] += 1
    pool.done(n, False, tokens=999, seconds=0.001)
    assert n["tok_s"] is None
