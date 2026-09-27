#!/usr/bin/env python3
"""Nœud qui rejoint ou part : serve relance llama-server avec la nouvelle liste RPC."""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.backends_llama_server as bls
import core.llama_server_backend as lsb


class FakeServer:
    started = []

    def __init__(self, path, rpc_hosts=None, **kw):
        if rpc_hosts and "mauvais:1" in rpc_hosts:
            raise RuntimeError("llama-server crashed: rpc injoignable")
        self.rpc = rpc_hosts or []
        self._base_url = "http://127.0.0.1:1"
        self.alive = True
        FakeServer.started.append(list(self.rpc))

    def shutdown(self):
        self.alive = False

    def generate(self, prompt, **kw):
        return f"ok via {self.rpc}"


def _adapter(monkeypatch, nodes):
    FakeServer.started = []
    monkeypatch.setattr(lsb, "LlamaServerBackend", FakeServer)
    monkeypatch.setattr("core.join.joined_rpc_hosts", lambda timeout=1.0: list(nodes))
    monkeypatch.setenv("VRM_AUTO_REPLAN", "0")            # le fil est piloté à la main ici
    monkeypatch.delenv("VRM_RPC_HOSTS", raising=False)
    a = bls.LlamaServerAdapter("m.gguf")
    a.load_model("m.gguf")
    return a


def test_joined_node_triggers_reload_with_new_rpc_list(monkeypatch):
    a = _adapter(monkeypatch, [])
    first = a._server
    assert a.replan(["10.0.0.7:50052"]) is True
    assert first.alive is False                            # VRAM libérée avant de relancer
    assert a.generate("x") == "ok via ['10.0.0.7:50052']"
    assert FakeServer.started == [[], ["10.0.0.7:50052"]]


def test_failed_reload_falls_back_to_previous_nodes(monkeypatch):
    a = _adapter(monkeypatch, ["10.0.0.7:50052"])
    assert a.replan(["10.0.0.7:50052", "mauvais:1"]) is False
    assert a.generate("x") == "ok via ['10.0.0.7:50052']"  # l'ancienne répartition sert
    assert a._rpc == ["10.0.0.7:50052", "mauvais:1"]       # pas de nouvel essai en boucle


def test_requests_wait_during_reload_instead_of_failing(monkeypatch):
    a = _adapter(monkeypatch, [])
    slow_start = a._start

    def slow(rpc):
        time.sleep(0.3)
        slow_start(rpc)
    monkeypatch.setattr(a, "_start", slow)
    t = threading.Thread(target=a.replan, args=(["10.0.0.7:50052"],))
    t.start()
    time.sleep(0.05)
    assert a.generate("x") == "ok via ['10.0.0.7:50052']"  # a attendu la fin du rechargement
    t.join()


def test_watcher_notices_membership_change(monkeypatch):
    nodes = []
    a = _adapter(monkeypatch, nodes)
    seen = threading.Event()
    monkeypatch.setattr(a, "replan", lambda rpc: (seen.set(), a._stop.set()))
    nodes.append("10.0.0.9:50052")
    threading.Thread(target=a._watch_nodes, args=(0.05,), daemon=True).start()
    assert seen.wait(2)
