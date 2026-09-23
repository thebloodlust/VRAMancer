#!/usr/bin/env python3
"""core/join.py — rejoindre un nœud en une commande (service d'invitation + registre)."""
import json
import os
import socket
import sys
import threading
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.join as join


@pytest.fixture
def nodes_file(tmp_path, monkeypatch):
    f = tmp_path / "nodes.json"
    monkeypatch.setattr(join, "NODES_FILE", f)
    return f


def test_script_pins_coordinator_version_and_bind_address():
    sh = join.render_script("sh", "http://10.0.0.1:5055", "tok123", "b11112", "10.0.0.7")
    assert 'TAG="b11112"' in sh and 'BIND="10.0.0.7"' in sh
    assert "0.0.0.0" not in sh                          # rpc-server sans auth : jamais partout
    assert "http://10.0.0.1:5055/register" in sh
    ps1 = join.render_script("ps1", "http://10.0.0.1:5055", "tok123", "b11112", "10.0.0.7")
    assert "win-vulkan-x64" in ps1 and '$Bind = "10.0.0.7"' in ps1


@pytest.mark.parametrize("field", ["tag", "bind", "token"])
def test_script_refuses_injection(field):
    args = {"tag": "b1", "bind": "10.0.0.7", "token": "t"}
    args[field] = 'x"; rm -rf ~; "'
    with pytest.raises(ValueError):
        join.render_script("sh", "http://h:1", args["token"], args["tag"], args["bind"])


def test_binary_tag_from_path():
    assert join.binary_tag("/c/bin/b11112-linux-vulkan/llama-b11112/llama-server") == "b11112"
    assert join.binary_tag("/usr/local/bin/llama-server") is None


def test_joined_hosts_skip_unreachable_nodes(nodes_file):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    join.register_node("127.0.0.1", port, {"os": "Linux"})
    join.register_node("127.0.0.1", 1, {"os": "éteint"})      # personne n'écoute
    try:
        assert join.joined_rpc_hosts(timeout=0.5) == [f"127.0.0.1:{port}"]
    finally:
        srv.close()


def _serve(nodes_file, joined):
    from http.server import ThreadingHTTPServer
    h = join.make_handler("secret", "b11112", "http://127.0.0.1:5055",
                          on_join=lambda k, i: joined.append(k), check_reachable=False)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), h)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_invite_flow_script_then_register(nodes_file):
    joined = []
    httpd, url = _serve(nodes_file, joined)
    try:
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(f"{url}/join.sh?t=faux")
        assert e.value.code == 403
        sh = urllib.request.urlopen(f"{url}/join.sh?t=secret").read().decode()
        assert 'BIND="127.0.0.1"' in sh                    # l'adresse vue par le nœud

        def post(body):
            req = urllib.request.Request(f"{url}/register", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            return urllib.request.urlopen(req)
        with pytest.raises(urllib.error.HTTPError) as e:
            post({"token": "faux", "port": 50052})
        assert e.value.code == 403
        assert json.loads(post({"token": "secret", "port": 50052, "os": "Linux"}).read())["ok"]
        assert joined == ["127.0.0.1:50052"]
        assert json.loads(nodes_file.read_text())["127.0.0.1:50052"]["os"] == "Linux"
    finally:
        httpd.shutdown()
