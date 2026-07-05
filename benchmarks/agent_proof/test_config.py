"""Tests de config — l'agent doit les mettre à jour (C1)."""
from benchmarks.agent_proof.config import parse_config


def test_parse_config_basic():
    cfg = parse_config({"host": "localhost", "port": "8080"})
    assert cfg == {"host": "localhost", "port": 8080, "workers": 1}
