"""Tests for core.config_manager."""
import os
import pytest
from core.config_manager import get_config, reload_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(os.environ.keys()):
        if k.startswith("VRM_") and k != "VRM_MINIMAL_TEST":
            monkeypatch.delenv(k, raising=False)
    yield


def test_defaults():
    reload_config()
    cfg = get_config()
    assert cfg.backend == "auto"
    assert cfg.trust_remote_code is False
    assert cfg.parallel_mode == "pp"


def test_env_override(monkeypatch):
    monkeypatch.setenv("VRM_TRUST_REMOTE_CODE", "1")
    monkeypatch.setenv("VRM_BACKEND", "vllm")
    reload_config()
    cfg = get_config()
    assert cfg.trust_remote_code is True
    assert cfg.backend == "vllm"


def test_yaml_override(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("backend: ollama\nquantization: nf4\n")
    monkeypatch.chdir(tmp_path)
    reload_config()
    cfg = get_config()
    assert cfg.backend == "ollama"
    assert cfg.quantization == "nf4"


def test_env_wins_over_yaml(tmp_path, monkeypatch):
    (tmp_path / "config.yaml").write_text("backend: ollama\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("VRM_BACKEND", "vllm")
    reload_config()
    assert get_config().backend == "vllm"


def test_singleton_caching():
    reload_config()
    assert get_config() is get_config()
