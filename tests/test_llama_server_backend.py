"""Tests for core.llama_server_backend — pure unit tests, no binary needed."""
import os
import platform
import pytest


# ── _platform_key ─────────────────────────────────────────────────────────────

def test_platform_key_returns_string():
    from core.llama_server_backend import _platform_key
    key = _platform_key()
    assert isinstance(key, str)
    assert key in ("linux-cuda", "linux-cpu", "darwin-arm", "darwin-x86", "windows")


def test_platform_key_darwin_arm(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    from importlib import reload
    import core.llama_server_backend as mod
    reload(mod)
    assert mod._platform_key() == "darwin-arm"


def test_platform_key_darwin_x86(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    from importlib import reload
    import core.llama_server_backend as mod
    reload(mod)
    assert mod._platform_key() == "darwin-x86"


def test_platform_key_windows(monkeypatch):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    from importlib import reload
    import core.llama_server_backend as mod
    reload(mod)
    assert mod._platform_key() == "windows"


# ── _asset_map coverage ───────────────────────────────────────────────────────

def test_asset_map_all_keys_have_tag_placeholder():
    from core.llama_server_backend import _ASSET_MAP
    for key, template in _ASSET_MAP.items():
        assert "{tag}" in template, f"Missing {{tag}} in _ASSET_MAP[{key!r}]"


def test_asset_map_linux_cpu_exists():
    from core.llama_server_backend import _ASSET_MAP
    assert "linux-cpu" in _ASSET_MAP


# ── Constants ─────────────────────────────────────────────────────────────────

def test_server_port_default():
    from core.llama_server_backend import SERVER_PORT
    # Default is 8081 when VRM_LLAMA_SERVER_PORT is not set
    expected = int(os.environ.get("VRM_LLAMA_SERVER_PORT", "8081"))
    assert SERVER_PORT == expected


def test_binary_dir_under_home():
    from core.llama_server_backend import BINARY_DIR
    from pathlib import Path
    home = Path.home()
    assert str(BINARY_DIR).startswith(str(home))


# ── LlamaServerBackend constructor guards ─────────────────────────────────────

def test_init_raises_without_binary(tmp_path):
    """Constructor must raise when binary not found (no auto-download in tests)."""
    from core.llama_server_backend import LlamaServerBackend
    fake_model = str(tmp_path / "model.gguf")
    # Write a dummy gguf file so the model path exists
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    with pytest.raises(Exception):
        LlamaServerBackend(
            model_path=fake_model,
            binary_path=str(tmp_path / "nonexistent_binary"),
        )
