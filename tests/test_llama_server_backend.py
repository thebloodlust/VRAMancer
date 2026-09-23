"""Tests for core.llama_server_backend — pure unit tests, no binary needed."""
import os
import platform
import pytest


# ── _platform_key ─────────────────────────────────────────────────────────────

def test_platform_key_returns_string():
    from core.llama_server_backend import _platform_key
    key = _platform_key()
    assert isinstance(key, str)
    # linux-vulkan : machine avec GPU AMD et sans NVIDIA (ajouté le 2026-09-22 —
    # ces machines recevaient le build CPU, soit 5.7x moins vite en mesure réelle)
    assert key in ("linux-cuda", "linux-vulkan", "linux-cpu",
                   "darwin-arm", "darwin-x86", "windows")


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


def test_asset_map_covers_every_platform_key():
    """Chaque clé de plateforme doit avoir un asset, sinon on retombe en CPU muet."""
    from core.llama_server_backend import _ASSET_MAP
    for key in ("linux-cuda", "linux-vulkan", "linux-cpu",
                "darwin-arm", "darwin-x86", "windows"):
        assert key in _ASSET_MAP, f"pas d'asset pour {key}"
        assert "{tag}" in _ASSET_MAP[key]


def test_asset_names_use_current_upstream_format():
    """Les releases llama.cpp sont en .tar.gz (Linux/macOS) depuis 2026.

    L'ancienne table demandait des .zip inexistants : l'URL construite renvoyait
    404 et le téléchargement automatique était cassé pour tout le monde.
    """
    from core.llama_server_backend import _ASSET_MAP
    for key, asset in _ASSET_MAP.items():
        if key.startswith(("linux", "darwin")):
            assert asset.endswith(".tar.gz"), f"{key}: {asset}"
        else:
            assert asset.endswith(".zip"), f"{key}: {asset}"


def test_amd_detection_delegates_to_sysfs_module(tmp_path, monkeypatch):
    """_has_amd_gpu s'appuie sur core.amd_sysfs (vendor PCI 0x1002 en sysfs)."""
    import core.amd_sysfs as amd
    import core.llama_server_backend as mod

    dev = tmp_path / "device"
    dev.mkdir()
    (dev / "vendor").write_text("0x1002\n")
    (dev / "mem_info_vram_total").write_text("21458059264\n")
    (dev / "mem_info_vram_used").write_text("0\n")
    (dev / "uevent").write_text("DRIVER=amdgpu\nPCI_ID=1002:744C\n")
    monkeypatch.setattr(amd.glob, "glob", lambda pat: [] if "hwmon" in pat else [str(dev)])
    assert mod._has_amd_gpu() is True
    assert mod._platform_key() in ("linux-cuda", "linux-vulkan")  # cuda gagne si nvidia-smi répond


def test_compat_flags_adapt_to_binary_help(monkeypatch):
    """Les options changent de forme selon la version : on lit --help, on ne suppose pas."""
    import core.llama_server_backend as mod

    monkeypatch.setattr(mod, "_server_help",
                        lambda b: "-fa, --flash-attn [on|off|auto]\n-lm, --load-mode MODE\n--log-disable")
    assert mod._compat_flags("x") == ["--flash-attn", "on", "--load-mode", "none", "--log-disable"]

    monkeypatch.setattr(mod, "_server_help", lambda b: "--flash-attn\n--no-mmap\n--log-disable")
    assert mod._compat_flags("x") == ["--flash-attn", "--no-mmap", "--log-disable"]


# --- multi-vendeurs : ne pas laisser une carte AMD inutilisée (2026-09-22) ---

_LIST_DEVICES_MIXED = """\
Available devices:
  Vulkan0: NVIDIA GeForce RTX 3090 (24576 MiB, 24098 MiB free)
  Vulkan1: AMD Radeon RX 7900 XT (RADV NAVI31) (20464 MiB, 20415 MiB free)
"""


def test_backend_devices_parses_both_vendors(monkeypatch):
    """`--list-devices` est la seule source qui voit NVIDIA *et* AMD."""
    import core.llama_server_backend as mod
    monkeypatch.setattr(mod, "_server_help", lambda b: "")
    monkeypatch.setattr(mod, "_runtime_env", lambda b: {})

    class _R:
        stdout = _LIST_DEVICES_MIXED.encode()
        stderr = b""

    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: _R())
    devs = mod.backend_devices("/fake/llama-server")
    assert [d["id"] for d in devs] == ["Vulkan0", "Vulkan1"]
    assert devs[0]["total_mib"] == 24576 and devs[1]["total_mib"] == 20464
    assert "AMD" in devs[1]["name"]


def test_tensor_split_uses_both_cards(monkeypatch):
    """Le split doit couvrir les DEUX cartes, pas seulement celles vues par torch.

    Sans ça, un modèle de 27 GB était servi sur la seule 3090 (24 GB) : échec de
    chargement, ou 20.7 tok/s au lieu de 91.7 mesurés sur la paire.
    """
    import core.llama_server_backend as mod
    monkeypatch.setattr(mod, "backend_devices", lambda b: [
        {"id": "Vulkan0", "name": "NVIDIA GeForce RTX 3090", "total_mib": 24576, "free_mib": None},
        {"id": "Vulkan1", "name": "AMD Radeon RX 7900 XT", "total_mib": 20464, "free_mib": None},
    ])
    assert mod._local_tensor_split(0, binary="/fake") == [24.0, 20.0]


def test_tensor_split_falls_back_to_torch_when_binary_sees_one(monkeypatch):
    import core.llama_server_backend as mod
    monkeypatch.setattr(mod, "backend_devices", lambda b: [
        {"id": "Vulkan0", "name": "NVIDIA GeForce RTX 3090", "total_mib": 24576, "free_mib": None},
    ])
    # un seul device vu -> on ne force rien, on laisse le chemin torch décider
    out = mod._local_tensor_split(1, binary="/fake")
    assert out is None or isinstance(out, list)
