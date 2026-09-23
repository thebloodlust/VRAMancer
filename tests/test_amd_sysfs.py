#!/usr/bin/env python3
"""core/amd_sysfs.py — détection AMD par sysfs (aucune dépendance ROCm/torch).

Les tests utilisent une fausse arborescence sysfs : ils passent donc aussi bien
sur une machine NVIDIA que sur une machine AMD.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.amd_sysfs as mod


def _fake_card(tmp_path, vendor="0x1002", total=21458059264, used=1000000000,
               busy="27", pci_id="1002:744C"):
    dev = tmp_path / "sys" / "class" / "drm" / "card0" / "device"
    dev.mkdir(parents=True)
    (dev / "vendor").write_text(vendor + "\n")
    (dev / "mem_info_vram_total").write_text(f"{total}\n")
    (dev / "mem_info_vram_used").write_text(f"{used}\n")
    (dev / "gpu_busy_percent").write_text(busy + "\n")
    (dev / "uevent").write_text(f"DRIVER=amdgpu\nPCI_ID={pci_id}\n")
    (dev / "power").mkdir()
    (dev / "power" / "runtime_status").write_text("active\n")
    return dev


def test_detects_amd_card(tmp_path, monkeypatch):
    dev = _fake_card(tmp_path)
    monkeypatch.setattr(mod.glob, "glob", lambda pat: [str(dev)])
    gpus = mod.amd_gpus()
    assert len(gpus) == 1
    g = gpus[0]
    assert g["backend"] == "amdgpu"
    assert g["total_bytes"] == 21458059264
    assert g["free_bytes"] == 21458059264 - 1000000000
    assert g["busy_percent"] == 27
    assert g["runtime_status"] == "active"
    assert mod.has_amd_gpu() is True


def test_ignores_non_amd_vendor(tmp_path, monkeypatch):
    dev = _fake_card(tmp_path, vendor="0x10de")     # NVIDIA
    monkeypatch.setattr(mod.glob, "glob", lambda pat: [str(dev)])
    assert mod.amd_gpus() == []
    assert mod.has_amd_gpu() is False


def test_no_card_at_all(monkeypatch):
    monkeypatch.setattr(mod.glob, "glob", lambda pat: [])
    assert mod.amd_gpus() == []
    assert mod.has_amd_gpu() is False


def test_missing_vram_counter_is_skipped(tmp_path, monkeypatch):
    """amdgpu non chargé : le dossier existe mais pas les compteurs → on ignore."""
    dev = tmp_path / "device"
    dev.mkdir()
    (dev / "vendor").write_text("0x1002\n")
    monkeypatch.setattr(mod.glob, "glob", lambda pat: [str(dev)])
    assert mod.amd_gpus() == []


def test_sensors_are_optional(tmp_path, monkeypatch):
    """Pas de hwmon (VM, carte passive) : les clés capteurs sont absentes, pas d'erreur."""
    dev = _fake_card(tmp_path)
    monkeypatch.setattr(mod.glob, "glob",
                        lambda pat: [] if "hwmon" in pat else [str(dev)])
    g = mod.amd_gpus()[0]
    assert "temp_c" not in g and "power_w" not in g


def test_pci_name_returns_none_for_garbage():
    assert mod.pci_device_name("") is None
    assert mod.pci_device_name("pas-un-id") is None


def test_benchmark_cli_uses_existing_profiler_api():
    """`vramancer benchmark` appelait profiler.benchmark_gpu(), qui n'existe pas.

    La commande échouait sur AttributeError pour tout le monde (2026-09-22).
    Ce test verrouille l'API réellement utilisée.
    """
    import inspect
    from core.layer_profiler import LayerProfiler
    import vramancer.main as main_mod

    src = inspect.getsource(main_mod._cmd_benchmark)
    body = src.replace(main_mod._cmd_benchmark.__doc__ or "", "")  # hors docstring
    assert "benchmark_gpu(" not in body, "méthode inexistante réintroduite"
    assert "profile_gpus()" in body
    assert hasattr(LayerProfiler, "profile_gpus")


def test_mixed_vendor_machine_routes_to_subprocess(monkeypatch):
    """Machine NVIDIA + AMD : le binding CUDA in-process ignorerait la carte AMD.

    Mesuré le 2026-09-22 : sur un modèle de 27 GB (qui ne tient pas dans les 24 GB
    de la 3090), le binding échoue à charger, alors que la paire donne 91.7 tok/s.
    select_backend doit donc router vers le sous-processus llama-server.
    """
    import core.backends as backends

    monkeypatch.setattr(backends, "_is_gguf_model", lambda n: True)
    monkeypatch.setattr(backends, "_llamacpp_available", lambda: True)
    monkeypatch.setattr(backends, "_llamacpp_can_offload", lambda: True)   # roue CUDA OK
    monkeypatch.setattr(backends, "_has_amd_gpu", lambda: True)            # mais AMD présente
    monkeypatch.delenv("VRM_FORCE_LLAMACPP_INPROC", raising=False)

    b = backends.select_backend("modele.gguf", backend="auto", num_gpus=1)
    assert type(b).__name__ == "LlamaServerAdapter", type(b).__name__


def test_force_inproc_escape_hatch(monkeypatch):
    """VRM_FORCE_LLAMACPP_INPROC=1 doit rendre la main au binding in-process."""
    import core.backends as backends

    monkeypatch.setattr(backends, "_is_gguf_model", lambda n: True)
    monkeypatch.setattr(backends, "_llamacpp_available", lambda: True)
    monkeypatch.setattr(backends, "_llamacpp_can_offload", lambda: True)
    monkeypatch.setattr(backends, "_has_amd_gpu", lambda: True)
    monkeypatch.setenv("VRM_FORCE_LLAMACPP_INPROC", "1")

    b = backends.select_backend("modele.gguf", backend="auto", num_gpus=1)
    assert type(b).__name__ == "LlamaCppBackend", type(b).__name__
