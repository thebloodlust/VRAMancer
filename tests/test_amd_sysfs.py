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
