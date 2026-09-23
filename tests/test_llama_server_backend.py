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


def test_adapter_exposes_real_tokenizer(monkeypatch):
    """Le chemin llama-server doit fournir un tokenizer, sinon `usage` est faux.

    Sans tokenizer, l'API compte des MOTS (`len(text.split())`) : 2 851 annoncés
    pour 8 454 tokens réels sur du code (mesuré le 2026-09-23). Le proxy interroge
    l'endpoint /tokenize de llama-server.
    """
    import core.backends_llama_server as mod

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"tokens": list(range(42))}

    calls = []

    def _post(url, json=None, timeout=None):
        calls.append((url, json))
        return _Resp()

    import requests
    monkeypatch.setattr(requests, "post", _post)
    tok = mod._ServerTokenizer("http://127.0.0.1:8081")
    assert len(tok.encode("def f(x): return x + 1")) == 42
    assert calls[0][0] == "http://127.0.0.1:8081/tokenize"
    assert calls[0][1] == {"content": "def f(x): return x + 1"}


def test_count_tokens_uses_adapter_tokenizer():
    """count_tokens() doit préférer le tokenizer fourni au repli par mots."""
    from core.api.validation import count_tokens

    class _Tok:
        def encode(self, text):
            return [0] * 99

    assert count_tokens("trois mots ici", _Tok()) == 99
    assert count_tokens("trois mots ici", None) == 3     # repli documenté


def test_rpc_devices_are_ordered_first(monkeypatch):
    """`--tensor-split` suit l'ordre interne de llama.cpp : RPC EN TÊTE.

    `--list-devices` affiche les GPU locaux d'abord ; sans réordonner, le split
    « 83 % sur la 3090 locale » partait sur la carte distante (mesuré : 3090 à
    6.6 GB, carte distante qui déborde, 12 tok/s au lieu de 98).
    """
    import core.llama_server_backend as mod
    calls = []

    class _R:
        stdout = (b"Available devices:\n"
                  b"  Vulkan0: NVIDIA GeForce RTX 3090 (24576 MiB, 24098 MiB free)\n"
                  b"  RPC0: 127.0.0.1:50052 (20464 MiB, 20415 MiB free)\n")
        stderr = b""

    def _run(cmd, **k):
        calls.append(cmd)
        return _R()

    monkeypatch.setattr(mod, "_runtime_env", lambda b: {})
    monkeypatch.setattr(mod.subprocess, "run", _run)
    devs = mod.backend_devices("/fake/llama-server", rpc_hosts=["127.0.0.1:50052"])
    assert [d["id"] for d in devs] == ["RPC0", "Vulkan0"]
    assert devs[0]["rpc"] is True and devs[1]["rpc"] is False
    assert "--rpc" in calls[0] and "127.0.0.1:50052" in calls[0]


def test_no_rpc_flag_without_rpc_hosts(monkeypatch):
    import core.llama_server_backend as mod
    calls = []

    class _R:
        stdout = b"  Vulkan0: NVIDIA GeForce RTX 3090 (24576 MiB, 24098 MiB free)\n"
        stderr = b""

    monkeypatch.setattr(mod, "_runtime_env", lambda b: {})
    monkeypatch.setattr(mod.subprocess, "run", lambda cmd, **k: calls.append(cmd) or _R())
    mod.backend_devices("/fake/llama-server")
    assert "--rpc" not in calls[0]


def test_spec_ngram_enabled_when_requested(monkeypatch):
    """VRM_SPEC=ngram -> --spec-type ngram-simple (7.8x mesuré sur une édition d'agent)."""
    import core.llama_server_backend as mod
    help_txt = "--spec-type none,draft-simple,ngram-simple,ngram-map-k\n--spec-draft-n-max N"
    monkeypatch.setenv("VRM_SPEC", "ngram")
    assert mod._spec_flags(help_txt) == ["--spec-type", "ngram-simple"]
    monkeypatch.setenv("VRM_SPEC_N_MAX", "16")
    assert mod._spec_flags(help_txt) == ["--spec-type", "ngram-simple", "--spec-draft-n-max", "16"]


def test_spec_off_by_default_and_on_old_binaries(monkeypatch):
    import core.llama_server_backend as mod
    monkeypatch.delenv("VRM_SPEC", raising=False)
    assert mod._spec_flags("--spec-type none,ngram-simple") == []
    monkeypatch.setenv("VRM_SPEC", "ngram")
    assert mod._spec_flags("--flash-attn [on|off|auto]") == []      # binaire trop ancien


def test_spec_unknown_type_is_ignored(monkeypatch):
    import core.llama_server_backend as mod
    monkeypatch.setenv("VRM_SPEC", "quantum-magic")
    assert mod._spec_flags("--spec-type none,ngram-simple") == []


def _write_gguf(path, kvs):
    """GGUF v3 minimal : en-tête + métadonnées (aucun tenseur)."""
    import struct as st
    with open(path, "wb") as f:
        f.write(b"GGUF" + st.pack("<IQQ", 3, 0, len(kvs)))
        for key, (typ, val) in kvs.items():
            k = key.encode()
            f.write(st.pack("<Q", len(k)) + k + st.pack("<I", typ))
            if typ == 8:
                v = val.encode()
                f.write(st.pack("<Q", len(v)) + v)
            elif typ == 9:                               # tableau de chaînes (vocabulaire)
                f.write(st.pack("<IQ", 8, len(val)))
                for item in val:
                    b = item.encode()
                    f.write(st.pack("<Q", len(b)) + b)
            else:
                f.write(st.pack("<I", val))


def test_gguf_expert_count_reads_moe_and_dense(tmp_path):
    import core.llama_server_backend as mod
    moe, dense = tmp_path / "moe.gguf", tmp_path / "dense.gguf"
    vocab = ["<a>", "<b>", "tok"] * 50
    _write_gguf(moe, {"general.architecture": (8, "qwen35moe"),
                      "tokenizer.ggml.tokens": (9, vocab),
                      "qwen35moe.expert_count": (4, 256)})
    _write_gguf(dense, {"general.architecture": (8, "qwen2"),
                        "tokenizer.ggml.tokens": (9, vocab)})
    assert mod.gguf_expert_count(moe) == 256
    assert mod.gguf_expert_count(dense) == 0
    (tmp_path / "x.txt").write_text("pas un gguf")
    assert mod.gguf_expert_count(tmp_path / "x.txt") is None


def test_spec_auto_enables_on_dense_disables_on_moe(tmp_path, monkeypatch):
    """Mesuré : dense ×7.5 avec n-grammes, MoE −65 %. « auto » ne doit jamais ralentir."""
    import core.llama_server_backend as mod
    help_txt = "--spec-type none,draft-simple,ngram-simple"
    moe, dense = tmp_path / "moe.gguf", tmp_path / "dense.gguf"
    _write_gguf(moe, {"qwen35moe.expert_count": (4, 256)})
    _write_gguf(dense, {"general.architecture": (8, "qwen2")})
    monkeypatch.setenv("VRM_SPEC", "auto")
    assert mod._spec_flags(help_txt, str(dense)) == ["--spec-type", "ngram-simple"]
    assert mod._spec_flags(help_txt, str(moe)) == []
    assert mod._spec_flags(help_txt, None) == []            # inconnu -> jamais par défaut


# ── en-tête GGUF maison, modèles PrismML ────────────────────────────────────

def _write_gguf_tensors(path, tensors, kv=None, align=32):
    """GGUF v3 minimal : métadonnées scalaires/chaînes + tenseurs (nom, type, octets)."""
    import struct
    kv = kv or {"general.architecture": "llama", "llama.block_count": 2}
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv)))

    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    for k, v in kv.items():
        out += s(k) + (struct.pack("<I", 8) + s(v) if isinstance(v, str)
                       else struct.pack("<II", 4, v))
    off = 0
    for name, ttype, n in tensors:
        out += s(name) + struct.pack("<I", 1) + struct.pack("<Q", n) + struct.pack("<IQ", ttype, off)
        off += -(-n // align) * align
    out += b"\0" * (-len(out) % align)
    for _name, _t, n in tensors:
        out += b"\1" * n + b"\0" * (-n % align)
    path.write_bytes(bytes(out))


def test_gguf_header_reads_sizes_of_unknown_tensor_types(tmp_path):
    """Les types ternaires PrismML (142) font lever le paquet gguf ; pas notre lecteur."""
    from core.llama_server_backend import _gguf_header, needs_prism_fork
    m = tmp_path / "bonsai.gguf"
    _write_gguf_tensors(m, [("blk.0.attn_q.weight", 142, 96), ("output_norm.weight", 0, 64)])
    h = _gguf_header(m, tensors=True)
    assert h["kv"]["general.architecture"] == "llama"
    assert [(n, t, b) for n, t, b in h["tensors"]] == [
        ("blk.0.attn_q.weight", 142, 96), ("output_norm.weight", 0, 64)]
    assert needs_prism_fork(m) is True


def test_ordinary_gguf_does_not_need_prism(tmp_path):
    from core.llama_server_backend import needs_prism_fork
    m = tmp_path / "q4.gguf"
    _write_gguf_tensors(m, [("blk.0.attn_q.weight", 12, 64)])
    assert needs_prism_fork(m) is False
    assert needs_prism_fork(tmp_path / "absent.gguf") is False


def test_mixed_nvidia_amd_takes_vulkan(monkeypatch):
    """Le build CUDA ne voit pas la carte AMD : NVIDIA + AMD → Vulkan."""
    import core.llama_server_backend as mod
    monkeypatch.setattr(mod.platform, "system", lambda: "Linux")
    monkeypatch.setattr(mod, "_has_nvidia", lambda: True)
    monkeypatch.setattr(mod, "_has_amd_gpu", lambda: True)
    assert mod._platform_key() == "linux-vulkan"
    monkeypatch.setattr(mod, "_has_amd_gpu", lambda: False)
    assert mod._platform_key() == "linux-cuda"


def test_prism_release_skips_releases_without_the_asset(monkeypatch):
    """Certaines releases PrismML ne publient que les cudart Windows."""
    import core.llama_server_backend as mod

    class R:
        def json(self):
            return [
                {"tag_name": "prism-b10710-aaaaaaa", "assets": [{"name": "cudart-win.zip"}]},
                {"tag_name": "prism-b10709-9a9394a", "assets": [
                    {"name": "llama-prism-b10709-9a9394a-bin-linux-cuda-12.8-x64.tar.gz"}]},
            ]
    monkeypatch.setattr(mod._requests, "get", lambda *a, **k: R())
    assert mod._latest_build_tag("prism", "linux-cuda") == "prism-b10709-9a9394a"


def _kv_model(tmp_path, weights_gib=0.0):
    m = tmp_path / "dense.gguf"
    _write_gguf_tensors(m, [("blk.0.attn_q.weight", 12, 64)], kv={
        "general.architecture": "qwen2", "qwen2.block_count": 64,
        "qwen2.embedding_length": 5120, "qwen2.attention.head_count": 40,
        "qwen2.attention.head_count_kv": 8})
    return m


def test_kv_bytes_per_token_dense_and_hybrid(tmp_path):
    from core.llama_server_backend import kv_bytes_per_token
    assert kv_bytes_per_token(_kv_model(tmp_path)) == 64 * 8 * 256 * 2     # Qwen2.5-32B
    h = tmp_path / "hybrid.gguf"
    _write_gguf_tensors(h, [("x", 0, 32)], kv={
        "general.architecture": "qwen35moe", "qwen35moe.block_count": 40,
        "qwen35moe.embedding_length": 2048, "qwen35moe.attention.head_count": 16,
        "qwen35moe.attention.head_count_kv": 2, "qwen35moe.attention.key_length": 256,
        "qwen35moe.attention.value_length": 256, "qwen35moe.full_attention_interval": 4})
    assert kv_bytes_per_token(h) == 10 * 2 * 512 * 2                          # 1 couche sur 4


def test_kv_cache_quantized_only_when_context_does_not_fit(tmp_path, monkeypatch):
    """Reproduit la mesure 3090 seule + Qwen2.5-32B Q4_K_M (18.5 Gio) : f16 → q8_0 → q4_0."""
    import core.llama_server_backend as mod
    monkeypatch.delenv("VRM_KV_TYPE", raising=False)
    monkeypatch.setattr(mod, "_model_size_gib", lambda p: 18.48)
    m, dev = _kv_model(tmp_path), [{"id": "Vulkan0", "free_mib": 24098}]
    assert mod.kv_cache_flags(m, 16384, dev) == []                 # mesuré : tient en f16
    assert mod.kv_cache_flags(m, 24576, dev)[1] == "q8_0"          # f16 OOM, q8_0 26.9 tok/s
    assert mod.kv_cache_flags(m, 49152, dev)[1] == "q4_0"          # q8_0 OOM, q4_0 20.2 tok/s
    monkeypatch.setattr(mod, "_model_size_gib", lambda p: 40.0)    # ne tient pas : le plan décide
    assert mod.kv_cache_flags(m, 49152, dev) == []
    monkeypatch.setenv("VRM_KV_TYPE", "q8_0")
    assert mod.kv_cache_flags(m, 1024, dev) == ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]


def test_model_bigger_than_ram_is_mmapped_not_loaded(monkeypatch):
    """Sans mmap, un modèle plus gros que la RAM finit tué par l'OOM killer."""
    import core.llama_server_backend as mod
    monkeypatch.setattr(mod, "_server_help", lambda b: "-lm, --load-mode MODE")
    monkeypatch.setattr(mod, "_spec_flags", lambda h, m: [])
    monkeypatch.setattr(mod, "_too_big_for_ram", lambda p: True)
    assert mod._compat_flags("x", "/m.gguf") == ["--load-mode", "mmap"]
    monkeypatch.setattr(mod, "_too_big_for_ram", lambda p: False)
    assert mod._compat_flags("x", "/m.gguf") == ["--load-mode", "none"]
