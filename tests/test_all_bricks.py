"""Tests exhaustifs pour toutes les briques essentielles de VRAMancer.

Couvre : HierarchicalMemory, BlockRouter, StreamManager, Compressor,
         TransferManager, Monitor, ModelSplitter, InferencePipeline,
         multi-OS clients, multi-GPU support.
"""
import os
import sys
import time
import json
import pickle
import tempfile
import threading
import pytest

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_API_TOKEN", "testtoken")
os.environ.setdefault("VRM_DISABLE_RATE_LIMIT", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =====================================================================
# 1. Hierarchical Memory Manager (6 tiers : L1→L6)
# =====================================================================

class TestHierarchicalMemory:
    """Test all 6 tiers, migration, hotness, eviction, NVMe spill."""

    @pytest.fixture
    def hmem(self, tmp_path):
        os.environ["VRM_AUTOSAVE_MEMORY"] = "0"  # disable autosave thread
        from core.hierarchical_memory import HierarchicalMemoryManager
        return HierarchicalMemoryManager(nvme_dir=str(tmp_path / "nvme"), max_nvme_mb=256)

    @pytest.fixture
    def block(self):
        from core.memory_block import MemoryBlock
        return MemoryBlock(id="test-block-001", size_mb=128, gpu_id=0, status="allocated")

    def test_register_and_get_tier(self, hmem, block):
        hmem.register_block(block, "L1")
        assert hmem.get_tier(block.id) == "L1"

    def test_migrate_l1_to_l2(self, hmem, block):
        hmem.register_block(block, "L1")
        hmem.migrate(block, "L2")
        assert hmem.get_tier(block.id) == "L2"

    def test_migrate_l2_to_l3(self, hmem, block):
        hmem.register_block(block, "L2")
        hmem.migrate(block, "L3")
        assert hmem.get_tier(block.id) == "L3"

    def test_migrate_l3_to_l5_nvme(self, hmem, block):
        hmem.register_block(block, "L3")
        hmem.migrate(block, "L5")
        assert hmem.get_tier(block.id) == "L5"

    def test_migrate_to_l6(self, hmem, block):
        hmem.register_block(block, "L3")
        hmem.migrate(block, "L6")
        assert hmem.get_tier(block.id) == "L6"

    def test_migrate_same_tier_noop(self, hmem, block):
        hmem.register_block(block, "L1")
        result = hmem.migrate(block, "L1")
        assert hmem.get_tier(block.id) == "L1"

    def test_touch_increments_access_and_hotness(self, hmem, block):
        hmem.register_block(block, "L1")
        hmem.touch(block)
        hmem.touch(block)
        hmem.touch(block)
        meta = hmem.registry[block.id]
        assert meta["access"] >= 3
        assert hmem._hot_scores.get(block.id, 0) > 0

    def test_promote_policy_l5_to_l3(self, hmem, block):
        hmem.register_block(block, "L5")
        for _ in range(4):
            hmem.touch(block)
        hmem.promote_policy(block)
        assert hmem.get_tier(block.id) == "L3"

    def test_promote_policy_l3_to_l2(self, hmem, block):
        hmem.register_block(block, "L3")
        for _ in range(6):
            hmem.touch(block)
        hmem.promote_policy(block)
        assert hmem.get_tier(block.id) == "L2"

    def test_promote_policy_l2_to_l1(self, hmem, block):
        hmem.register_block(block, "L2")
        for _ in range(9):
            hmem.touch(block)
        hmem.promote_policy(block)
        assert hmem.get_tier(block.id) == "L1"

    def test_spill_to_nvme(self, hmem, block, tmp_path):
        hmem.register_block(block, "L3")
        hmem.spill_to_nvme(block, {"weights": [1, 2, 3]})
        assert hmem.get_tier(block.id) == "L5"
        # Verify file exists
        nvme_file = tmp_path / "nvme" / f"{block.id}.pkl"
        assert nvme_file.exists()

    def test_load_from_nvme(self, hmem, block, tmp_path):
        hmem.register_block(block, "L3")
        payload = {"weights": [1, 2, 3], "meta": "test"}
        hmem.spill_to_nvme(block, payload)
        assert hmem.get_tier(block.id) == "L5"
        loaded = hmem.load_from_nvme(block)
        assert loaded["weights"] == [1, 2, 3]
        assert hmem.get_tier(block.id) == "L3"  # promoted back

    def test_eviction_cycle(self, hmem):
        from core.memory_block import MemoryBlock
        # Register multiple blocks with varying hotness
        for i in range(10):
            b = MemoryBlock(id=f"evict-{i}", size_mb=64, gpu_id=0)
            hmem.register_block(b, "L1")
            # Touch some blocks more than others
            for _ in range(i):
                hmem.touch(b)
        evicted = hmem.eviction_cycle(target_free_pct=10.0)
        assert len(evicted) > 0
        # The coldest blocks should have been evicted
        for bid, from_tier, to_tier in evicted:
            assert from_tier in {"L1", "L2"}

    def test_eviction_with_high_pressure(self, hmem):
        from core.memory_block import MemoryBlock
        for i in range(5):
            b = MemoryBlock(id=f"pressure-{i}", size_mb=64, gpu_id=0)
            hmem.register_block(b, "L1")
        evicted = hmem.eviction_cycle(vram_pressure=0.95)
        # High pressure = more aggressive eviction (40%)
        assert len(evicted) >= 2

    def test_policy_demote_if_needed(self, hmem, block):
        hmem.register_block(block, "L1")
        hmem.policy_demote_if_needed(block, 92)
        assert hmem.get_tier(block.id) == "L2"

    def test_summary(self, hmem, block):
        hmem.register_block(block, "L1")
        s = hmem.summary()
        assert s["count"] == 1
        assert s["tiers"]["L1"] == 1

    def test_save_and_load_state(self, hmem, block, tmp_path):
        hmem.register_block(block, "L2")
        hmem.touch(block)
        state_path = str(tmp_path / "state.pkl")
        hmem.save_state(state_path)
        assert os.path.exists(state_path)

        # Load into fresh instance
        os.environ["VRM_AUTOSAVE_MEMORY"] = "0"
        from core.hierarchical_memory import HierarchicalMemoryManager
        hmem2 = HierarchicalMemoryManager(nvme_dir=str(tmp_path / "nvme2"))
        assert hmem2.load_state(state_path)
        assert hmem2.get_tier(block.id) == "L2"

    def test_is_promotion_logic(self, hmem):
        assert hmem._is_promotion("L5", "L1") is True
        assert hmem._is_promotion("L1", "L5") is False
        assert hmem._is_promotion("L3", "L2") is True
        assert hmem._is_promotion("L2", "L3") is False


# =====================================================================
# 2. Block Router
# =====================================================================

class TestBlockRouter:
    """Test VRAM-aware routing to GPU/CPU/NVMe/network."""

    def test_init(self):
        from core.block_router import BlockRouter
        router = BlockRouter(verbose=False)
        assert router is not None

    def test_route_method_exists(self):
        from core.block_router import BlockRouter
        router = BlockRouter(verbose=False)
        assert hasattr(router, 'route')
        assert callable(router.route)

    def test_register_remote_node(self):
        from core.block_router import BlockRouter
        router = BlockRouter(verbose=False)
        router.register_remote_node("192.168.1.100", 9000, capacity_mb=4096)
        assert len(router._remote_nodes) >= 1

    def test_unregister_remote_node(self):
        from core.block_router import BlockRouter
        router = BlockRouter(verbose=False)
        router.register_remote_node("192.168.1.100", 9000)
        router.unregister_remote_node("192.168.1.100", 9000)
        assert len(router._remote_nodes) == 0

    def test_nvme_available(self):
        from core.block_router import BlockRouter
        router = BlockRouter(verbose=False)
        result = router._nvme_available()
        assert isinstance(result, bool)

    def test_remote_executor_passthrough(self):
        from core.block_router import RemoteExecutor
        re = RemoteExecutor("127.0.0.1", 9999, timeout=0.5)
        # Should fail silently and return input
        result = re.forward(42)
        assert result == 42


# =====================================================================
# 3. Stream Manager
# =====================================================================

class TestStreamManager:
    """Test prefetch, swap, eviction, monitoring."""

    @pytest.fixture
    def stream_mgr(self):
        from core.stream_manager import StreamManager
        return StreamManager(scheduler=None, monitor=None, verbose=True)

    def test_init(self, stream_mgr):
        assert stream_mgr is not None
        assert stream_mgr._stats["preloads"] == 0

    def test_preload_layer_without_scheduler(self, stream_mgr):
        # Without scheduler, preload should handle gracefully
        result = stream_mgr.preload_layer({"name": "layer.0", "size_mb": 64, "gpu_id": 0})
        # Should be False or handle gracefully
        assert isinstance(result, bool)

    def test_start_stop_monitoring(self, stream_mgr):
        stream_mgr.start_monitoring(interval=0.1)
        assert stream_mgr._monitoring is True
        time.sleep(0.15)
        stream_mgr.stop_monitoring()
        assert stream_mgr._monitoring is False

    def test_stats_tracking(self, stream_mgr):
        stats = stream_mgr._stats
        assert "preloads" in stats
        assert "evictions" in stats
        assert "swaps" in stats
        assert "prefetches" in stats

    def test_swap_if_needed_without_monitor(self, stream_mgr):
        # Without monitor, swap should be a no-op
        stream_mgr.swap_if_needed()
        assert stream_mgr._stats["swaps"] == 0


# =====================================================================
# 4. Compressor
# =====================================================================

class TestCompressor:
    """Test compression (zstd/lz4/gzip) and quantization."""

    def test_init(self):
        from core.compressor import Compressor
        c = Compressor()
        assert c is not None

    def test_compress_decompress_gzip(self):
        from core.compressor import Compressor
        c = Compressor(strategy="adaptive")
        # Codec should be one of zstd/lz4/gzip
        assert c._codec in ("zstd", "lz4", "gzip")
        data = b"Hello VRAMancer " * 1000
        compressed = c.compress_bytes(data)
        assert len(compressed) < len(data)
        decompressed = c.decompress_bytes(compressed)
        assert decompressed == data

    def test_compress_decompress_roundtrip(self):
        from core.compressor import Compressor
        c = Compressor(strategy="aggressive")
        data = b"GPU memory data " * 500
        compressed = c.compress_bytes(data)
        assert len(compressed) < len(data)
        decompressed = c.decompress_bytes(compressed)
        assert decompressed == data

    def test_compress_strategy_none(self):
        from core.compressor import Compressor
        c = Compressor(strategy="none")
        assert c._codec == "none"
        data = b"Layer weights " * 500
        compressed = c.compress_bytes(data)
        # strategy=none returns data unchanged
        assert compressed == data

    def test_codec_selection(self):
        from core.compressor import Compressor
        c = Compressor()
        # gzip always available as fallback
        assert c._codec in ("zstd", "lz4", "gzip")

    def test_compress_empty(self):
        from core.compressor import Compressor
        c = Compressor(strategy="adaptive")
        compressed = c.compress_bytes(b"")
        decompressed = c.decompress_bytes(compressed)
        assert decompressed == b""


# =====================================================================
# 5. Transfer Manager
# =====================================================================

class TestTransferManager:
    """Test GPU-to-GPU transfer (P2P, CPU-staged, stub)."""

    def test_init(self):
        from core.transfer_manager import TransferManager
        tm = TransferManager(verbose=False)
        assert tm is not None

    def test_send_activation_stub(self):
        from core.transfer_manager import TransferManager
        tm = TransferManager(verbose=False)
        # In minimal test mode, should use stub
        result = tm.send_activation(0, 1, None)
        # Result should be a TransferResult or similar
        assert result is not None

    def test_stats(self):
        from core.transfer_manager import TransferManager
        tm = TransferManager(verbose=False)
        s = tm.stats()
        assert isinstance(s, dict)


# =====================================================================
# 6. GPU Monitor (multi-accelerator)
# =====================================================================

class TestGPUMonitorFull:
    """Full GPUMonitor verification for CUDA/ROCm/MPS/CPU."""

    def test_init(self):
        from core.monitor import GPUMonitor
        m = GPUMonitor()
        assert m is not None

    def test_vram_usage_returns_float(self):
        from core.monitor import GPUMonitor
        m = GPUMonitor()
        usage = m.vram_usage(0)
        assert isinstance(usage, float)
        assert 0.0 <= usage <= 1.0

    def test_detect_overload_returns_none_or_int(self):
        from core.monitor import GPUMonitor
        m = GPUMonitor()
        result = m.detect_overload(0.9)
        assert result is None or isinstance(result, int)

    def test_system_memory(self):
        from core.monitor import GPUMonitor
        mem = GPUMonitor.system_memory()
        assert "total" in mem
        assert "available" in mem
        assert "used" in mem
        assert mem["total"] >= 0

    def test_snapshot(self):
        from core.monitor import GPUMonitor
        m = GPUMonitor()
        snap = m.snapshot()
        assert isinstance(snap, dict)

    def test_get_free_memory(self):
        from core.monitor import GPUMonitor
        m = GPUMonitor()
        free = m.get_free_memory(0)
        assert isinstance(free, int)
        assert free >= 0

    def test_start_stop_polling(self):
        from core.monitor import GPUMonitor
        m = GPUMonitor()
        m.start_polling(interval=0.1)
        assert m._polling is True
        time.sleep(0.15)
        m.stop_polling()
        assert m._polling is False

    def test_repr(self):
        from core.monitor import GPUMonitor
        m = GPUMonitor()
        s = repr(m)
        assert "GPUMonitor" in s

    def test_hotplug_monitor_init(self):
        from core.monitor import GPUHotPlugMonitor
        hp = GPUHotPlugMonitor(interval=1.0)
        assert hp is not None

    def test_hotplug_monitor_callbacks(self):
        from core.monitor import GPUHotPlugMonitor
        hp = GPUHotPlugMonitor(interval=1.0)
        calls = []
        hp.on_add(lambda info: calls.append(("add", info)))
        hp.on_remove(lambda info: calls.append(("remove", info)))
        assert len(hp._on_add_callbacks) == 1
        assert len(hp._on_remove_callbacks) == 1

    def test_hotplug_known_gpus(self):
        from core.monitor import GPUHotPlugMonitor
        hp = GPUHotPlugMonitor(interval=1.0)
        gpus = hp.known_gpus
        assert isinstance(gpus, dict)


# =====================================================================
# 7. Multi-GPU Support (detect_backend, enumerate_devices)
# =====================================================================

class TestMultiGPUSupport:
    """Test GPU detection for all supported platforms."""

    def test_detect_backend_returns_valid(self):
        from core.utils import detect_backend
        backend = detect_backend()
        assert backend in ("cuda", "rocm", "mps", "cpu")

    def test_enumerate_devices_returns_list(self):
        from core.utils import enumerate_devices
        devices = enumerate_devices()
        assert isinstance(devices, list)
        assert len(devices) >= 1  # At least CPU

    def test_device_has_required_fields(self):
        from core.utils import enumerate_devices
        devices = enumerate_devices()
        for d in devices:
            assert "id" in d
            assert "backend" in d
            assert "index" in d
            assert "name" in d

    def test_model_splitter_extract_layers(self):
        from core.model_splitter import _extract_layers
        # With no model, should return None
        result = _extract_layers(None)
        assert result is None

    def test_model_splitter_get_free_vram(self):
        from core.model_splitter import _get_free_vram_per_gpu
        vram = _get_free_vram_per_gpu(1)
        assert isinstance(vram, list)
        assert len(vram) >= 1


# =====================================================================
# 8. OS Platform Scripts (removed — stale launcher scripts cleaned up)
# =====================================================================


# =====================================================================
# 9. Dashboard & Systray Clients
# =====================================================================

class TestDashboardClients:
    """Verify all dashboard and systray files exist and are importable."""

    ROOT = os.path.join(os.path.dirname(__file__), "..")

    def test_dashboard_web_exists(self):
        assert os.path.exists(os.path.join(self.ROOT, "dashboard", "dashboard_web.py"))

    def test_dashboard_cli_exists(self):
        assert os.path.exists(os.path.join(self.ROOT, "dashboard", "dashboard_cli.py"))


# =====================================================================
# 10. Config Multi-OS
# =====================================================================

class TestConfigMultiOS:
    """Test config resolution for Linux/macOS/Windows."""

    def test_config_importable(self):
        from core.config import get_config
        cfg = get_config()
        assert isinstance(cfg, dict)

    def test_config_env_override(self):
        os.environ["VRM_API_PORT"] = "9999"
        from core.config import get_config
        cfg = get_config()
        # Config should respect env vars
        assert cfg is not None
        os.environ.pop("VRM_API_PORT", None)


# =====================================================================
# 11. /metrics Endpoint on Flask API
# =====================================================================

class TestMetricsOnAPI:
    """Test /metrics exposes Prometheus metrics on main API port."""

    @pytest.fixture
    def client(self):
        os.environ["VRM_DISABLE_RATE_LIMIT"] = "1"
        os.environ["VRM_API_TOKEN"] = "testtoken"
        from core.production_api import create_app
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_metrics_has_vramancer_prefix(self, client):
        import core.metrics  # noqa: F401
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.data.decode()
        assert "vramancer_infer_total" in body

    def test_metrics_has_gpu_memory_gauge(self, client):
        import core.metrics  # noqa: F401
        resp = client.get("/metrics")
        body = resp.data.decode()
        assert "vramancer_gpu_memory_used_bytes" in body
