"""Tests for the cross-vendor GPU bridge (AMD ↔ NVIDIA).

Run with: VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 pytest tests/test_cross_vendor.py -v
"""
import os
import sys
import struct
import pytest

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")
os.environ.setdefault("VRM_DISABLE_RATE_LIMIT", "1")


# ═══════════════════════════════════════════════════════════════════════════
# CrossVendorBridge — imports and enums
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossVendorImports:
    """Verify all cross-vendor modules import cleanly."""

    def test_import_bridge(self):
        from core.cross_vendor_bridge import CrossVendorBridge
        assert CrossVendorBridge is not None

    def test_import_enums(self):
        from core.cross_vendor_bridge import (
            CrossVendorMethod, GPUVendor, CrossVendorResult,
        )
        assert CrossVendorMethod.PIPELINED_ASYNC.name == "PIPELINED_ASYNC"
        assert CrossVendorMethod.DMABUF_ZERO_COPY.name == "DMABUF_ZERO_COPY"
        assert CrossVendorMethod.REBAR_MMAP.name == "REBAR_MMAP"
        assert CrossVendorMethod.SHARED_MEMORY.name == "SHARED_MEMORY"
        assert CrossVendorMethod.CPU_STAGED.name == "CPU_STAGED"
        assert CrossVendorMethod.STUB.name == "STUB"

        assert GPUVendor.NVIDIA.value == "nvidia"
        assert GPUVendor.AMD.value == "amd"
        assert GPUVendor.INTEL.value == "intel"
        assert GPUVendor.UNKNOWN.value == "unknown"

    def test_import_transports(self):
        from core.cross_vendor_bridge import (
            PipelinedTransport, SharedMemTransport,
            DMABufTransport, ReBarTransport,
        )
        assert PipelinedTransport is not None
        assert SharedMemTransport is not None
        assert DMABufTransport is not None
        assert ReBarTransport is not None

    def test_import_utilities(self):
        from core.cross_vendor_bridge import (
            detect_gpu_vendor, detect_rebar, detect_dmabuf_support,
            is_cross_vendor, is_consumer_gpu,
            get_cross_vendor_bridge, reset_cross_vendor_bridge,
        )
        assert callable(detect_gpu_vendor)
        assert callable(detect_rebar)
        assert callable(detect_dmabuf_support)
        assert callable(is_cross_vendor)
        assert callable(is_consumer_gpu)


# ═══════════════════════════════════════════════════════════════════════════
# GPUVendor detection
# ═══════════════════════════════════════════════════════════════════════════

class TestGPUVendorDetection:
    """Test GPU vendor identification logic."""

    def test_detect_vendor_stub_mode(self):
        from core.cross_vendor_bridge import detect_gpu_vendor, GPUVendor
        # In VRM_MINIMAL_TEST mode, should return UNKNOWN (no real GPU)
        vendor = detect_gpu_vendor(0)
        assert vendor == GPUVendor.UNKNOWN

    def test_is_consumer_nvidia(self):
        from core.cross_vendor_bridge import is_consumer_gpu
        assert is_consumer_gpu("NVIDIA GeForce RTX 4090") is True
        assert is_consumer_gpu("NVIDIA GeForce GTX 1080 Ti") is True
        assert is_consumer_gpu("NVIDIA RTX 3090") is True
        assert is_consumer_gpu("NVIDIA RTX 5070 Ti") is True
        assert is_consumer_gpu("Titan RTX") is True

    def test_is_consumer_amd(self):
        from core.cross_vendor_bridge import is_consumer_gpu
        assert is_consumer_gpu("AMD Radeon RX 7900 XTX") is True
        assert is_consumer_gpu("AMD Radeon RX 6800 XT") is True
        assert is_consumer_gpu("Radeon Pro W7900") is True
        assert is_consumer_gpu("RX 9070 XT") is True

    def test_is_not_consumer_pro(self):
        from core.cross_vendor_bridge import is_consumer_gpu
        assert is_consumer_gpu("NVIDIA A100") is False
        assert is_consumer_gpu("NVIDIA H100") is False
        assert is_consumer_gpu("AMD Instinct MI300X") is False

    def test_is_cross_vendor_stub(self):
        from core.cross_vendor_bridge import is_cross_vendor
        # In stub mode, always returns False
        assert is_cross_vendor(0, 1) is False


# ═══════════════════════════════════════════════════════════════════════════
# CrossVendorResult
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossVendorResult:
    """Test result metadata dataclass."""

    def test_bandwidth_calculation(self):
        from core.cross_vendor_bridge import (
            CrossVendorResult, CrossVendorMethod, GPUVendor,
        )
        result = CrossVendorResult(
            method=CrossVendorMethod.PIPELINED_ASYNC,
            source_gpu=0, target_gpu=1,
            source_vendor=GPUVendor.NVIDIA, target_vendor=GPUVendor.AMD,
            bytes_transferred=1_000_000_000,  # 1 GB
            duration_s=0.05,  # 50 ms
            chunks_used=4,
        )
        # 1 GB in 50ms = 160 Gbps
        assert result.bandwidth_gbps == pytest.approx(160.0, rel=0.01)

    def test_zero_duration(self):
        from core.cross_vendor_bridge import (
            CrossVendorResult, CrossVendorMethod, GPUVendor,
        )
        result = CrossVendorResult(
            method=CrossVendorMethod.STUB,
            source_gpu=0, target_gpu=1,
            source_vendor=GPUVendor.UNKNOWN, target_vendor=GPUVendor.UNKNOWN,
            bytes_transferred=0, duration_s=0.0,
        )
        assert result.bandwidth_gbps == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# GPUDeviceInfo
# ═══════════════════════════════════════════════════════════════════════════

class TestGPUDeviceInfo:
    """Test GPU device info dataclass."""

    def test_pcie_bandwidth_gen4_x16(self):
        from core.cross_vendor_bridge import GPUDeviceInfo, GPUVendor
        info = GPUDeviceInfo(
            index=0, vendor=GPUVendor.NVIDIA, name="RTX 4090",
            pcie_gen=4, pcie_width=16,
        )
        # PCIe 4.0 x16 ≈ 31.5 GB/s (128b/130b encoding)
        assert 30.0 < info.pcie_bandwidth_gbps < 33.0

    def test_pcie_bandwidth_gen5_x16(self):
        from core.cross_vendor_bridge import GPUDeviceInfo, GPUVendor
        info = GPUDeviceInfo(
            index=0, vendor=GPUVendor.AMD, name="RX 9070 XT",
            pcie_gen=5, pcie_width=16,
        )
        # PCIe 5.0 x16 ≈ 63 GB/s
        assert 60.0 < info.pcie_bandwidth_gbps < 66.0


# ═══════════════════════════════════════════════════════════════════════════
# ReBAR detection
# ═══════════════════════════════════════════════════════════════════════════

class TestReBarDetection:
    """Test Resizable BAR detection via sysfs."""

    def test_rebar_stub_mode(self):
        from core.cross_vendor_bridge import detect_rebar
        enabled, bar_size = detect_rebar(0)
        # In stub mode, ReBAR detection is disabled
        assert enabled is False
        assert bar_size == 0

    def test_rebar_non_linux(self):
        """On non-Linux systems, ReBAR detection should gracefully return False."""
        from core.cross_vendor_bridge import detect_rebar
        enabled, _ = detect_rebar(0)
        assert enabled is False


# ═══════════════════════════════════════════════════════════════════════════
# DMA-BUF detection
# ═══════════════════════════════════════════════════════════════════════════

class TestDMABufDetection:
    """Test DMA-BUF availability detection."""

    def test_dmabuf_stub_mode(self):
        from core.cross_vendor_bridge import detect_dmabuf_support
        supported, nodes = detect_dmabuf_support()
        assert supported is False


# ═══════════════════════════════════════════════════════════════════════════
# SharedMemTransport
# ═══════════════════════════════════════════════════════════════════════════

class TestSharedMemTransport:
    """Test the shared memory ring buffer transport."""

    def test_init(self):
        from core.cross_vendor_bridge import SharedMemTransport
        shm = SharedMemTransport(name="/vrm_test_ring")
        assert shm.shm_name == "/vrm_test_ring"
        assert shm.total_size > 0
        assert shm.MAGIC == 0x56524D58

    @pytest.mark.skipif(sys.platform != "linux",
                        reason="SharedMem requires Linux /dev/shm")
    def test_open_close(self):
        from core.cross_vendor_bridge import SharedMemTransport
        import pathlib
        shm = SharedMemTransport(name="/vrm_test_open")
        try:
            opened = shm.open(create=True)
            assert opened is True
        finally:
            shm.close()
            # Cleanup
            p = pathlib.Path("/dev/shm/vrm_test_open")
            if p.exists():
                p.unlink()

    @pytest.mark.skipif(sys.platform != "linux",
                        reason="SharedMem requires Linux /dev/shm")
    def test_header_roundtrip(self):
        from core.cross_vendor_bridge import SharedMemTransport
        import pathlib
        shm = SharedMemTransport(name="/vrm_test_header")
        try:
            shm.open(create=True)
            # Write header
            shm._write_header(0x56524D58, 5, 3, 42, 0)
            # Read back
            magic, write_off, read_off, msg_count, flags = shm._read_header()
            assert magic == 0x56524D58
            assert write_off == 5
            assert read_off == 3
            assert msg_count == 42
        finally:
            shm.close()
            p = pathlib.Path("/dev/shm/vrm_test_header")
            if p.exists():
                p.unlink()


# ═══════════════════════════════════════════════════════════════════════════
# CrossVendorBridge — stub mode
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossVendorBridgeStub:
    """Test the bridge in stub mode (VRM_MINIMAL_TEST=1)."""

    def test_init_stub(self):
        from core.cross_vendor_bridge import CrossVendorBridge
        bridge = CrossVendorBridge()
        assert bridge.available is False

    def test_transfer_stub(self):
        from core.cross_vendor_bridge import (
            CrossVendorBridge, CrossVendorMethod,
        )
        bridge = CrossVendorBridge()
        output, result = bridge.transfer(0, 1, None)
        assert output is None
        assert result.method == CrossVendorMethod.STUB
        assert result.bytes_transferred == 0

    def test_stats_stub(self):
        from core.cross_vendor_bridge import CrossVendorBridge
        bridge = CrossVendorBridge()
        stats = bridge.stats()
        assert stats["available"] is False
        assert stats["cross_vendor_pairs"] == 0
        assert stats["has_dmabuf"] is False
        assert stats["has_rebar"] is False
        assert stats["transfers"] == 0

    def test_is_cross_vendor_pair_stub(self):
        from core.cross_vendor_bridge import CrossVendorBridge
        bridge = CrossVendorBridge()
        assert bridge.is_cross_vendor_pair(0, 1) is False

    def test_close_safe(self):
        from core.cross_vendor_bridge import CrossVendorBridge
        bridge = CrossVendorBridge()
        bridge.close()  # Should not crash


# ═══════════════════════════════════════════════════════════════════════════
# Singleton
# ═══════════════════════════════════════════════════════════════════════════

class TestCrossVendorSingleton:
    """Test singleton lifecycle."""

    def test_get_singleton(self):
        from core.cross_vendor_bridge import (
            get_cross_vendor_bridge, reset_cross_vendor_bridge,
        )
        bridge = get_cross_vendor_bridge()
        bridge2 = get_cross_vendor_bridge()
        assert bridge is bridge2
        reset_cross_vendor_bridge()

    def test_reset_singleton(self):
        from core.cross_vendor_bridge import (
            get_cross_vendor_bridge, reset_cross_vendor_bridge,
        )
        bridge1 = get_cross_vendor_bridge()
        reset_cross_vendor_bridge()
        bridge2 = get_cross_vendor_bridge()
        assert bridge1 is not bridge2
        reset_cross_vendor_bridge()


# ═══════════════════════════════════════════════════════════════════════════
# TransferManager integration
# ═══════════════════════════════════════════════════════════════════════════

class TestTransferManagerCrossVendor:
    """Test TransferManager cross-vendor integration."""

    def test_cross_vendor_transport_method_enum(self):
        from core.transfer_manager import TransportMethod
        assert TransportMethod.CROSS_VENDOR.name == "CROSS_VENDOR"

    def test_consumer_gpu_amd(self):
        from core.transfer_manager import _is_consumer_gpu
        assert _is_consumer_gpu("AMD Radeon RX 7900 XTX") is True
        assert _is_consumer_gpu("Radeon RX 6800 XT") is True
        assert _is_consumer_gpu("RX 7800 XT") is True

    def test_consumer_gpu_nvidia(self):
        from core.transfer_manager import _is_consumer_gpu
        assert _is_consumer_gpu("NVIDIA GeForce RTX 4090") is True
        assert _is_consumer_gpu("RTX 5070 Ti") is True

    def test_consumer_gpu_professional(self):
        from core.transfer_manager import _is_consumer_gpu
        assert _is_consumer_gpu("NVIDIA A100") is False
        assert _is_consumer_gpu("AMD Instinct MI300X") is False

    def test_stats_has_method_preference(self):
        from core.transfer_manager import TransferManager
        tm = TransferManager()
        stats = tm.stats()
        assert "CROSS_VENDOR" in stats["method_preference"]


# ═══════════════════════════════════════════════════════════════════════════
# Utils per-device backend
# ═══════════════════════════════════════════════════════════════════════════

class TestPerDeviceBackend:
    """Test detect_device_backend() per-device vendor detection."""

    def test_import(self):
        from core.utils import detect_device_backend
        assert callable(detect_device_backend)

    def test_stub_returns_cpu(self):
        from core.utils import detect_device_backend
        # No real GPU in test mode
        result = detect_device_backend(0)
        assert result == 'cpu'

    def test_enumerate_devices_has_vendor_field(self):
        from core.utils import enumerate_devices
        devices = enumerate_devices()
        assert len(devices) >= 1
        for dev in devices:
            assert 'vendor' in dev


# ═══════════════════════════════════════════════════════════════════════════
# VRAM Lending cross-vendor
# ═══════════════════════════════════════════════════════════════════════════

class TestVRAMLendingCrossVendor:
    """Test VRAM lending pool with cross-vendor awareness."""

    def test_gpu_budget_vendor_field(self):
        from core.vram_lending import GPUBudget
        budget = GPUBudget(gpu_id=0, vendor="nvidia")
        assert budget.vendor == "nvidia"

    def test_lending_policy_cross_vendor(self):
        from core.vram_lending import LendingPolicy
        policy = LendingPolicy()
        assert hasattr(policy, 'prefer_same_vendor')
        assert hasattr(policy, 'cross_vendor_penalty')
        assert policy.prefer_same_vendor is True
        assert 0.0 <= policy.cross_vendor_penalty <= 1.0

    def test_register_gpu_with_vendor(self):
        from core.vram_lending import VRAMLendingPool
        pool = VRAMLendingPool()
        budget = pool.register_gpu(
            gpu_id=0, total_bytes=int(24e9), model_bytes=int(12e9),
            device_name="NVIDIA RTX 4090", vendor="nvidia",
        )
        assert budget.vendor == "nvidia"

        budget2 = pool.register_gpu(
            gpu_id=1, total_bytes=int(16e9), model_bytes=int(8e9),
            device_name="AMD Radeon RX 7900 XTX", vendor="amd",
        )
        assert budget2.vendor == "amd"
        pool.close()

    def test_vendor_auto_detect(self):
        from core.vram_lending import VRAMLendingPool
        pool = VRAMLendingPool()
        budget = pool.register_gpu(
            gpu_id=0, total_bytes=int(24e9),
            device_name="AMD Instinct MI300X",
        )
        assert budget.vendor == "amd"

        budget2 = pool.register_gpu(
            gpu_id=1, total_bytes=int(24e9),
            device_name="NVIDIA GeForce RTX 5090",
        )
        assert budget2.vendor == "nvidia"
        pool.close()

    def test_cross_vendor_lending_works(self):
        """Cross-vendor lending should work, with a scoring penalty."""
        from core.vram_lending import VRAMLendingPool, LendingPolicy

        policy = LendingPolicy(cross_vendor_penalty=0.15)
        pool = VRAMLendingPool(policy=policy)

        pool.register_gpu(
            gpu_id=0, total_bytes=int(24e9), model_bytes=int(12e9),
            device_name="NVIDIA RTX 4090", vendor="nvidia",
        )
        pool.register_gpu(
            gpu_id=1, total_bytes=int(16e9), model_bytes=int(4e9),
            device_name="AMD Radeon RX 7900 XTX", vendor="amd",
        )

        # GPU 0 (NVIDIA) borrows from GPU 1 (AMD) — cross-vendor lending
        lease = pool.borrow(
            borrower_gpu=0, size_bytes=int(2e9), purpose="kv_cache",
        )
        # Should succeed — cross-vendor lending is allowed, just penalized
        assert lease is not None
        assert lease.owner_gpu == 1
        assert lease.borrower_gpu == 0

        pool.close()

    def test_same_vendor_preferred(self):
        """When a same-vendor GPU is available, it should score higher."""
        from core.vram_lending import VRAMLendingPool, LendingPolicy

        policy = LendingPolicy(cross_vendor_penalty=0.15)
        pool = VRAMLendingPool(policy=policy)

        # GPU 0: NVIDIA borrower
        pool.register_gpu(
            gpu_id=0, total_bytes=int(24e9), model_bytes=int(20e9),
            device_name="NVIDIA RTX 4090", vendor="nvidia",
        )
        # GPU 1: AMD lender (cross-vendor)
        pool.register_gpu(
            gpu_id=1, total_bytes=int(16e9), model_bytes=int(4e9),
            device_name="AMD RX 7900 XTX", vendor="amd",
        )
        # GPU 2: NVIDIA lender (same vendor)
        pool.register_gpu(
            gpu_id=2, total_bytes=int(16e9), model_bytes=int(4e9),
            device_name="NVIDIA RTX 3090", vendor="nvidia",
        )

        lease = pool.borrow(borrower_gpu=0, size_bytes=int(2e9))
        assert lease is not None
        # Same-vendor NVIDIA should be preferred
        assert lease.owner_gpu == 2

        pool.close()


# ═══════════════════════════════════════════════════════════════════════════
# Dtype mapping utilities
# ═══════════════════════════════════════════════════════════════════════════

class TestDtypeUtils:
    """Test tensor serialization utilities."""

    def test_dtype_roundtrip(self):
        from core.cross_vendor_bridge import _dtype_to_code, _code_to_dtype
        # These need torch to be available
        codes_tested = [0, 1, 4, 7, 8]
        for code in codes_tested:
            # In stub mode, functions handle gracefully
            result = _code_to_dtype(code)
            # Just verify no crash

    def test_torch_to_numpy_dtype(self):
        from core.cross_vendor_bridge import _torch_to_numpy_dtype
        # In stub mode, should return a default
        result = _torch_to_numpy_dtype(None)
        assert result == 'float32'
