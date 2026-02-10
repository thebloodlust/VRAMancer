"""Tests for VTP (VRAMancer Transport Protocol) — LLM-optimised transport.

Tests cover:
  - TensorHeader binary serialization/deserialization
  - LLMTransport stub mode (no GPU/RDMA needed)
  - Transport tier selection logic
  - KV cache streaming protocol
  - Connection info exchange
  - Stats tracking
  - GPUMemoryRegion and DoubleBufferedRegion (stubs)

All tests run without GPU/RDMA hardware (VRM_MINIMAL_TEST=1).
"""
import os
import sys
import struct
import pytest

# Ensure test environment
os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_API_TOKEN", "testtoken")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.network.llm_transport import (
    TensorHeader, VTPOpcode, VTPFlags, TransportTier,
    LLMTransport, VTPConnectionInfo, VTPConnection,
    select_optimal_tier, VTP_VERSION, GPUMemoryRegion,
    DoubleBufferedRegion,
)


# ═══════════════════════════════════════════════════════════════════════════
# TensorHeader tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTensorHeader:
    """Test binary header serialization/deserialization."""

    def test_encode_decode_roundtrip(self):
        """Header encodes to 64 bytes and decodes back identically."""
        hdr = TensorHeader(
            opcode=VTPOpcode.TENSOR,
            flags=VTPFlags.GPUDIRECT | VTPFlags.ONE_SIDED,
            payload_bytes=1024,
            layer_id=12,
            seq_id=42,
            src_gpu=0,
            dst_gpu=1,
            ndim=3,
            dtype_code=0,
            shape=(16, 32, 768),
        )
        encoded = hdr.encode()
        assert len(encoded) == 64, f"Header should be 64 bytes, got {len(encoded)}"

        decoded = TensorHeader.decode(encoded)
        assert decoded.opcode == VTPOpcode.TENSOR
        assert decoded.flags == (VTPFlags.GPUDIRECT | VTPFlags.ONE_SIDED)
        assert decoded.payload_bytes == 1024
        assert decoded.layer_id == 12
        assert decoded.seq_id == 42
        assert decoded.src_gpu == 0
        assert decoded.dst_gpu == 1
        assert decoded.ndim == 3
        assert decoded.dtype_code == 0
        assert decoded.shape == (16, 32, 768)

    def test_encode_size_always_64(self):
        """Header is always exactly 64 bytes regardless of shape."""
        for ndim in range(0, 9):
            shape = tuple(range(1, ndim + 1))
            hdr = TensorHeader(ndim=min(ndim, 8), shape=shape[:8])
            assert len(hdr.encode()) == 64

    def test_version_mismatch_raises(self):
        """Decoding a header with wrong version raises ValueError."""
        hdr = TensorHeader()
        data = bytearray(hdr.encode())
        data[0] = 99  # corrupt version
        with pytest.raises(ValueError, match="version mismatch"):
            TensorHeader.decode(bytes(data))

    def test_checksum_present(self):
        """Checksum field is nonzero after encoding."""
        hdr = TensorHeader(payload_bytes=100, layer_id=5)
        encoded = hdr.encode()
        decoded = TensorHeader.decode(encoded)
        assert decoded.checksum > 0

    def test_all_opcodes(self):
        """All opcodes survive roundtrip."""
        for op in VTPOpcode:
            hdr = TensorHeader(opcode=op)
            decoded = TensorHeader.decode(hdr.encode())
            assert decoded.opcode == op

    def test_all_flags(self):
        """All flags survive roundtrip."""
        flags = (VTPFlags.GPUDIRECT | VTPFlags.COMPRESSED |
                 VTPFlags.LAST_CHUNK | VTPFlags.INLINE |
                 VTPFlags.ONE_SIDED | VTPFlags.URGENT |
                 VTPFlags.KV_PARTIAL)
        hdr = TensorHeader(flags=flags)
        decoded = TensorHeader.decode(hdr.encode())
        assert decoded.flags == flags

    def test_kv_cache_opcode(self):
        """KV cache opcode roundtrip."""
        hdr = TensorHeader(
            opcode=VTPOpcode.KV_CACHE,
            flags=VTPFlags.KV_PARTIAL,
            layer_id=7,
            shape=(1, 32, 128, 64),
            ndim=4,
            dtype_code=1,  # float16
        )
        decoded = TensorHeader.decode(hdr.encode())
        assert decoded.opcode == VTPOpcode.KV_CACHE
        assert decoded.flags & VTPFlags.KV_PARTIAL
        assert decoded.shape == (1, 32, 128, 64)

    def test_short_data_raises(self):
        """Too-short data raises ValueError."""
        with pytest.raises(ValueError, match="too short"):
            TensorHeader.decode(b'\x00' * 10)


# ═══════════════════════════════════════════════════════════════════════════
# LLMTransport stub mode tests
# ═══════════════════════════════════════════════════════════════════════════

class TestLLMTransportStub:
    """Test LLMTransport in stub mode (no GPU/RDMA)."""

    def test_create_transport(self):
        """Transport creates in stub mode."""
        t = LLMTransport(node_id="test-node")
        assert t.node_id == "test-node"
        assert t.tier == TransportTier.STUB

    def test_register_gpu_stub(self):
        """GPU registration is no-op in stub mode."""
        t = LLMTransport(node_id="test")
        assert t.register_gpu(0, size_mb=256) is True

    def test_connect_peer_stub(self):
        """Peer connection is no-op in stub mode."""
        t = LLMTransport(node_id="test")
        info = VTPConnectionInfo(node_id="peer")
        assert t.connect_peer("peer", info) is True

    def test_send_tensor_stub(self):
        """Stub send returns correct metadata."""
        t = LLMTransport(node_id="test")
        # Simulate a simple send with no real tensor
        result = t._stub_send(b"hello", "peer", 0, 5)
        assert result["method"] == "stub"
        assert result["dst_node"] == "peer"
        assert result["layer_id"] == 5
        assert result["tier"] == TransportTier.STUB.name

    def test_send_tensor_records_stats(self):
        """Sends record statistics."""
        t = LLMTransport(node_id="test")
        # In stub mode, send_tensor goes through _stub_send
        try:
            import torch
            tensor = torch.randn(10, 10)
            result = t.send_tensor(tensor, dst_node="peer", dst_gpu=0, layer_id=3)
            assert result["method"] == "stub"
            assert t._stats["tensors_sent"] == 1
            assert t._stats["bytes_sent"] > 0
        except ImportError:
            # No torch — test with raw bytes
            result = t._stub_send(b"x" * 100, "peer", 0, 3)
            assert result["method"] == "stub"

    def test_stats(self):
        """Stats returns correct structure."""
        t = LLMTransport(node_id="test")
        s = t.stats()
        assert "tier" in s
        assert s["tier"] == TransportTier.STUB.name
        assert "rdma_available" in s
        assert "gpudirect_available" in s
        assert "connections" in s
        assert "tensors_sent" in s

    def test_get_local_info(self):
        """Local info returns VTPConnectionInfo."""
        t = LLMTransport(node_id="my-node")
        info = t.get_local_info()
        assert isinstance(info, VTPConnectionInfo)
        assert info.node_id == "my-node"
        assert info.vtp_version == VTP_VERSION

    def test_close_is_safe(self):
        """Close doesn't crash even on empty transport."""
        t = LLMTransport(node_id="test")
        t.close()  # Should not raise

    def test_kv_cache_stream_stub(self):
        """KV cache streaming in stub mode."""
        t = LLMTransport(node_id="test")
        try:
            import torch
            k = torch.randn(4, 2, 8, 16, 64)
            v = torch.randn(4, 2, 8, 16, 64)
            results = t.stream_kv_cache(k, v, dst_node="peer",
                                        layer_ids=[0, 2])
            assert len(results) == 4  # 2 layers × (k + v)
            assert all(r["method"] == "stub" for r in results)
        except ImportError:
            pytest.skip("torch not available")


# ═══════════════════════════════════════════════════════════════════════════
# Transport tier selection tests
# ═══════════════════════════════════════════════════════════════════════════

class TestTierSelection:
    """Test adaptive transport tier selection logic."""

    def test_small_tensor_prefers_rdma_if_available(self):
        tier = select_optimal_tier(100, has_rdma=True)
        assert tier == TransportTier.CPU_STAGED_RDMA

    def test_small_tensor_falls_back_tcp(self):
        tier = select_optimal_tier(100, has_rdma=False)
        assert tier == TransportTier.ZEROCOPY_TCP

    def test_large_tensor_with_gpudirect(self):
        tier = select_optimal_tier(1024 * 1024, has_rdma=True,
                                   has_gpudirect=True, has_cuda=True)
        assert tier == TransportTier.GPUDIRECT_RDMA

    def test_large_tensor_rdma_no_gpudirect(self):
        tier = select_optimal_tier(1024 * 1024, has_rdma=True,
                                   has_gpudirect=False)
        assert tier == TransportTier.CPU_STAGED_RDMA

    def test_no_hardware_falls_back_tcp(self):
        tier = select_optimal_tier(1024 * 1024, has_rdma=False)
        assert tier == TransportTier.ZEROCOPY_TCP

    def test_medium_tensor_rdma(self):
        tier = select_optimal_tier(8192, has_rdma=True)
        assert tier == TransportTier.CPU_STAGED_RDMA


# ═══════════════════════════════════════════════════════════════════════════
# VTPConnectionInfo tests
# ═══════════════════════════════════════════════════════════════════════════

class TestVTPConnectionInfo:
    """Test connection info data structure."""

    def test_default_values(self):
        info = VTPConnectionInfo()
        assert info.node_id == ""
        assert info.qp_num == 0
        assert info.vtp_version == VTP_VERSION
        assert info.gpu_count == 0
        assert info.gpu_regions == {}

    def test_with_gpu_regions(self):
        info = VTPConnectionInfo(
            node_id="node-1",
            gpu_regions={0: {"rkey": 42, "addr": 0x1000, "size": 256 * 1024 * 1024}},
        )
        assert info.gpu_count == 0  # Not auto-computed in __init__
        assert 0 in info.gpu_regions
        assert info.gpu_regions[0]["rkey"] == 42


# ═══════════════════════════════════════════════════════════════════════════
# VTPConnection tests (no RDMA hardware)
# ═══════════════════════════════════════════════════════════════════════════

class TestVTPConnection:
    """Test VTPConnection without RDMA hardware."""

    def test_create_without_rdma(self):
        """Connection creates safely without RDMA hardware."""
        conn = VTPConnection()
        # May or may not have RDMA depending on environment

    def test_detect_tier_no_hardware(self):
        """Tier detection without hardware returns BASIC_TCP or higher."""
        conn = VTPConnection()
        tier = conn._detect_tier()
        assert tier in (TransportTier.GPUDIRECT_RDMA,
                        TransportTier.CPU_STAGED_RDMA,
                        TransportTier.BASIC_TCP)

    def test_close_is_safe(self):
        conn = VTPConnection()
        conn.close()  # Should not raise


# ═══════════════════════════════════════════════════════════════════════════
# GPUMemoryRegion tests (no GPU)
# ═══════════════════════════════════════════════════════════════════════════

class TestGPUMemoryRegion:
    """Test GPUMemoryRegion without actual GPU."""

    def test_create_region(self):
        region = GPUMemoryRegion(gpu_id=0, size_bytes=1024)
        assert region.gpu_id == 0
        assert region.size_bytes == 1024
        assert not region.registered

    def test_register_without_cuda(self):
        """Registration fails gracefully without CUDA."""
        region = GPUMemoryRegion(gpu_id=0, size_bytes=1024)
        # Without CUDA/GPUDIRECT/PYVERBS, should return False
        # (or True if CUDA happens to be available in CI)
        result = region.allocate_and_register()
        # Just verify it doesn't crash

    def test_close_is_safe(self):
        region = GPUMemoryRegion(gpu_id=0, size_bytes=1024)
        region.close()
        assert not region.registered


class TestDoubleBufferedRegion:
    """Test double-buffered regions."""

    def test_create_and_swap(self):
        db = DoubleBufferedRegion(gpu_id=0, size_bytes=1024)
        assert db._active == 0
        assert db.active is db.regions[0]
        assert db.staging is db.regions[1]

        db.swap()
        assert db._active == 1
        assert db.active is db.regions[1]
        assert db.staging is db.regions[0]

        db.swap()
        assert db._active == 0

    def test_close_is_safe(self):
        db = DoubleBufferedRegion(gpu_id=0, size_bytes=1024)
        db.close()


# ═══════════════════════════════════════════════════════════════════════════
# Upgraded module tests
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeEngineUpgrade:
    """Test ComputeEngine production upgrades."""

    def test_stats(self):
        from core.compute_engine import ComputeEngine
        engine = ComputeEngine(backend="cpu", verbose=False)
        s = engine.stats()
        assert "backend" in s
        assert s["backend"] == "cpu"
        assert s["executions"] == 0
        assert s["errors"] == 0

    def test_repr(self):
        from core.compute_engine import ComputeEngine
        engine = ComputeEngine(backend="cpu", verbose=False)
        r = repr(engine)
        assert "ComputeEngine" in r
        assert "cpu" in r


class TestBlockRouterUpgrade:
    """Test BlockRouter production upgrades."""

    def test_stats(self):
        from core.block_router import BlockRouter
        router = BlockRouter(verbose=False)
        s = router.stats
        assert "gpu_routes" in s
        assert "cpu_routes" in s
        assert "errors" in s
        assert "total_routes" in s

    def test_repr(self):
        from core.block_router import BlockRouter
        router = BlockRouter(verbose=False)
        r = repr(router)
        assert "BlockRouter" in r

    def test_thread_safe_register(self):
        from core.block_router import BlockRouter
        router = BlockRouter(verbose=False)
        import threading
        errors = []
        def register():
            try:
                router.register_remote_node("host", 9000, 100)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=register) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestTransportFactoryUpgrade:
    """Test TransportFactory production upgrades."""

    def test_summary_with_vtp(self):
        from core.transport_factory import TransportFactory
        factory = TransportFactory()
        s = factory.summary()
        assert "node_id" in s
        assert "total_transfers" in s
        assert "errors" in s

    def test_vtp_transport_init(self):
        from core.transport_factory import TransportFactory
        factory = TransportFactory()
        vtp = factory.get_vtp_transport()
        # VTP may or may not init depending on environment
        # Just ensure no crash


class TestStreamManagerUpgrade:
    """Test StreamManager production upgrades."""

    def test_stats_includes_errors(self):
        from core.stream_manager import StreamManager
        sm = StreamManager(verbose=False)
        assert "errors" in sm._stats

    def test_monitor_interruptible_stop(self):
        """Monitor loop stops cleanly via Event."""
        from core.stream_manager import StreamManager
        sm = StreamManager(verbose=False)
        sm.start_monitoring(interval=0.1)
        import time
        time.sleep(0.2)
        sm.stop_monitoring()
        assert not sm._monitoring
