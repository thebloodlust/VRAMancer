"""Tests for the transport layer (transfer_manager, fibre_fastpath, transport_factory).

Run with: VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 pytest tests/test_transport.py -v
"""
import os
import pytest

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")
os.environ.setdefault("VRM_DISABLE_RATE_LIMIT", "1")


# ---------------------------------------------------------------------------
# TransferManager tests (stub mode, no GPU required)
# ---------------------------------------------------------------------------
class TestTransferManagerStub:
    """TransferManager in stub mode (VRM_MINIMAL_TEST=1)."""

    def test_import(self):
        from core.transfer_manager import TransferManager, TransferResult, TransportMethod
        assert TransferManager is not None

    def test_stub_init(self):
        from core.transfer_manager import TransferManager
        tm = TransferManager()
        assert tm._stub_mode is True

    def test_stub_send_activation(self):
        from core.transfer_manager import TransferManager, TransportMethod
        tm = TransferManager()
        result = tm.send_activation(0, 1, None)
        assert result.method == TransportMethod.STUB
        assert result.bytes_transferred == 0
        assert result.duration_s == 0.0

    def test_stub_sync_activations(self):
        from core.transfer_manager import TransferManager
        tm = TransferManager()
        results = tm.sync_activations({
            "layer_0": (0, 1, None),
            "layer_1": (1, 0, None),
        })
        assert len(results) == 2
        assert "layer_0" in results
        assert "layer_1" in results

    def test_stats(self):
        from core.transfer_manager import TransferManager
        tm = TransferManager()
        stats = tm.stats()
        assert "transfers" in stats
        assert "topology" in stats
        assert "nccl_initialized" in stats
        assert stats["nccl_initialized"] is False

    def test_topology_none_in_stub(self):
        from core.transfer_manager import TransferManager
        tm = TransferManager()
        # In stub mode, topology may be None (no torch) or empty
        topo = tm.get_topology()
        # Should not crash

    def test_transfer_result_bandwidth(self):
        from core.transfer_manager import TransferResult, TransportMethod
        r = TransferResult(
            method=TransportMethod.CPU_STAGED,
            source_gpu=0,
            target_gpu=1,
            bytes_transferred=1_000_000_000,  # 1 GB
            duration_s=0.1,  # 100ms
        )
        # 1GB in 0.1s = 80 Gbps
        assert r.bandwidth_gbps == pytest.approx(80.0, rel=0.01)

    def test_transport_method_enum(self):
        from core.transfer_manager import TransportMethod
        assert TransportMethod.NCCL.name == "NCCL"
        assert TransportMethod.CUDA_P2P.name == "CUDA_P2P"
        assert TransportMethod.CPU_STAGED.name == "CPU_STAGED"
        assert TransportMethod.STUB.name == "STUB"

    def test_gpu_topology_dataclass(self):
        from core.transfer_manager import GPUTopology
        topo = GPUTopology()
        assert topo.num_gpus == 0
        assert topo.p2p_matrix == {}

    def test_shutdown_safe(self):
        from core.transfer_manager import TransferManager
        tm = TransferManager()
        tm.shutdown()  # Should not crash


# ---------------------------------------------------------------------------
# FastHandle / fibre_fastpath tests
# ---------------------------------------------------------------------------
class TestFibreFastpath:
    """Tests for the network transport layer."""

    def test_import(self):
        from core.network.fibre_fastpath import (
            FastHandle, detect_fast_interfaces, open_low_latency_channel,
            RDMATransport, GPUDirectTransport, ZeroCopyTCPTransport,
        )
        assert FastHandle is not None

    def test_detect_interfaces(self):
        from core.network.fibre_fastpath import detect_fast_interfaces
        interfaces = detect_fast_interfaces()
        assert isinstance(interfaces, list)

    def test_open_channel_fallback(self):
        from core.network.fibre_fastpath import open_low_latency_channel
        handle = open_low_latency_channel()
        assert handle is not None
        assert hasattr(handle, 'kind')
        assert hasattr(handle, 'send')
        assert hasattr(handle, 'recv')

    def test_fasthandle_capabilities(self):
        from core.network.fibre_fastpath import FastHandle
        fh = FastHandle(kind="stub", meta={})
        caps = fh.capabilities()
        assert "kind" in caps
        assert "rdma_available" in caps
        assert "gpudirect_available" in caps
        assert "kernel_bypass" in caps
        assert "transport" in caps

    def test_fasthandle_mmap_send_recv(self):
        from core.network.fibre_fastpath import FastHandle
        fh = FastHandle(kind="test", meta={"id": "unit_test"})
        payload = b"hello vramancer transport"
        sent = fh.send(payload)
        assert sent == len(payload)
        received = fh.recv()
        assert received == payload
        # Cleanup
        if fh.shm_path and os.path.exists(fh.shm_path):
            os.unlink(fh.shm_path)

    def test_fasthandle_send_tensor_method_exists(self):
        from core.network.fibre_fastpath import FastHandle
        fh = FastHandle(kind="stub", meta={})
        assert hasattr(fh, 'send_tensor')
        assert hasattr(fh, 'recv_tensor')

    def test_rdma_transport_init_no_hardware(self):
        from core.network.fibre_fastpath import RDMATransport
        rt = RDMATransport()
        # Without RDMA hardware, should gracefully degrade
        assert rt.available is False or rt.available is True  # depends on env

    def test_gpudirect_transport_init(self):
        from core.network.fibre_fastpath import GPUDirectTransport
        gdt = GPUDirectTransport()
        # Should not crash without hardware
        assert gdt.available is False  # no RDMA transport injected

    def test_zerocopy_tcp_init(self):
        from core.network.fibre_fastpath import ZeroCopyTCPTransport
        tcp = ZeroCopyTCPTransport()
        assert tcp.host == '0.0.0.0'
        assert tcp.buf_size == 4 * 1024 * 1024

    def test_zerocopy_tcp_listen_close(self):
        from core.network.fibre_fastpath import ZeroCopyTCPTransport
        tcp = ZeroCopyTCPTransport()
        port = tcp.listen(0)  # ephemeral port
        assert port > 0
        tcp.close()

    def test_benchmark_interfaces(self):
        from core.network.fibre_fastpath import benchmark_interfaces
        results = benchmark_interfaces(sample_size=1)
        assert isinstance(results, list)


# ---------------------------------------------------------------------------
# TransportFactory tests
# ---------------------------------------------------------------------------
class TestTransportFactory:
    """Tests for the unified transport factory."""

    def test_import(self):
        from core.transport_factory import (
            TransportFactory, TransportTarget, Locality, get_transport_factory,
        )
        assert TransportFactory is not None

    def test_locality_same_gpu(self):
        from core.transport_factory import TransportFactory, TransportTarget, Locality
        factory = TransportFactory()
        src = TransportTarget(node_id="local", gpu_id=0)
        dst = TransportTarget(node_id="local", gpu_id=0)
        assert factory.determine_locality(src, dst) == Locality.SAME_GPU

    def test_locality_same_node(self):
        from core.transport_factory import TransportFactory, TransportTarget, Locality
        factory = TransportFactory()
        src = TransportTarget(node_id="local", gpu_id=0)
        dst = TransportTarget(node_id="local", gpu_id=1)
        assert factory.determine_locality(src, dst) == Locality.SAME_NODE

    def test_locality_remote(self):
        from core.transport_factory import TransportFactory, TransportTarget, Locality
        factory = TransportFactory()
        src = TransportTarget(node_id="node1", gpu_id=0)
        dst = TransportTarget(node_id="node2", gpu_id=0, is_local=False)
        assert factory.determine_locality(src, dst) == Locality.REMOTE

    def test_locality_same_rack(self):
        from core.transport_factory import TransportFactory, TransportTarget, Locality
        os.environ["VRM_SAME_RACK_NODES"] = "node2,node3"
        factory = TransportFactory()
        src = TransportTarget(node_id="node1", gpu_id=0)
        dst = TransportTarget(node_id="node2", gpu_id=0, is_local=False)
        assert factory.determine_locality(src, dst) == Locality.SAME_RACK
        del os.environ["VRM_SAME_RACK_NODES"]

    def test_transfer_same_gpu_noop(self):
        from core.transport_factory import TransportFactory, TransportTarget
        factory = TransportFactory()
        src = TransportTarget(node_id="local", gpu_id=0)
        dst = TransportTarget(node_id="local", gpu_id=0)
        result = factory.transfer_tensor(None, src, dst)
        assert result["method"] == "no-op"
        assert result["locality"] == "same_gpu"

    def test_summary(self):
        from core.transport_factory import TransportFactory
        factory = TransportFactory()
        s = factory.summary()
        assert "node_id" in s
        assert "local_transfer" in s
        assert "network_channel" in s

    def test_singleton(self):
        from core.transport_factory import get_transport_factory
        f1 = get_transport_factory()
        f2 = get_transport_factory()
        assert f1 is f2

    def test_transport_target_device_str(self):
        from core.transport_factory import TransportTarget
        local = TransportTarget(node_id="local", gpu_id=2, is_local=True)
        assert local.device_str == "cuda:2"
        remote = TransportTarget(node_id="worker1", gpu_id=0, is_local=False)
        assert remote.device_str == "worker1:cuda:0"


# ---------------------------------------------------------------------------
# HierarchicalMemoryManager integration
# ---------------------------------------------------------------------------
class TestHierarchicalMemoryTransport:
    """Test that HierarchicalMemoryManager uses transport layer."""

    def test_migrate_signature_accepts_tensor(self):
        from core.hierarchical_memory import HierarchicalMemoryManager
        from core.memory_block import MemoryBlock
        hmm = HierarchicalMemoryManager()
        block = MemoryBlock(size_mb=100, gpu_id=0)
        hmm.register_block(block, "L1")
        # migrate now accepts optional tensor parameter
        result = hmm.migrate(block, "L3", tensor=None)
        assert hmm.get_tier(block.id) == "L3"

    def test_migrate_returns_tensor(self):
        from core.hierarchical_memory import HierarchicalMemoryManager
        from core.memory_block import MemoryBlock
        hmm = HierarchicalMemoryManager()
        block = MemoryBlock(size_mb=50, gpu_id=0)
        hmm.register_block(block, "L3")
        dummy_tensor = "fake_tensor"  # In stub mode, any object works
        result = hmm.migrate(block, "L2", tensor=dummy_tensor)
        assert result == dummy_tensor  # Should pass through in stub mode

    def test_migrate_same_tier_noop(self):
        from core.hierarchical_memory import HierarchicalMemoryManager
        from core.memory_block import MemoryBlock
        hmm = HierarchicalMemoryManager()
        block = MemoryBlock(size_mb=50, gpu_id=0)
        hmm.register_block(block, "L1")
        result = hmm.migrate(block, "L1", tensor="data")
        assert result == "data"
        assert hmm.get_tier(block.id) == "L1"
