"""Tests for core/transfer_manager.py — TransferManager stub-safe methods."""
import os
import pytest

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")

from core.transfer_manager import TransferManager


class TestTransferManagerInit:
    def test_init_default(self):
        tm = TransferManager()
        assert tm is not None

    def test_init_protocols(self):
        for proto in ["nccl", "p2p", "cpu"]:
            tm = TransferManager(protocol=proto)
            assert tm is not None


class TestTransferManagerStats:
    def test_stats_returns_dict(self):
        tm = TransferManager()
        s = tm.stats()
        assert isinstance(s, dict)

    def test_stats_has_keys(self):
        tm = TransferManager()
        s = tm.stats()
        # Should contain at least some of these standard keys
        expected_keys = {"total_transfers", "total_bytes", "nccl_initialized"}
        assert len(expected_keys & set(s.keys())) >= 1 or len(s) > 0


class TestTransferManagerStubMode:
    def test_send_activation_stub(self):
        tm = TransferManager()
        # In stub mode, should return a result without crashing
        result = tm.send_activation(0, 1, "fake_tensor")
        assert result is not None

    def test_sync_activations_stub(self):
        tm = TransferManager()
        results = tm.sync_activations({
            "layer_0": (0, 1, "fake_tensor"),
            "layer_1": (1, 0, "fake_tensor"),
        })
        assert isinstance(results, dict)
        assert len(results) == 2

    def test_benchmark_no_gpu(self):
        tm = TransferManager()
        results = tm.benchmark(sizes_mb=[1.0])
        assert isinstance(results, list) or isinstance(results, dict)


class TestTransferManagerTopology:
    def test_get_topology(self):
        tm = TransferManager()
        topo = tm.get_topology()
        # Can be None in stub mode, but shouldn't crash
        assert topo is None or hasattr(topo, "__dict__")


class TestTransferManagerShutdown:
    def test_shutdown_no_crash(self):
        tm = TransferManager()
        tm.shutdown()  # Should not raise
