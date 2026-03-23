"""Tests for TODO features: GPU ordering, asymmetric split, private groups,
K-replication, battery-aware scheduling, WoI enhancements."""

import os
import sys
import tempfile
import threading

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_API_TOKEN", "testtoken")
os.environ.setdefault("VRM_DISABLE_RATE_LIMIT", "1")

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# GPU Ordering & Asymmetric Load Balancing  (model_splitter)
# ═══════════════════════════════════════════════════════════════════════════

class TestGPURanking:
    def test_rank_gpus_default(self):
        """rank_gpus returns indices for 0 GPUs without crash."""
        from core.model_splitter import rank_gpus
        assert rank_gpus(0) == []

    def test_rank_gpus_env_override(self, monkeypatch):
        """VRM_GPU_ORDER forces explicit ordering."""
        from core.model_splitter import rank_gpus
        monkeypatch.setenv("VRM_GPU_ORDER", "2,0,1")
        assert rank_gpus(3) == [2, 0, 1]

    def test_rank_gpus_env_override_wrong_count(self, monkeypatch):
        """VRM_GPU_ORDER with wrong count falls back to auto."""
        from core.model_splitter import rank_gpus
        monkeypatch.setenv("VRM_GPU_ORDER", "0,1")
        result = rank_gpus(3)
        assert len(result) == 3

    def test_compute_scores_returns_list(self):
        """_get_gpu_compute_scores returns a list of floats with len == num_gpus."""
        from core.model_splitter import _get_gpu_compute_scores
        scores = _get_gpu_compute_scores(4)
        assert len(scores) == 4
        assert all(isinstance(s, float) for s in scores)
        assert all(s > 0 for s in scores)


class TestAsymmetricSplit:
    def _make_layers(self, n):
        """Create n fake nn.Module-like objects."""
        import types
        return [types.SimpleNamespace(name=f"layer_{i}") for i in range(n)]

    def test_split_by_vram_equal(self):
        """Equal VRAM -> roughly equal layers."""
        from core.model_splitter import _split_by_vram
        try:
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not available")
        layers = [nn.Identity() for _ in range(20)]
        blocks = _split_by_vram(layers, [8000, 8000])
        assert len(blocks) == 2
        total = sum(len(list(b.children())) for b in blocks)
        assert total == 20

    def test_split_by_vram_asymmetric_compute(self):
        """Higher compute score -> more layers."""
        from core.model_splitter import _split_by_vram
        try:
            import torch.nn as nn
        except ImportError:
            pytest.skip("torch not available")
        layers = [nn.Identity() for _ in range(20)]
        # Same VRAM, but GPU 0 has 3x compute score
        blocks = _split_by_vram(layers, [8000, 8000], compute_scores=[3.0, 1.0])
        assert len(blocks) == 2
        g0_layers = len(list(blocks[0].children()))
        g1_layers = len(list(blocks[1].children()))
        assert g0_layers > g1_layers  # GPU 0 should get more layers


# ═══════════════════════════════════════════════════════════════════════════
# Swarm Private Groups  (swarm_ledger)
# ═══════════════════════════════════════════════════════════════════════════

class TestSwarmPrivateGroups:
    @pytest.fixture(autouse=True)
    def _tmp_ledger(self, tmp_path, monkeypatch):
        """Use a temporary SQLite database for each test."""
        db_path = str(tmp_path / "test_ledger.db")
        monkeypatch.setattr("core.swarm_ledger._LEDGER_PATH", db_path)
        # Re-instantiate the ledger so it picks up the new path
        from core.swarm_ledger import SwarmLedger
        self.ledger = SwarmLedger()

    def test_create_group(self):
        uid, _ = self.ledger.create_user("alice")
        gid, token = self.ledger.create_group("Team A", uid)
        assert gid
        assert token.startswith("grp-")

    def test_join_group(self):
        uid1, _ = self.ledger.create_user("bob")
        uid2, _ = self.ledger.create_user("carol")
        gid, token = self.ledger.create_group("Team B", uid1)

        joined_gid = self.ledger.join_group(token, uid2)
        assert joined_gid == gid

    def test_is_member(self):
        uid, _ = self.ledger.create_user("dave")
        gid, token = self.ledger.create_group("Team C", uid)
        assert self.ledger.is_group_member(gid, uid)

    def test_non_member(self):
        uid1, _ = self.ledger.create_user("eve")
        uid2, _ = self.ledger.create_user("frank")
        gid, _ = self.ledger.create_group("Team D", uid1)
        assert not self.ledger.is_group_member(gid, uid2)

    def test_invalid_token(self):
        uid, _ = self.ledger.create_user("grace")
        result = self.ledger.join_group("grp-invalid-token", uid)
        assert result is None

    def test_max_members(self):
        uid1, _ = self.ledger.create_user("owner")
        gid, token = self.ledger.create_group("Tiny", uid1, max_members=2)
        uid2, _ = self.ledger.create_user("member2")
        assert self.ledger.join_group(token, uid2) == gid
        uid3, _ = self.ledger.create_user("member3")
        assert self.ledger.join_group(token, uid3) is None  # Full

    def test_list_members(self):
        uid1, _ = self.ledger.create_user("owner2")
        uid2, _ = self.ledger.create_user("member")
        gid, token = self.ledger.create_group("Team E", uid1)
        self.ledger.join_group(token, uid2)
        members = self.ledger.get_group_members(gid)
        assert len(members) == 2
        aliases = {m["alias"] for m in members}
        assert "owner2" in aliases
        assert "member" in aliases

    def test_user_groups(self):
        uid, _ = self.ledger.create_user("multi")
        gid1, _ = self.ledger.create_group("G1", uid)
        gid2, _ = self.ledger.create_group("G2", uid)
        groups = self.ledger.get_user_groups(uid)
        assert len(groups) == 2

    def test_validate_token(self):
        uid, _ = self.ledger.create_user("val")
        gid, token = self.ledger.create_group("VG", uid)
        assert self.ledger.validate_group_token(token) == gid
        assert self.ledger.validate_group_token("grp-bogus") is None


# ═══════════════════════════════════════════════════════════════════════════
# K-Replication  (gpu_fault_tolerance)
# ═══════════════════════════════════════════════════════════════════════════

class TestBlockReplicator:
    def test_init(self):
        from core.gpu_fault_tolerance import BlockReplicator
        br = BlockReplicator(k=2)
        assert br.k == 2

    def test_no_replication_k1(self):
        from core.gpu_fault_tolerance import BlockReplicator
        br = BlockReplicator(k=1)
        count = br.replicate(block_id=0, tensor=None, primary_gpu=0)
        assert count == 0

    def test_replicate_cpu(self):
        from core.gpu_fault_tolerance import BlockReplicator
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        br = BlockReplicator(k=2, prefer_cpu=True)
        tensor = torch.randn(4, 4)
        count = br.replicate(block_id=42, tensor=tensor, primary_gpu=0)
        assert count >= 1
        assert br.has_replica(42)

    def test_get_replica(self):
        from core.gpu_fault_tolerance import BlockReplicator
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        br = BlockReplicator(k=2, prefer_cpu=True)
        tensor = torch.randn(4, 4)
        br.replicate(block_id=7, tensor=tensor, primary_gpu=0)
        replica = br.get_replica(7, target_device="cpu")
        assert replica is not None
        assert replica.shape == (4, 4)

    def test_drop_replicas(self):
        from core.gpu_fault_tolerance import BlockReplicator
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        br = BlockReplicator(k=2, prefer_cpu=True)
        br.replicate(block_id=99, tensor=torch.randn(2, 2), primary_gpu=0)
        assert br.has_replica(99)
        br.drop_replicas(99)
        assert not br.has_replica(99)

    def test_checkpoint(self):
        from core.gpu_fault_tolerance import BlockReplicator
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")
        br = BlockReplicator(k=2)
        hidden = torch.randn(1, 16)
        br.checkpoint("infer-001", layer_idx=5, hidden_state=hidden)
        cp = br.get_checkpoint("infer-001")
        assert cp is not None
        assert cp["layer_idx"] == 5
        br.clear_checkpoint("infer-001")
        assert br.get_checkpoint("infer-001") is None

    def test_stats(self):
        from core.gpu_fault_tolerance import BlockReplicator
        br = BlockReplicator(k=3)
        s = br.stats()
        assert s["k"] == 3
        assert s["replicated_blocks"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Wake On Inference
# ═══════════════════════════════════════════════════════════════════════════

class TestWakeOnInference:
    def test_register_and_unregister(self):
        from core.wake_on_inference import WakeOnInferenceManager
        woi = WakeOnInferenceManager()
        woi.register_node("AA:BB:CC:DD:EE:FF")
        assert len(woi.known_macs) == 1
        woi.unregister_node("AA:BB:CC:DD:EE:FF")
        assert len(woi.known_macs) == 0

    def test_wake_all_empty(self):
        from core.wake_on_inference import WakeOnInferenceManager
        woi = WakeOnInferenceManager()
        count = woi.wake_all()
        assert count == 0

    def test_stats(self):
        from core.wake_on_inference import WakeOnInferenceManager
        woi = WakeOnInferenceManager()
        woi.register_node("11:22:33:44:55:66")
        s = woi.stats
        assert s["registered_macs"] == 1

    def test_singleton(self):
        from core.wake_on_inference import get_woi_manager
        m1 = get_woi_manager()
        m2 = get_woi_manager()
        assert m1 is m2


# ═══════════════════════════════════════════════════════════════════════════
# Battery-aware scheduling (WebGPU node)
# ═══════════════════════════════════════════════════════════════════════════

class TestBatteryAwareScoring:
    """Test the get_gpu_score logic extracted from WebGPU node manager."""

    def _score(self, client_data):
        """Reproduce the scoring logic from webgpu_node._task_dispatcher."""
        hw = client_data.get("hw_specs", {})
        battery = hw.get("battery", 100)
        is_charging = hw.get("charging", True)
        is_edge = client_data.get("is_edge", False)

        if is_edge:
            if battery < 15 and not is_charging:
                return -1
            if battery < 30 and not is_charging:
                return 1
            if battery < 50:
                return 3
            return 5
        gpu_name = client_data.get("gpu", "").lower()
        vram = hw.get("vram_gb", 0)
        if "5090" in gpu_name or "5080" in gpu_name or "5070 ti" in gpu_name:
            return 120
        if "rtx" in gpu_name or "m3 max" in gpu_name or "m4 max" in gpu_name:
            return 100
        if "m2" in gpu_name or "m1 max" in gpu_name or "rx 7900" in gpu_name or "m3 pro" in gpu_name:
            return 80
        if "m1" in gpu_name or "gtx 1080" in gpu_name:
            return 50
        if vram >= 24:
            return 90
        if vram >= 12:
            return 60
        return 10

    def test_edge_dead_battery_excluded(self):
        c = {"is_edge": True, "hw_specs": {"battery": 10, "charging": False}}
        assert self._score(c) == -1

    def test_edge_low_battery_last_resort(self):
        c = {"is_edge": True, "hw_specs": {"battery": 25, "charging": False}}
        assert self._score(c) == 1

    def test_edge_charging_ok(self):
        c = {"is_edge": True, "hw_specs": {"battery": 25, "charging": True}}
        # Charging -> battery threshold doesn't apply for < 30 rule
        # but battery < 50 -> score 3
        assert self._score(c) == 3

    def test_blackwell_highest(self):
        c = {"is_edge": False, "gpu": "NVIDIA RTX 5070 Ti", "hw_specs": {"vram_gb": 16}}
        assert self._score(c) == 120

    def test_rtx_high(self):
        c = {"is_edge": False, "gpu": "NVIDIA RTX 4090", "hw_specs": {"vram_gb": 24}}
        assert self._score(c) == 100

    def test_high_vram_unknown(self):
        c = {"is_edge": False, "gpu": "Unknown GPU", "hw_specs": {"vram_gb": 24}}
        assert self._score(c) == 90

    def test_ordering(self):
        """Blackwell > RTX > high-VRAM > edge > dead-edge."""
        clients = [
            {"is_edge": True, "hw_specs": {"battery": 10, "charging": False}, "gpu": ""},
            {"is_edge": True, "hw_specs": {"battery": 80, "charging": True}, "gpu": ""},
            {"is_edge": False, "gpu": "NVIDIA RTX 4090", "hw_specs": {"vram_gb": 24}},
            {"is_edge": False, "gpu": "NVIDIA RTX 5070 Ti", "hw_specs": {"vram_gb": 16}},
        ]
        scores = [self._score(c) for c in clients]
        # Blackwell(120) > RTX(100) > edge-ok(5) > dead(-1)
        sorted_scores = sorted(scores, reverse=True)
        assert sorted_scores == [120, 100, 5, -1]
