"""Tests for IPv6 Anycast Load Balancer and Network RAID."""
import os
import struct
import time

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_API_TOKEN", "testtoken")

import pytest


# ═══════════════════════════════════════════════════════════════════════
# Anycast Load Balancer tests
# ═══════════════════════════════════════════════════════════════════════

class TestAnycastLoadBalancer:
    """Test anycast load balancer strategies and node management."""

    def test_register_and_list_nodes(self):
        from core.network.anycast_balancer import AnycastLoadBalancer
        lb = AnycastLoadBalancer(strategy="weighted")
        lb.register_node("n1", "fe80::1", port=9100, vram_free=24000)
        lb.register_node("n2", "fe80::2", port=9100, vram_free=16000)
        lb.register_node("n3", "fe80::3", port=9100, vram_free=8000)
        nodes = lb.get_nodes()
        assert len(nodes) == 3
        ids = {n.node_id for n in nodes}
        assert ids == {"n1", "n2", "n3"}

    def test_unregister_node(self):
        from core.network.anycast_balancer import AnycastLoadBalancer
        lb = AnycastLoadBalancer()
        lb.register_node("n1", "fe80::1")
        lb.register_node("n2", "fe80::2")
        lb.unregister_node("n1")
        assert len(lb.get_nodes()) == 1
        assert lb.get_nodes()[0].node_id == "n2"

    def test_select_weighted_returns_node(self):
        from core.network.anycast_balancer import AnycastLoadBalancer
        lb = AnycastLoadBalancer(strategy="weighted")
        lb.register_node("n1", "fe80::1")
        lb.register_node("n2", "fe80::2")
        target = lb.select_target()
        assert target is not None
        assert target.node_id in ("n1", "n2")

    def test_select_least_latency(self):
        from core.network.anycast_balancer import AnycastLoadBalancer
        lb = AnycastLoadBalancer(strategy="least_latency")
        lb.register_node("slow", "fe80::1")
        lb.register_node("fast", "fe80::2")
        # Manually set latencies
        lb._nodes["slow"].latency_ms = 50.0
        lb._nodes["fast"].latency_ms = 2.0
        target = lb.select_target()
        assert target.node_id == "fast"

    def test_select_round_robin(self):
        from core.network.anycast_balancer import AnycastLoadBalancer
        lb = AnycastLoadBalancer(strategy="round_robin")
        lb.register_node("n1", "fe80::1")
        lb.register_node("n2", "fe80::2")
        # Both have strength 1.0, so order is stable
        results = set()
        for _ in range(4):
            t = lb.select_target()
            results.add(t.node_id)
        # Should have hit both nodes
        assert results == {"n1", "n2"}

    def test_no_healthy_nodes_returns_none(self):
        from core.network.anycast_balancer import AnycastLoadBalancer
        lb = AnycastLoadBalancer()
        # No nodes registered
        assert lb.select_target() is None

    def test_unhealthy_node_excluded(self):
        from core.network.anycast_balancer import AnycastLoadBalancer
        lb = AnycastLoadBalancer(strategy="weighted")
        lb.register_node("sick", "fe80::1")
        lb.register_node("healthy", "fe80::2")
        # Kill the sick node's strength
        lb._nodes["sick"].strength = 0.01  # below ANYCAST_MIN_STRENGTH (0.1)
        target = lb.select_target()
        assert target.node_id == "healthy"

    def test_stale_node_excluded(self):
        from core.network.anycast_balancer import AnycastLoadBalancer
        lb = AnycastLoadBalancer()
        lb.register_node("stale", "fe80::1")
        lb.register_node("fresh", "fe80::2")
        # Make stale node old
        lb._nodes["stale"].last_seen = time.time() - 60
        target = lb.select_target()
        assert target.node_id == "fresh"

    def test_select_multiple_targets(self):
        from core.network.anycast_balancer import AnycastLoadBalancer
        lb = AnycastLoadBalancer()
        for i in range(5):
            lb.register_node(f"n{i}", f"fe80::{i+1}")
        targets = lb.select_targets(3)
        assert len(targets) == 3
        # Should be unique
        ids = [t.node_id for t in targets]
        assert len(set(ids)) == 3

    def test_exclude_nodes(self):
        from core.network.anycast_balancer import AnycastLoadBalancer
        lb = AnycastLoadBalancer()
        lb.register_node("n1", "fe80::1")
        lb.register_node("n2", "fe80::2")
        target = lb.select_target(exclude=["n1"])
        assert target.node_id == "n2"

    def test_record_result_feedback(self):
        from core.network.anycast_balancer import AnycastLoadBalancer
        lb = AnycastLoadBalancer()
        lb.register_node("n1", "fe80::1")
        # Should not raise even without connectome
        lb.record_result("n1", success=True)
        lb.record_result("n1", success=False)

    def test_status_report(self):
        from core.network.anycast_balancer import AnycastLoadBalancer
        lb = AnycastLoadBalancer(strategy="least_latency")
        lb.register_node("n1", "fe80::1")
        s = lb.status()
        assert s["strategy"] == "least_latency"
        assert s["total_nodes"] == 1
        assert s["healthy_nodes"] == 1
        assert len(s["nodes"]) == 1

    def test_invalid_strategy_falls_back(self):
        from core.network.anycast_balancer import AnycastLoadBalancer
        lb = AnycastLoadBalancer(strategy="invalid_strategy")
        assert lb.strategy == "weighted"

    def test_update_existing_node(self):
        from core.network.anycast_balancer import AnycastLoadBalancer
        lb = AnycastLoadBalancer()
        lb.register_node("n1", "fe80::1", vram_free=8000)
        lb.register_node("n1", "fe80::1", vram_free=16000)
        assert len(lb.get_nodes()) == 1
        assert lb._nodes["n1"].vram_free == 16000


# ═══════════════════════════════════════════════════════════════════════
# Network RAID tests
# ═══════════════════════════════════════════════════════════════════════

class TestRaidShardHeader:
    """Test RAID shard header packing/unpacking."""

    def test_pack_unpack_roundtrip(self):
        from core.network.network_raid import (
            RaidShardInfo, _pack_shard_header, _unpack_shard_header,
        )
        info = RaidShardInfo(
            raid_id=b"\x01" * 16,
            total_shards=6,
            shard_idx=2,
            data_shards=4,
            parity_shards=2,
            original_size=1024,
        )
        header = _pack_shard_header(info)
        payload = b"hello shard data"
        packet = header + payload

        info2, data2 = _unpack_shard_header(packet)
        assert info2.raid_id == info.raid_id
        assert info2.total_shards == 6
        assert info2.shard_idx == 2
        assert info2.data_shards == 4
        assert info2.parity_shards == 2
        assert info2.original_size == 1024
        assert data2 == payload

    def test_invalid_magic_raises(self):
        from core.network.network_raid import _unpack_shard_header, RAID_SHARD_HEADER_SIZE
        bad = b"XX" + b"\x00" * (RAID_SHARD_HEADER_SIZE - 2) + b"payload"
        with pytest.raises(ValueError, match="magic invalid"):
            _unpack_shard_header(bad)

    def test_too_small_raises(self):
        from core.network.network_raid import _unpack_shard_header
        with pytest.raises(ValueError, match="too small"):
            _unpack_shard_header(b"short")


class TestShardReassembler:
    """Test shard collection and RS reassembly."""

    def test_reassemble_all_data_shards(self):
        from core.network.network_raid import ShardReassembler, RaidShardInfo
        r = ShardReassembler()
        raid_id = b"\xAA" * 16
        original = b"Hello RAID world! " * 10  # 180 bytes

        # Split manually into 4 shards
        shard_size = 45
        shards = [original[i*45:(i+1)*45] for i in range(4)]

        result = None
        for i in range(4):
            info = RaidShardInfo(
                raid_id=raid_id, total_shards=6, shard_idx=i,
                data_shards=4, parity_shards=2, original_size=len(original),
            )
            result = r.add_shard(raid_id, info, shards[i])

        assert result is not None
        assert result == original

    def test_pending_until_enough_shards(self):
        from core.network.network_raid import ShardReassembler, RaidShardInfo
        r = ShardReassembler()
        raid_id = b"\xBB" * 16

        info = RaidShardInfo(
            raid_id=raid_id, total_shards=4, shard_idx=0,
            data_shards=3, parity_shards=1, original_size=100,
        )
        # Only 1 shard — not enough
        result = r.add_shard(raid_id, info, b"\x00" * 34)
        assert result is None
        assert r.pending_count() == 1

    def test_expire_stale(self):
        from core.network.network_raid import ShardReassembler, RaidShardInfo
        r = ShardReassembler(timeout=0.1)
        raid_id = b"\xCC" * 16
        info = RaidShardInfo(
            raid_id=raid_id, total_shards=4, shard_idx=0,
            data_shards=3, parity_shards=1, original_size=100,
        )
        r.add_shard(raid_id, info, b"\x00" * 34)
        assert r.pending_count() == 1

        time.sleep(0.2)
        r.expire_stale()
        assert r.pending_count() == 0


class TestNetworkRAID:
    """Test NetworkRAID encode/decode integration."""

    def test_init_defaults(self):
        from core.network.network_raid import NetworkRAID
        raid = NetworkRAID(data_shards=4, parity_shards=2)
        assert raid.data_shards == 4
        assert raid.parity_shards == 2
        raid.stop()

    def test_stripe_send_no_targets_returns_none(self):
        from core.network.network_raid import NetworkRAID
        raid = NetworkRAID(data_shards=4, parity_shards=2)
        # No target_nodes, no balancer → should return None
        result = raid.stripe_send(b"data" * 100, layer_id=1, target_nodes=[])
        assert result is None
        raid.stop()

    def test_full_encode_decode_cycle(self):
        """Simulate full RAID cycle: encode, transmit shards, reassemble."""
        from core.network.network_raid import (
            NetworkRAID, RaidShardInfo,
            _pack_shard_header, _unpack_shard_header, _make_raid_id,
        )
        from core.network.aitp_fec import FastFEC

        original = os.urandom(1000)
        d_shards = 4
        p_shards = 2

        # Encode
        fec = FastFEC(data_shards=d_shards, parity_shards=p_shards)
        all_shards = fec.encode(original)
        assert len(all_shards) == d_shards + p_shards

        # Wrap with RAID headers
        raid_id = _make_raid_id(original)
        packets = []
        for i, shard_data in enumerate(all_shards):
            info = RaidShardInfo(
                raid_id=raid_id,
                total_shards=d_shards + p_shards,
                shard_idx=i,
                data_shards=d_shards,
                parity_shards=p_shards,
                original_size=len(original),
            )
            packets.append(_pack_shard_header(info) + shard_data)

        # Simulate receiving — drop 2 shards (parity can handle it)
        raid = NetworkRAID(data_shards=d_shards, parity_shards=p_shards)
        received_packets = packets[:2] + packets[4:]  # drop shards 2 and 3

        result = None
        for pkt in received_packets:
            info, payload = _unpack_shard_header(pkt)
            result = raid._reassembler.add_shard(info.raid_id, info, payload)

        assert result is not None
        assert result == original
        raid.stop()

    def test_status(self):
        from core.network.network_raid import NetworkRAID
        raid = NetworkRAID(data_shards=3, parity_shards=1)
        s = raid.status()
        assert s["data_shards"] == 3
        assert s["parity_shards"] == 1
        assert s["pending_reassemblies"] == 0
        raid.stop()

    def test_handle_non_raid_data(self):
        """Non-RAID data should be silently skipped."""
        from core.network.network_raid import NetworkRAID
        raid = NetworkRAID()
        result = raid.handle_incoming_shard(0, b"not a raid shard", 0, None)
        assert result is None
        raid.stop()

    def test_completion_callback(self):
        """Test that completion callback fires on reassembly."""
        from core.network.network_raid import NetworkRAID, RaidShardInfo, _make_raid_id
        import math

        original = b"callback test data " * 5  # 95 bytes
        d_shards = 2
        raid = NetworkRAID(data_shards=d_shards, parity_shards=0)
        callback_results = []
        raid.set_completion_callback(lambda rid, data: callback_results.append(data))

        raid_id = _make_raid_id(original)
        # Split into 2 equal-size shards (same logic as FastFEC/stripe_send)
        shard_size = math.ceil(len(original) / d_shards)
        padded = original.ljust(shard_size * d_shards, b"\x00")
        shards = [padded[i * shard_size:(i + 1) * shard_size] for i in range(d_shards)]

        from core.network.network_raid import _pack_shard_header
        for i, sd in enumerate(shards):
            info = RaidShardInfo(
                raid_id=raid_id, total_shards=d_shards, shard_idx=i,
                data_shards=d_shards, parity_shards=0, original_size=len(original),
            )
            pkt = _pack_shard_header(info) + sd
            raid.handle_incoming_shard(0, pkt, 0, None)

        assert len(callback_results) == 1
        assert callback_results[0] == original
        raid.stop()


class TestAITPProtocolIntegration:
    """Test AITP protocol integration with LB and RAID."""

    def test_send_balanced_no_nodes(self):
        """send_balanced should return False when no nodes available."""
        from core.network.aitp_protocol import AITPProtocol
        proto = AITPProtocol(port=0)
        # Reset global balancer to empty
        import core.network.anycast_balancer as ab
        old = ab._global_balancer
        ab._global_balancer = None
        try:
            result = proto.send_balanced(0, b"test tensor")
            # With no nodes registered, should return False
            assert result is False
        finally:
            ab._global_balancer = old

    def test_send_raid_import_works(self):
        """send_raid method exists and handles no-targets gracefully."""
        from core.network.aitp_protocol import AITPProtocol
        proto = AITPProtocol(port=0)
        # Reset balancer
        import core.network.anycast_balancer as ab
        old = ab._global_balancer
        ab._global_balancer = None
        try:
            result = proto.send_raid(0, b"test", data_shards=2, parity_shards=1)
            # Should not crash — returns True (fallback) or False
            assert isinstance(result, bool)
        finally:
            ab._global_balancer = old
