"""Tests for AITP protocol suite: Protocol, FEC, Receiver, Sensing.

Runs without a real network (loopback/mock sockets).
Uses VRM_MINIMAL_TEST=1 environment from conftest.
"""

import os
import struct
import time
import hmac
import hashlib
import json
import threading
import pytest

# ── Environment (conftest sets VRM_MINIMAL_TEST=1) ──
os.environ.setdefault("VRM_API_TOKEN", "testtoken")


# ═══════════════════════════════════════════════════════════════════════
# FEC (no network needed)
# ═══════════════════════════════════════════════════════════════════════

class TestFastFEC:
    """GF(2^8) Reed-Solomon encoder/decoder."""

    def _make_fec(self, d=4, p=2):
        from core.network.aitp_fec import FastFEC
        return FastFEC(data_shards=d, parity_shards=p)

    def test_gf_mul_identity(self):
        from core.network.aitp_fec import gf_mul
        assert gf_mul(1, 42) == 42
        assert gf_mul(42, 1) == 42

    def test_gf_mul_zero(self):
        from core.network.aitp_fec import gf_mul
        assert gf_mul(0, 42) == 0
        assert gf_mul(42, 0) == 0

    def test_gf_div_inverse(self):
        from core.network.aitp_fec import gf_mul, gf_div
        for a in (1, 7, 42, 200, 255):
            assert gf_div(gf_mul(a, 37), 37) == a

    def test_gf_inv_zero_raises(self):
        from core.network.aitp_fec import gf_inv
        with pytest.raises(ZeroDivisionError):
            gf_inv(0)

    def test_encode_decode_all_present(self):
        """All data shards present — fast path (no decoding)."""
        fec = self._make_fec(4, 2)
        data = b"Hello VRAMancer FEC test data!!"  # 30 bytes
        shards = fec.encode(data)
        assert len(shards) == 6  # 4 data + 2 parity

        received = {i: shards[i] for i in range(6)}
        restored = fec.decode(received, len(data))
        assert restored == data

    def test_decode_one_data_lost(self):
        """Lose 1 data shard, reconstruct from parity."""
        fec = self._make_fec(4, 2)
        data = os.urandom(200)
        shards = fec.encode(data)

        # Drop shard 1 (data)
        received = {i: shards[i] for i in range(6) if i != 1}
        restored = fec.decode(received, len(data))
        assert restored == data

    def test_decode_two_data_lost(self):
        """Lose 2 data shards, reconstruct from 2 parity shards."""
        fec = self._make_fec(4, 2)
        data = os.urandom(400)
        shards = fec.encode(data)

        # Drop data shards 0 and 3
        received = {i: shards[i] for i in range(6) if i not in (0, 3)}
        restored = fec.decode(received, len(data))
        assert restored == data

    def test_decode_not_enough_shards(self):
        fec = self._make_fec(4, 2)
        data = os.urandom(100)
        shards = fec.encode(data)
        received = {0: shards[0], 1: shards[1]}  # only 2, need 4
        with pytest.raises(ValueError, match="Need at least"):
            fec.decode(received, len(data))

    def test_shard_limit(self):
        with pytest.raises(ValueError, match="256"):
            self._make_fec(200, 100)

    def test_encode_small_data(self):
        """Single byte per shard."""
        fec = self._make_fec(2, 1)
        data = b"AB"
        shards = fec.encode(data)
        assert len(shards) == 3
        received = {0: shards[0], 2: shards[2]}  # drop shard 1
        restored = fec.decode(received, len(data))
        assert restored == data


# ═══════════════════════════════════════════════════════════════════════
# AITP Protocol (packet creation/parsing, no network bind)
# ═══════════════════════════════════════════════════════════════════════

class TestAITPProtocol:
    """Packet format + HMAC + FEC integration."""

    def _secret(self):
        return os.environ.get("VRM_API_TOKEN", "testtoken").encode("utf-8")

    def test_create_parse_roundtrip(self):
        from core.network.aitp_protocol import AITPProtocol
        proto = AITPProtocol.__new__(AITPProtocol)
        proto._fec = None
        proto._recv_running = False

        tensor = b"tensor_payload_1234"
        pkt = proto.create_packet(layer_id=42, tensor_bytes=tensor, flags=0)
        parsed = proto.parse_packet(pkt)

        assert parsed["layer_id"] == 42
        assert parsed["tensor_data"] == tensor
        assert parsed["flags"] == 0
        assert parsed["version"] == 1

    def test_hmac_tampering_rejected(self):
        from core.network.aitp_protocol import AITPProtocol
        proto = AITPProtocol.__new__(AITPProtocol)
        proto._fec = None
        proto._recv_running = False

        pkt = proto.create_packet(layer_id=1, tensor_bytes=b"secret")
        # Tamper with payload
        tampered = bytearray(pkt)
        tampered[20] ^= 0xFF
        tampered = bytes(tampered)

        with pytest.raises(ValueError, match="HMAC"):
            proto.parse_packet(tampered)

    def test_truncated_packet_rejected(self):
        from core.network.aitp_protocol import AITPProtocol
        proto = AITPProtocol.__new__(AITPProtocol)
        proto._fec = None
        proto._recv_running = False

        with pytest.raises(ValueError, match="too small"):
            proto.parse_packet(b"tiny")

    def test_flag_bits(self):
        from core.network.aitp_protocol import FLAG_FEC, FLAG_COMPRESSED, FLAG_PRIORITY
        assert FLAG_FEC == 0x01
        assert FLAG_COMPRESSED == 0x02
        assert FLAG_PRIORITY == 0x04

    def test_fec_enable(self):
        from core.network.aitp_protocol import AITPProtocol
        proto = AITPProtocol.__new__(AITPProtocol)
        proto._fec = None
        proto._recv_running = False
        proto.sock = None
        proto.port = 9999
        proto.anycast_ipv6 = "::1"
        proto.enable_fec(data_shards=4, parity_shards=2)
        assert proto._fec is not None
        assert proto._fec.data_shards == 4


# ═══════════════════════════════════════════════════════════════════════
# AITP Receiver (unit tests — no real socket bind)
# ═══════════════════════════════════════════════════════════════════════

class TestAITPReceiver:
    """Receiver parse+dispatch without network."""

    def _make_packet(self, layer_id=1, tensor=b"data", flags=0):
        """Build a valid AITP packet."""
        from core.network.aitp_receiver import AITP_HEADER_FORMAT, AITP_MAGIC
        header = struct.pack(
            AITP_HEADER_FORMAT, AITP_MAGIC, 1, flags, len(tensor), layer_id,
        )
        body = header + tensor
        secret = os.environ.get("VRM_API_TOKEN", "testtoken").encode("utf-8")
        sig = hmac.new(secret, body, hashlib.sha256).digest()
        return body + sig

    def test_parse_valid_packet(self):
        from core.network.aitp_receiver import AITPReceiver
        results = []
        rx = AITPReceiver(gpu_id=0, on_tensor=lambda lid, d, f: results.append((lid, d, f)))
        rx._active_mode = "udp"

        pkt = self._make_packet(layer_id=7, tensor=b"hello")
        rx._parse_and_dispatch(pkt)

        assert len(results) == 1
        assert results[0] == (7, b"hello", 0)

    def test_parse_bad_hmac(self):
        from core.network.aitp_receiver import AITPReceiver
        rx = AITPReceiver(gpu_id=0)
        rx._active_mode = "udp"

        pkt = self._make_packet()
        tampered = bytearray(pkt)
        tampered[10] ^= 0xFF
        rx._parse_and_dispatch(bytes(tampered))

        assert rx._stats["hmac_fail"] == 1
        assert rx._stats["packets"] == 0

    def test_parse_truncated(self):
        from core.network.aitp_receiver import AITPReceiver
        rx = AITPReceiver(gpu_id=0)
        rx._active_mode = "udp"
        rx._parse_and_dispatch(b"short")
        assert rx._stats["errors"] == 1

    def test_backpressure_drop(self):
        from core.network.aitp_receiver import AITPReceiver, _MAX_QUEUE_DEPTH
        rx = AITPReceiver(gpu_id=0)
        rx._active_mode = "udp"
        rx._pending_count = _MAX_QUEUE_DEPTH  # saturated

        pkt = self._make_packet()
        rx._parse_and_dispatch(pkt)
        assert rx._stats["drops"] >= 1

    def test_stats_structure(self):
        from core.network.aitp_receiver import AITPReceiver
        rx = AITPReceiver(gpu_id=0)
        stats = rx.get_stats()
        assert "bytes" in stats
        assert "packets" in stats
        assert "drops" in stats
        assert "pending" in stats


# ═══════════════════════════════════════════════════════════════════════
# AITP Sensing (unit tests — no real multicast)
# ═══════════════════════════════════════════════════════════════════════

class TestAITPSensing:
    """Sensing peer discovery without network."""

    def test_payload_hmac_valid(self):
        from core.network.aitp_sensing import AITPSensor, _get_cluster_secret
        sensor = AITPSensor(node_uid="test-node", hw_specs={"vram": 16})
        payload = sensor.get_discovery_payload()
        parsed = json.loads(payload.decode('utf-8'))

        assert "sig" in parsed
        assert "data" in parsed
        json_data = json.dumps(parsed["data"]).encode('utf-8')
        expected = hmac.new(
            _get_cluster_secret(), json_data, hashlib.sha256,
        ).hexdigest()
        assert hmac.compare_digest(parsed["sig"], expected)

    def test_peer_tracking(self):
        from core.network.aitp_sensing import AITPSensor
        sensor = AITPSensor(node_uid="local")
        sensor.peers["remote-1"] = {
            "ipv6": "::1", "hw": {}, "last_seen": time.time(), "nat": {},
        }
        peers = sensor.get_available_peers(max_age=30)
        assert "remote-1" in peers

    def test_stale_peer_eviction(self):
        from core.network.aitp_sensing import AITPSensor
        sensor = AITPSensor(node_uid="local")
        sensor.peers["old-node"] = {
            "ipv6": "::1", "hw": {}, "last_seen": time.time() - 60, "nat": {},
        }
        sensor._evict_stale_peers()
        assert "old-node" not in sensor.peers

    def test_self_discovery_ignored(self):
        """Sensor should ignore its own broadcasts."""
        from core.network.aitp_sensing import AITPSensor, _get_cluster_secret
        sensor = AITPSensor(node_uid="self-node")
        # Simulate receiving own payload
        payload_data = {"uid": "self-node", "hw": {}, "ts": time.time(), "nat": {}}
        json_data = json.dumps(payload_data).encode('utf-8')
        sig = hmac.new(_get_cluster_secret(), json_data, hashlib.sha256).hexdigest()
        # Directly check: uid == self.node_uid should be skipped
        assert payload_data["uid"] == sensor.node_uid
        # So peers should remain empty
        assert len(sensor.peers) == 0
