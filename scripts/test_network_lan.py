#!/usr/bin/env python3
"""Test VRAMancer network stack between machines on the same LAN.

Run this script on EACH machine (Mac Mini, MacBook Air, PC portable).
It tests:
  1. Cluster discovery (mDNS + UDP broadcast)
  2. AITP protocol (UDP + HMAC-SHA256 + FEC)
  3. Peer sensing (heartbeat + latency)

Usage:
    # On each machine:
    pip install -e .
    python scripts/test_network_lan.py

    # The script auto-discovers other nodes on the LAN.
    # Run on at least 2 machines simultaneously.
"""

import os
import sys
import time
import socket
import threading
import json

os.environ["VRM_TEST_MODE"] = "1"

def get_local_ip():
    """Get the LAN IP (not 127.0.0.1)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def test_cluster_discovery():
    """Test mDNS + UDP broadcast discovery."""
    print("[1/4] Cluster Discovery")
    print(f"      Local IP: {get_local_ip()}")
    print(f"      Hostname: {socket.gethostname()}")

    try:
        from core.network.cluster_discovery import ClusterDiscovery

        discovery = ClusterDiscovery(port=55555, heartbeat_interval=5.0)
        discovery.start()
        print("      Discovery started (mDNS + UDP broadcast)")
        print("      Waiting 10s for peer discovery...")
        time.sleep(10)
        nodes = discovery.get_nodes()
        discovery.stop()

        if nodes:
            print(f"      Found {len(nodes)} node(s):")
            for n in nodes:
                name = n.get("hostname", n.get("node_id", "?"))
                ip = n.get("ip", "?")
                gpus = n.get("gpu_count", "?")
                print(f"        - {name} ({ip}) — {gpus} GPU(s)")
            print("      PASS\n")
        else:
            print("      No peers found (run on another machine too)")
            print("      SKIP\n")
        return len(nodes) > 0
    except Exception as e:
        print(f"      ERROR: {e}\n")
        return False


def test_aitp_protocol():
    """Test AITP UDP protocol (loopback)."""
    print("[2/4] AITP Protocol (local loopback test)")

    try:
        from core.network.aitp_protocol import AITPProtocol

        proto = AITPProtocol(port=9843)
        test_data = b"Hello from VRAMancer AITP test!" * 10  # ~310 bytes

        # Create a packet and parse it back (round-trip test)
        packet = proto.create_packet(layer_id=42, tensor_bytes=test_data)
        parsed = proto.parse_packet(packet)

        if parsed and parsed["layer_id"] == 42 and parsed["tensor_data"] == test_data:
            print(f"      Packet round-trip OK (layer_id=42, {len(test_data)} bytes)")
            print("      PASS\n")
            return True
        else:
            print("      Packet parse mismatch")
            print("      FAIL\n")
            return False
    except ImportError:
        print("      AITP not importable, testing raw UDP...")
        return test_raw_udp()
    except Exception as e:
        print(f"      ERROR: {e}\n")
        return False


def test_raw_udp():
    """Fallback: raw UDP loopback test."""
    port = 9844
    received = []

    def recv():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(3)
        s.bind(("0.0.0.0", port))
        try:
            data, addr = s.recvfrom(1024)
            received.append(data)
        except socket.timeout:
            pass
        s.close()

    t = threading.Thread(target=recv, daemon=True)
    t.start()
    time.sleep(0.3)

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.sendto(b"VRAMancer UDP test", ("127.0.0.1", port))
    s.close()

    t.join(timeout=4)
    if received:
        print("      UDP loopback OK")
        print("      PASS\n")
    else:
        print("      FAIL\n")
    return len(received) > 0


def test_peer_sensing():
    """Test heartbeat peer sensing."""
    print("[3/4] Peer Sensing (heartbeat)")

    try:
        from core.network.aitp_sensing import AITPSensor

        sensor = AITPSensor(
            node_uid=f"test-{socket.gethostname()}",
            hw_specs={"type": "gpu", "compute": 100, "vram": 8_000_000_000},
        )
        sensor.start_listening()
        print("      Sensor started, broadcasting heartbeat...")

        # Broadcast a few times
        for _ in range(3):
            sensor.broadcast_presence()
            time.sleep(2)

        peers = sensor.get_available_peers()
        sensor.stop()

        print(f"      Alive peers: {len(peers)}")
        for uid, info in peers.items():
            print(f"        - {uid}: {info.get('hw', {}).get('type', '?')}")
        if peers:
            print("      PASS\n")
        else:
            print("      No peers (run on another machine too)")
            print("      SKIP\n")
        return len(peers) > 0
    except Exception as e:
        print(f"      ERROR: {e}\n")
        return False


def test_connectivity():
    """Test basic TCP connectivity to known VRAMancer port."""
    print("[4/4] TCP Connectivity Check")
    local_ip = get_local_ip()
    port = 9742

    # Start a simple TCP listener
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("0.0.0.0", port))
        srv.listen(1)
        srv.settimeout(2)
        print(f"      Listening on {local_ip}:{port}")

        # Self-connect
        cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cli.settimeout(2)
        cli.connect((local_ip, port))
        conn, addr = srv.accept()
        cli.send(b"VRM_PING")
        data = conn.recv(1024)
        cli.close()
        conn.close()

        if data == b"VRM_PING":
            print("      Self-connect OK")
            print("      PASS\n")
            return True
        else:
            print("      FAIL\n")
            return False
    except Exception as e:
        print(f"      ERROR: {e}\n")
        return False
    finally:
        srv.close()


if __name__ == "__main__":
    print("=" * 60)
    print("VRAMancer Network Stack Test — LAN")
    print(f"Host: {socket.gethostname()} ({get_local_ip()})")
    print("=" * 60)
    print()

    results = {}
    results["discovery"] = test_cluster_discovery()
    results["aitp"] = test_aitp_protocol()
    results["sensing"] = test_peer_sensing()
    results["tcp"] = test_connectivity()

    print("=" * 60)
    print("RESULTS:")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'SKIP/FAIL'}")
    passed = sum(1 for v in results.values() if v)
    print(f"\n{passed}/{len(results)} tests passed")
    print("=" * 60)
