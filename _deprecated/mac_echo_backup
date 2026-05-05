#!/usr/bin/env python3
"""VTP Echo Worker — standalone, zero dependencies (Python 3.10+ stdlib only).

Run on any machine (Mac, Windows, Linux) to benchmark VTP transport:
    python3 vtp_echo_worker.py [--port 18951] [--host 0.0.0.0]

Implements the VTP binary protocol echo mode:
  Request:  VTP1(4B) | start_layer(H) | end_layer(H) | seq_len(I) |
            ndim(B) | dtype(B) | shape(I*ndim) | payload_len(I) | raw_bytes
  Response: VTP1(4B) | ndim(B) | dtype(B) | shape(I*ndim) | payload_len(I) | raw_bytes
"""

import socket
import struct
import threading
import argparse
import time
import sys

VTP_MAGIC = b"VTP1"


def recv_exact(sock, n):
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(min(remaining, 1048576))
        if not chunk:
            raise ConnectionError("closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def handle_conn(conn, addr):
    print(f"[+] Connection from {addr}")
    total_requests = 0
    total_bytes = 0
    t0 = time.monotonic()
    try:
        while True:
            # Read magic
            try:
                magic = recv_exact(conn, 4)
            except ConnectionError:
                break
            if magic != VTP_MAGIC:
                print(f"[!] Bad magic from {addr}: {magic!r}")
                break

            # Header: start_layer(H) + end_layer(H) + seq_len(I) + ndim(B) + dtype(B) = 10 bytes
            hdr = recv_exact(conn, 10)
            _start, _end, _seq, ndim, dtype_code = struct.unpack("!HHIBB", hdr)

            # Shape
            shape_data = recv_exact(conn, ndim * 4)
            shape = struct.unpack(f"!{ndim}I", shape_data)

            # Payload length + payload
            plen_data = recv_exact(conn, 4)
            payload_len = struct.unpack("!I", plen_data)[0]
            payload = recv_exact(conn, payload_len)

            total_requests += 1
            total_bytes += payload_len

            # Echo response (no model = echo mode)
            resp = VTP_MAGIC
            resp += struct.pack("!BB", ndim, dtype_code)
            resp += struct.pack(f"!{ndim}I", *shape)
            resp += struct.pack("!I", payload_len)
            conn.sendall(resp)
            conn.sendall(payload)

    except Exception as e:
        print(f"[!] Error from {addr}: {e}")
    finally:
        elapsed = time.monotonic() - t0
        if total_requests > 0:
            mb = total_bytes / (1024 * 1024)
            print(f"[-] {addr} closed — {total_requests} requests, "
                  f"{mb:.1f} MB in {elapsed:.1f}s")
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="VTP Echo Worker (standalone)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18951)
    args = parser.parse_args()

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    except Exception:
        pass
    srv.bind((args.host, args.port))
    srv.listen(8)

    print(f"=== VTP Echo Worker ===")
    print(f"Listening on {args.host}:{args.port}")
    print(f"Platform: {sys.platform}")
    print(f"No dependencies needed — pure echo mode")
    print(f"Press Ctrl+C to stop\n")

    try:
        while True:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            try:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
            except Exception:
                pass
            threading.Thread(target=handle_conn, args=(conn, addr), daemon=True).start()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        srv.close()


if __name__ == "__main__":
    main()
