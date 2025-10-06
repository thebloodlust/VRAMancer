"""Télémetrie compacte type "SNMP minimal" pour VRAMancer.

Objectif: fournir un format extrêmement léger pour diffuser l'état des nœuds
edge / cluster avec overhead minimal.

Format binaire (packet fixe 24 octets + id dynamique):
  struct header: >B I H H H H I
    version (1 byte)
    timestamp (uint32)
    cpu_load_x100 (uint16)  # ex: 5234 => 52.34%
    free_cores (uint16)
    vram_used_mb (uint16)
    vram_total_mb (uint16)
    node_id_hash (uint32)   # FNV32 du node_id (collision tolérée)
  + node_id (utf-8, longueur variable) + b'\0'

Plusieurs packets peuvent être concaténés (stream). Pas de table d'index;
le parseur lit jusqu'au NUL final de chaque id.

Format texte ultra-compact (ligne par nœud):
  id=<id> cl=<cpu_load%> fc=<free_cores> vu=<vram_used_mb> vt=<vram_total_mb> ts=<unix>

Usage:
    from core.telemetry import encode_packet, decode_stream, format_text_line
"""
from __future__ import annotations
import struct, time, hashlib
from typing import Iterable, Dict, Any, Iterator

VERSION = 1
HEADER_FMT = ">B I H H H H I"  # taille fixe 1+4+2+2+2+2+4 = 17 octets

def fnv1a_32(data: bytes) -> int:
    h = 0x811c9dc5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xffffffff
    return h

def encode_packet(node: Dict[str, Any]) -> bytes:
    ts = int(time.time())
    cpu = int((node.get("cpu_load_pct") or 0) * 100)  # en centi-%
    free = int(node.get("free_cores") or 0)
    vu = int(node.get("vram_used_mb") or 0)
    vt = int(node.get("vram_total_mb") or 0)
    node_id = str(node.get("id", "?"))
    hid = fnv1a_32(node_id.encode())
    header = struct.pack(HEADER_FMT, VERSION, ts, cpu, free, vu, vt, hid)
    return header + node_id.encode() + b"\0"

def encode_stream(nodes: Iterable[Dict[str, Any]]) -> bytes:
    return b"".join(encode_packet(n) for n in nodes)

def decode_stream(data: bytes) -> Iterator[Dict[str, Any]]:
    i = 0
    L = len(data)
    while i < L:
        if i + struct.calcsize(HEADER_FMT) > L:
            break
        header = data[i:i+struct.calcsize(HEADER_FMT)]
        version, ts, cpu_x100, free, vu, vt, hid = struct.unpack(HEADER_FMT, header)
        i += struct.calcsize(HEADER_FMT)
        # lire node_id jusqu'au NUL
        end = data.find(b"\0", i)
        if end == -1:
            break
        node_id = data[i:end].decode(errors="replace")
        i = end + 1
        yield {
            "version": version,
            "timestamp": ts,
            "cpu_load_pct": cpu_x100 / 100.0,
            "free_cores": free,
            "vram_used_mb": vu,
            "vram_total_mb": vt,
            "id": node_id,
            "node_id_hash": hid,
        }

def format_text_line(node: Dict[str, Any]) -> str:
    return (
        f"id={node.get('id')} cl={node.get('cpu_load_pct',0):.2f} "
        f"fc={node.get('free_cores',0)} vu={node.get('vram_used_mb',0)} "
        f"vt={node.get('vram_total_mb',0)} ts={int(time.time())}"
    )

__all__ = [
    "encode_packet", "encode_stream", "decode_stream", "format_text_line", "fnv1a_32"
]
