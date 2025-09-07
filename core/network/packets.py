import struct
import json

# ------------------------------------------------------------------
# 1️⃣  Format de base des paquets (sans TCP/IP)
# ------------------------------------------------------------------
class Packet:
    """
    Un paquet est composé de :
        - header  (4 octets : longueur du payload en bytes)
        - payload (bytes)
    """

    HEADER_FORMAT = "!I"   # unsigned int, network order

    def __init__(self, payload: bytes):
        self.payload = payload

    def pack(self) -> bytes:
        """Serialize le paquet (header + payload)."""
        header = struct.pack(self.HEADER_FORMAT, len(self.payload))
        return header + self.payload

    @staticmethod
    def unpack(raw: bytes):
        """Inverse de pack – renvoie (payload, rest_of_stream)."""
        header_size = struct.calcsize(Packet.HEADER_FORMAT)
        if len(raw) < header_size:
            return None, raw
        (payload_len,) = struct.unpack(Packet.HEADER_FORMAT, raw[:header_size])
        if len(raw) < header_size + payload_len:
            return None, raw
        payload = raw[header_size:header_size+payload_len]
        rest    = raw[header_size+payload_len:]
        return payload, rest
