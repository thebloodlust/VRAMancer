import hashlib
import struct

class PacketBuilder:
    def __init__(self, secure=True):
        self.secure = secure

    def build_packet(self, source_id, dest_id, data_bytes, packet_type="DATA", flags=0, sync_id=None):
        """
        Construit un paquet VRAMancer : [HEADER][DATA][SYNC][CHECKSUM]
        """
        header = struct.pack(
            ">IIBI",  # source_id, dest_id, type_code, data_size
            source_id,
            dest_id,
            self._type_to_code(packet_type),
            len(data_bytes)
        )
        flag_byte = struct.pack("B", flags)
        payload = header + flag_byte + data_bytes
        if sync_id is not None:
            sync_bytes = struct.pack(">Q", sync_id)  # 8 bytes sync id
            payload += sync_bytes
        checksum = self._checksum(payload)
        return payload + checksum

    def _type_to_code(self, packet_type):
        types = {"DATA": 1, "SYNC": 2, "ACK": 3, "ERROR": 4}
        return types.get(packet_type.upper(), 0)

    def _checksum(self, payload):
        if self.secure:
            return hashlib.sha256(payload).digest()[:8]  # 64-bit digest
        else:
            return struct.pack(">I", sum(payload) % (2**32))  # simple CRC

    def parse_packet(self, packet_bytes):
        """
        Décode un paquet reçu, extrait sync_id si présent.
        """
        header = struct.unpack(">IIBI", packet_bytes[:13])
        flags = packet_bytes[13]
        # Vérifie si sync_id est présent (optionnel)
        if len(packet_bytes) > 22:
            data = packet_bytes[14:-16]
            sync_id = struct.unpack(">Q", packet_bytes[-16:-8])[0]
        else:
            data = packet_bytes[14:-8]
            sync_id = None
        checksum = packet_bytes[-8:]
        return {
            "source_id": header[0],
            "dest_id": header[1],
            "type": header[2],
            "size": header[3],
            "flags": flags,
            "data": data,
            "sync_id": sync_id,
            "checksum": checksum
        }
