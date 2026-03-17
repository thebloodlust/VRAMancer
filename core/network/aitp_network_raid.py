import logging
import os
import uuid
from .aitp_protocol import get_aitp_protocol
from .aitp_sensing import AITPSensor
from .aitp_fec import FastFEC

logger = logging.getLogger(__name__)

class NetworkRAIDManager:
    """
    Le chef d'orchestre absolu du futur Protocole AITP.
    Combine :
    1. L'instinct de groupe (Sensing Multicast)
    2. La tolérance aux pannes (FEC / Reed-Solomon)
    3. L'expédition Anycast "Fire-and-Forget" (UDP IPv6)
    4. (Optionnel) Hook XDP Kernel Bypass pour les noeuds sous Linux
    """

    def __init__(self, node_hardware: dict):
        self.node_uid = str(uuid.uuid4())
        self.aitp = get_aitp_protocol()
        self.fec = FastFEC(data_shards=8, parity_shards=4) # Résiste à 33% de perte de paquets
        self.sensor = AITPSensor(self.node_uid, node_hardware)
        
        self._try_load_ebpf()

    def _try_load_ebpf(self):
        """Tente de charger le code C (csrc/aitp_xdp_bypass.c) directement dans la carte réseau (NIC)"""
        if os.name != 'posix':
            logger.info("[XDP] Bypass Kernel ignoré : OS non-Linux détecté.")
            return
            
        try:
            # En production, bcc (BPF Compiler Collection) compilerait le fichier .c
            # et l'attacherait à eth0 (ou wlan0).
            # import bcc
            # b = bcc.BPF(src_file="csrc/aitp_xdp_bypass.c")
            # b.attach_xdp(dev="eth0", fn=b.load_func("aitp_direct_to_gpu", bcc.BPF.XDP))
            logger.info("[XDP] Moteur eBPF/XDP simulé chargé : Zero-Copy activé sur le port 9109.")
        except Exception as e:
            logger.warning(f"[XDP] L'injection Kernel a échoué (Droits Root requis?) : {e}")

    def start(self):
        """Démarre le moniteur (Radar IPv6)"""
        self.sensor.start_listening()
        self.sensor.broadcast_presence()
        logger.info(f"🚀 AITP Network RAID Démarré ! UID: {self.node_uid}")

    def dispatch_tensor_raid(self, layer_id: int, tensor_data: bytes, anycast_target="ff02::vrm"):
        """
        La méthode magique.
        Prend un tenseur mathématique massif, le protège, et l'atomise sur le réseau.
        """
        # 1. Protection Mathématique (FEC) - On éclate le Tenseur
        shards = self.fec.encode(tensor_data)
        
        total_shards = len(shards)
        logger.debug(f"[AITP RAID] Dispersion du Layer {layer_id} en {total_shards} fragments Anycast...")

        # 2. Arrosage du réseau (Inondation UDP IPv6 ciblée)
        # Chaque shard est envoyé avec un FLAG précisant s'il est une donnée ou une parité
        for i, shard_bytes in enumerate(shards):
            is_parity = i >= self.fec.data_shards
            flags = 1 if is_parity else 0
            
            packet = self.aitp.create_packet(layer_id, shard_bytes, flags=flags)
            
            # Fire and forget absolu. Le réseau IPv6 fera le routage vers le GPU le plus proche
            # répondant à l'adresse Anycast_target.
            self.aitp.sock.sendto(packet, (anycast_target, self.aitp.port))
            
        logger.info(f"💥 Tenseur Layer {layer_id} pulvérisé sur le réseau avec succès ! (Zéro Acquittement TCP)")

# Exemple d'usage rapide :
if __name__ == "__main__":
    raid = NetworkRAIDManager({"type": "GPU", "compute": "RTX 4090", "vram": 16})
    raid.start()
    
    # Simulation d'un forward Pytorch : un tenseur de 1 Mega-octet
    dummy_tensor = bytes([42] * 1024 * 1024) 
    raid.dispatch_tensor_raid(layer_id=1, tensor_data=dummy_tensor)
