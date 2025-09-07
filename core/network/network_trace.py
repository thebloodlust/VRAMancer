# network_trace.py
from scapy.all import sniff

def trace_custom_protocol(port, iface="eth0"):
    def packet_callback(pkt):
        # Exemple de filtre pour un protocole custom (e.g. TCP port 12345)
        if pkt.haslayer('TCP') and pkt['TCP'].dport == 12345:
            print(f"Custom packet from {pkt['IP'].src} -> {pkt['IP'].dst}")

    sniff(iface=iface, prn=packet_callback, filter="tcp port 12345", store=0)
