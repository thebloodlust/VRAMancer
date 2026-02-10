# network_trace.py
"""Capture reseau via scapy (optionnel)."""
try:
    from scapy.all import sniff  # type: ignore
    _HAS_SCAPY = True
except ImportError:
    sniff = None  # type: ignore
    _HAS_SCAPY = False

def trace_custom_protocol(port=12345, iface="eth0"):
    if not _HAS_SCAPY:
        raise ImportError("scapy requis : pip install scapy")

    def packet_callback(pkt):
        if pkt.haslayer('TCP') and pkt['TCP'].dport == port:
            print(f"Custom packet from {pkt['IP'].src} -> {pkt['IP'].dst}")

    sniff(iface=iface, prn=packet_callback, filter=f"tcp port {port}", store=0)
