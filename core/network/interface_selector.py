import psutil

def list_interfaces():
    print("🔍 Interfaces réseau disponibles :")
    for iface, addrs in psutil.net_if_addrs().items():
        print(f" - {iface}")

def select_best_interface(preferred=None):
    if preferred is None:
        preferred = ["enp1s0f0", "eth0", "eno1"]
    available = list(psutil.net_if_addrs().keys())
    if not available:
        print("❌ Aucune interface réseau détectée.")
        return None
    for iface in preferred:
        if iface in available:
            print(f"✅ Interface sélectionnée automatiquement : {iface}")
            return iface
    fallback = available[0]
    print(f"⚠️ Interface par défaut : {fallback}")
    return fallback