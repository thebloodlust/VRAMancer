import psutil

def list_interfaces():
    print("🔍 Interfaces réseau disponibles :")
    for iface, addrs in psutil.net_if_addrs().items():
        print(f" - {iface}")

def select_best_interface():
    preferred = ["enp1s0f0", "eth0", "eno1"]
    available = psutil.net_if_addrs().keys()
    for iface in preferred:
        if iface in available:
            print(f"✅ Interface sélectionnée automatiquement : {iface}")
            return iface
    fallback = list(available)[0]
    print(f"⚠️ Interface par défaut : {fallback}")
    return fallback
