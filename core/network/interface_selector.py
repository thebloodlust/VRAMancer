def select_network_interface(mode="auto", preferred=None):
    """
    Sélectionne une interface réseau :
    - mode="auto" : choisit la meilleure interface dispo (via select_best_interface)
    - mode="manual" : demande à l’utilisateur de choisir (via list_interfaces)
    - preferred : liste d’interfaces à privilégier (optionnel)
    """
    available = list(psutil.net_if_addrs().keys())
    if not available:
        print("❌ Aucune interface réseau détectée.")
        return None
    if mode == "manual":
        print("Interfaces réseau disponibles :")
        for i, iface in enumerate(available, 1):
            print(f"  [{i}] {iface}")
        try:
            idx = int(input("Sélectionnez l’interface (1‑N) : ")) - 1
            if 0 <= idx < len(available):
                print(f"✅ Interface sélectionnée manuellement : {available[idx]}")
                return available[idx]
        except Exception:
            pass
        print(f"⚠️ Sélection automatique fallback : {available[0]}")
        return available[0]
    # mode auto
    return select_best_interface(preferred)
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