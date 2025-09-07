import psutil

def list_network_interfaces() -> List[str]:
    """Retourne la liste des interfaces réseau (avec leur type)."""
    ifaces = psutil.net_if_addrs()
    names = []
    for iface, addrs in ifaces.items():
        # On ne garde que les interfaces Ethernet ou SFP (oui, on peut filtrer via `addrs[0].family`)
        if any(addr.family == socket.AF_LINK for addr in addrs):
            names.append(iface)
    return names

def choose_interface() -> str:
    """Affiche un menu et renvoie l’interface choisie."""
    interfaces = list_network_interfaces()
    print("Sélectionnez un port réseau :")
    for i, name in enumerate(interfaces, 1):
        print(f"  [{i}] {name}")
    choice = int(input("Entrée (1‑N) : "))
    return interfaces[choice-1]
