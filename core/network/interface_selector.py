import logging

try:
    import psutil
except ImportError:
    psutil = None

def select_network_interface(mode="auto", preferred=None):
    """
    Sélectionne une interface réseau :
    - mode="auto" : choisit la meilleure interface dispo (via select_best_interface)
    - mode="manual" : demande à l'utilisateur de choisir (via list_interfaces)
    - preferred : liste d'interfaces à privilégier (optionnel)
    """
    if psutil is None:
        logging.warning("psutil not available — cannot select network interface")
        return None
    available = list(psutil.net_if_addrs().keys())
    if not available:
        logging.info(" Aucune interface réseau détectée.")
        return None
    if mode == "manual":
        logging.info("Interfaces réseau disponibles :")
        for i, iface in enumerate(available, 1):
            logging.info(f"  [{i}] {iface}")
        try:
            idx = int(input("Sélectionnez l’interface (1‑N) : ")) - 1
            if 0 <= idx < len(available):
                logging.info(f" Interface sélectionnée manuellement : {available[idx]}")
                return available[idx]
        except Exception:
            pass
        logging.info(f" Sélection automatique fallback : {available[0]}")
        return available[0]
    # mode auto
    return select_best_interface(preferred)


def list_interfaces():
    if psutil is None:
        logging.warning("psutil not available")
        return
    logging.info(" Interfaces réseau disponibles :")
    for iface, addrs in psutil.net_if_addrs().items():
        logging.info(f" - {iface}")

def select_best_interface(preferred=None):
    if psutil is None:
        return None
    if preferred is None:
        preferred = ["enp1s0f0", "eth0", "eno1"]
    available = list(psutil.net_if_addrs().keys())
    if not available:
        logging.info(" Aucune interface réseau détectée.")
        return None
    for iface in preferred:
        if iface in available:
            logging.info(f" Interface sélectionnée automatiquement : {iface}")
            return iface
    fallback = available[0]
    logging.info(f" Interface par défaut : {fallback}")
    return fallback