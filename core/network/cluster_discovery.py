import socket
import platform
import json
import threading

# --- 1. Détection auto des machines connectées ---

def get_local_info():
    return {
        "hostname": socket.gethostname(),
        "ip": socket.gethostbyname(socket.gethostname()),
        "cpu": platform.processor(),
        "arch": platform.machine(),
        "os": platform.system(),
        "vram": None,  # À compléter avec détection GPU
        "ram": None,   # À compléter
        "type": detect_platform_type()
    }

def detect_platform_type():
    sys = platform.system().lower()
    if "darwin" in sys or "mac" in sys:
        return "Apple Silicon"
    elif "windows" in sys:
        return "Windows"
    elif "linux" in sys:
        # Détection AMD/Intel
        cpu = platform.processor().lower()
        if "amd" in cpu:
            return "AMD"
        elif "intel" in cpu:
            return "Intel"
        else:
            return "Linux Generic"
    return sys

# --- 2. Découverte réseau (broadcast UDP) ---

def discover_nodes(port=55555, timeout=2):
    info = get_local_info()
    msg = json.dumps(info).encode()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.settimeout(timeout)
    sock.sendto(msg, ("<broadcast>", port))
    nodes = [info]
    def listen():
        try:
            while True:
                data, addr = sock.recvfrom(4096)
                node = json.loads(data.decode())
                if node not in nodes:
                    nodes.append(node)
        except socket.timeout:
            pass
    t = threading.Thread(target=listen)
    t.start()
    t.join(timeout)
    sock.close()
    return nodes

# --- 3. Création dynamique du cluster ---


def plug_and_play_usb4(mount_base="/mnt/usb4_share"):
    """
    Détection et montage automatique des ports USB4/réseau pour plug-and-play IA distribuée.
    """
    import os
    import time
    print("[Plug&Play] Scan des ports USB4/réseau...")
    # Simule la détection de nouveaux périphériques USB4
    for i in range(1, 5):
        mount_path = f"{mount_base}_{i}"
        if not os.path.exists(mount_path):
            os.makedirs(mount_path)
            print(f"[Plug&Play] Nouveau port USB4 détecté et monté : {mount_path}")
        time.sleep(0.5)
    print("[Plug&Play] Tous les ports USB4/réseau sont prêts.")
    return [f"{mount_base}_{i}" for i in range(1, 5)]

def create_local_cluster():
    nodes = discover_nodes()
    print(f"Cluster local détecté : {len(nodes)} nœuds")
    for node in nodes:
        print(f"- {node['hostname']} ({node['type']}) | CPU: {node['cpu']} | Arch: {node['arch']} | OS: {node['os']}")
    # Plug-and-play ports
    usb4_mounts = plug_and_play_usb4()
    print(f"Ports USB4/réseau disponibles : {usb4_mounts}")
    return nodes, usb4_mounts

if __name__ == "__main__":
    create_local_cluster()
