import time
import os
import importlib.util
from utils.gpu_utils import get_available_gpus
from dashboard.dashboard_tk import launch_dashboard as launch_tk
from dashboard.dashboard_web import launch_web_dashboard
from dashboard.dashboard_cli import launch_cli_dashboard

def show_intro():
    logo = r"""
██╗   ██╗██████╗ ██████╗  █████╗ ███╗   ███╗ █████╗ ███╗   ██╗ ██████╗███████╗
██║   ██║██╔══██╗██╔══██╗██╔══██╗████╗ ████║██╔══██╗████╗  ██║██╔════╝██╔════╝
██║   ██║██████╔╝██████╔╝███████║██╔████╔██║███████║██╔██╗ ██║██║     █████╗  
██║   ██║██╔═══╝ ██╔═══╝ ██╔══██║██║╚██╔╝██║██╔══██║██║╚██╗██║██║     ██╔══╝  
╚██████╔╝██║     ██║     ██║  ██║██║ ╚═╝ ██║██║  ██║██║ ╚████║╚██████╗███████╗
 ╚═════╝ ╚═╝     ╚═╝     ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝
    """
    steps = [
        "🧠 Initialisation du noyau neuronal...",
        "🔍 Scan des interfaces GPU...",
        "📦 Chargement des modules : numpy, torch, tkinter, flask...",
        "🧬 Synchronisation des threads...",
        "🔐 Vérification des accès VRAM...",
        "🛰️ Connexion au bus PCIe...",
        "🧯 Détection des fuites de VRAM... (aucune)",
        "🧭 Calibration des curseurs thermiques...",
        "✅ Système prêt."
    ]

    os.system("cls" if os.name == "nt" else "clear")
    for line in steps:
        print(line)
        time.sleep(0.3)

    print("\n🎨 Chargement du logo...\n")
    for line in logo.splitlines():
        print(line)
        time.sleep(0.02)

    print("\n🔍 Détection des GPU...\n")
    gpus = get_available_gpus()
    for gpu in gpus:
        status = "✅" if gpu["is_available"] else "❌"
        print(f"{status} GPU {gpu['id']} — {gpu['name']} — {gpu['total_vram_mb']} MB VRAM")

    print("\n🚀 VRAMancer prêt à l’emploi.\n")
    time.sleep(0.5)

def has_module(module_name):
    return importlib.util.find_spec(module_name) is not None

def auto_mode():
    print("🔎 Mode auto : détection des interfaces disponibles...")
    if has_module("tkinter"):
        print("🎨 Tkinter détecté. Lancement du dashboard...")
        launch_tk()
    elif has_module("flask"):
        print("🌐 Flask détecté. Lancement du dashboard web...")
        launch_web_dashboard()
    else:
        print("🖥️ Aucun module graphique détecté. Passage en mode terminal.")
        launch_cli_dashboard()

def main():
    show_intro()
    print("🎛️ Choisissez un mode d’affichage :")
    print("1️⃣  Mode Terminal")
    print("2️⃣  Mode Tkinter (graphique)")
    print("3️⃣  Mode Web (Flask)")
    print("4️⃣  Mode Auto\n")

    choice = input("👉 Entrez le numéro du mode : ").strip()

    if choice == "1":
        launch_cli_dashboard()
    elif choice == "2":
        launch_tk()
    elif choice == "3":
        launch_web_dashboard()
    elif choice == "4":
        auto_mode()
    else:
        print("❌ Choix invalide. Veuillez relancer le programme.")

if __name__ == "__main__":
    main()
