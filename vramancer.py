import time
import sys
import importlib.util

from utils.gpu_utils import get_available_gpus
from dashboard.dashboard_tk import launch_dashboard

def show_gpu_summary():
    print("\n🔍 GPU Summary:")
    gpus = get_available_gpus()
    for gpu in gpus:
        status = "✅" if gpu["is_available"] else "❌"
        print(f"{status} GPU {gpu['id']} — {gpu['name']} — {gpu['total_vram_mb']} MB VRAM")

def has_module(module_name):
    return importlib.util.find_spec(module_name) is not None

def auto_mode():
    print("🔎 Détection des modules disponibles...")
    if has_module("PyQt5"):
        print("🧪 PyQt5 détecté. Lancement du mode PyQt5...")
        # from dashboard.dashboard_qt import launch_qt_dashboard
        # launch_qt_dashboard()
        print("🚧 (à implémenter)")
    elif has_module("tkinter"):
        print("🎨 Tkinter détecté. Lancement du dashboard...")
        launch_dashboard()
    else:
        print("🖥️ Aucun module graphique détecté. Affichage terminal.")
        show_gpu_summary()

def main():
    print("\033c")
    print("🎛️ Bienvenue dans VRAMancer\n")
    print("Choisissez un mode :")
    print("1️⃣  Mode Terminal")
    print("2️⃣  Mode Tkinter")
    print("3️⃣  Mode PyQt5 (expérimental)")
    print("4️⃣  Mode Auto (choix intelligent)\n")

    choice = input("👉 Entrez le numéro du mode : ").strip()

    if choice == "1":
        show_gpu_summary()
    elif choice == "2":
        launch_dashboard()
    elif choice == "3":
        print("🚧 Mode PyQt5 non encore implémenté.")
    elif choice == "4":
        auto_mode()
    else:
        print("❌ Choix invalide. Veuillez relancer le programme.")

if __name__ == "__main__":
    main()
