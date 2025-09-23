# launcher.py

import argparse
import importlib.util
import subprocess
import sys

from dashboard import dashboard_cli, dashboard_tk, dashboard_qt

def launch_web():
    subprocess.run([sys.executable, "dashboard/app.py"])

def is_available(module_name):
    return importlib.util.find_spec(module_name) is not None

def main():
    parser = argparse.ArgumentParser(description="VRAMancer Dashboard Launcher")
    parser.add_argument(
        "--mode",
        choices=["cli", "tk", "qt", "web", "auto"],
        default="cli",
        help="Choisir l'interface : cli, tk, qt, web, auto"
    )
    args = parser.parse_args()

    if args.mode == "cli":
        dashboard_cli.launch()

    elif args.mode == "tk":
        dashboard_tk.launch_dashboard()

    elif args.mode == "qt":
        dashboard_qt.launch_dashboard()

    elif args.mode == "web":
        launch_web()

    elif args.mode == "auto":
        if is_available("PyQt5"):
            print("🪟 Qt détecté → lancement Qt")
            dashboard_qt.launch_dashboard()
        elif is_available("tkinter"):
            print("🧱 Tkinter détecté → lancement Tk")
            dashboard_tk.launch_dashboard()
        else:
            print("🖥️ Aucun GUI détecté → lancement CLI")
            dashboard_cli.launch()

    else:
        print("❌ Mode inconnu.")

if __name__ == "__main__":
    main()
