import argparse
from dashboard import dashboard_cli, dashboard_tk, dashboard_qt

def main():
    parser = argparse.ArgumentParser(description="VRAMancer Dashboard Launcher")
    parser.add_argument("--mode", choices=["cli", "tk", "qt"], default="cli", help="Choisir l'interface")
    args = parser.parse_args()

    if args.mode == "cli":
        dashboard_cli.launch()
    elif args.mode == "tk":
        dashboard_tk.launch_dashboard()
    elif args.mode == "qt":
        dashboard_qt.launch_dashboard()
    else:
        print("Mode inconnu.")

if __name__ == "__main__":
    main()
