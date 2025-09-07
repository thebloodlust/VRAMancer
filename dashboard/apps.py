# dashboard/app.py
import argparse
import sys

# On importe les trois modes
from dashboard import dashboard_cli, dashboard_tk, dashboard_qt

def main():
    parser = argparse.ArgumentParser(description="Lanceur de dashboard VRAMancer")
    parser.add_argument(
        "--mode",
        choices=["cli", "tk", "qt"],
        default="cli",
        help="Choisir l’interface d’affichage (cli, tk, qt)",
    )
    args = parser.parse_args()

    if args.mode == "cli":
        dashboard_cli.launch()
    elif args.mode == "tk":
        dashboard_tk.launch_dashboard()
    elif args.mode == "qt":
        dashboard_qt.launch_dashboard()
    else:
        print("Mode inconnu.")
        sys.exit(1)

if __name__ == "__main__":
    main()
