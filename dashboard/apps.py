# dashboard/app.py
import argparse
import sys

# Import des wrappers
from dashboard.dashboard_cli import launch
from dashboard.dashboard_tk import launch_dashboard as launch_tk
from dashboard.dashboard_qt import launch_dashboard as launch_qt  # optional

def main():
    parser = argparse.ArgumentParser(description="VRAMancer Dashboard Launcher")
    parser.add_argument(
        "--mode",
        choices=["cli", "tk", "qt"],
        default="cli",
        help="Choisir l'interface d'affichage",
    )
    args = parser.parse_args()

    if args.mode == "cli":
        launch()
    elif args.mode == "tk":
        launch_tk()
    elif args.mode == "qt":
        launch_qt()
    else:
        print("Mode inconnu.")
        sys.exit(1)

if __name__ == "__main__":
    main()
