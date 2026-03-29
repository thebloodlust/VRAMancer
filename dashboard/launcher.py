# launcher.py

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="VRAMancer Dashboard Launcher")
    parser.add_argument(
        "--mode",
        choices=["cli", "web"],
        default="web",
        help="Choisir l'interface : cli, web"
    )
    args = parser.parse_args()

    if args.mode == "cli":
        from dashboard import launch_cli_dashboard
        launch_cli_dashboard()
    else:
        from dashboard.dashboard_web import launch
        launch()

if __name__ == "__main__":
    main()
