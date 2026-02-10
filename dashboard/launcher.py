# launcher.py

import argparse
import importlib.util
import subprocess
import sys

from dashboard import dashboard_web

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
        dashboard_web.launch()
