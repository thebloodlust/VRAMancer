import argparse
from core.scheduler import SimpleScheduler
from core.memory_balancer import MemoryBalancer
from core.network.transport import Transport
from dashboard.flask_app import app as dashboard_app
from core.network.interface_selector import select_best_interface, list_interfaces

def check_environment():
    print("✅ Environnement vérifié (mock)")

def launch_scheduler(config_path):
    # Tu peux adapter ici si SimpleScheduler attend autre chose que config_path
    scheduler = SimpleScheduler(blocks=[], callbacks={})  # Exemple minimal
    print("🚀 Scheduler lancé (mock)")

def launch_transport(config_path, interface):
    transport = Transport(config_path, interface)
    transport.initialize()

def launch_dashboard(port, debug):
    dashboard_app.run(port=port, debug=debug)

def main():
    parser = argparse.ArgumentParser(description="VRAMancer Launcher")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dashboard", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--interface", type=str)
    parser.add_argument("--list-interfaces", action="store_true")
    args = parser.parse_args()

    if args.list_interfaces:
        list_interfaces()
        return

    check_environment()

    if not args.dry_run:
        launch_scheduler(args.config)
        iface = args.interface or select_best_interface()
        launch_transport(args.config, iface)

    if args.dashboard:
        launch_dashboard(args.port, args.debug)

if __name__ == "__main__":
    main()
