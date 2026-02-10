"""CLI Dashboard — redirige vers le module canonique."""
# Le code canonique vit dans dashboard/cli_dashboard.py
# Ce fichier existe pour compatibilite des imports.
try:
    from dashboard.cli_dashboard import launch, clear_console  # noqa: F401
except ImportError:
    from core.memory_balancer import MemoryBalancer  # noqa: F401
    def launch():
        print("Dashboard CLI indisponible (dashboard package manquant)")
    def clear_console():
        import os
        os.system("cls" if os.name == "nt" else "clear")

launch_cli_dashboard = launch  # alias compat

if __name__ == "__main__":
    launch()
