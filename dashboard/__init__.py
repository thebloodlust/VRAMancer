# Rend le package 'dashboard' importable et expose les interfaces principales

try:
    from .dashboard_cli import launch as launch_cli_dashboard  # type: ignore
except Exception:
    def launch_cli_dashboard():  # pragma: no cover
        print("CLI dashboard non disponible")

from .dashboard_web import launch as dashboard_web

__all__ = ['launch_cli_dashboard', 'dashboard_web']
