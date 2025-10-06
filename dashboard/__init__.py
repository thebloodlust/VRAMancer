# Rend le package 'dashboard' importable et expose les interfaces principales


try:
    from .dashboard_cli import launch as launch_cli_dashboard  # type: ignore
except Exception:
    def launch_cli_dashboard():  # pragma: no cover
        print("CLI dashboard non disponible")
try:
    from .dashboard_tk import launch_dashboard as dashboard_tk  # type: ignore
except Exception:
    dashboard_tk = lambda: print("Tk dashboard non disponible")
try:
    from .dashboard_qt import launch_dashboard as dashboard_qt  # type: ignore
except Exception:
    dashboard_qt = lambda: print("Qt dashboard non disponible")
from .dashboard_web import launch as dashboard_web

# Optionnel : expose aussi l'updater si toujours utilisé
try:
    from .updater import update_dashboard
except ImportError:
    def update_dashboard():  # pragma: no cover
        return False

__all__ = [
    'launch_cli_dashboard', 'dashboard_tk', 'dashboard_qt', 'dashboard_web', 'update_dashboard'
]
