# Rend le package 'dashboard' importable et expose les interfaces principales

from .cli.main import launch as dashboard_cli
from .tk.main import launch_dashboard as dashboard_tk
from .qt.main import launch_dashboard as dashboard_qt
from .dashboard_web import launch as dashboard_web

# Optionnel : expose aussi l'updater si toujours utilisé
try:
    from .updater import update_dashboard
except ImportError:
    update_dashboard = None
