# Rend le package 'dashboard' importable et expose les interfaces principales

from .dashboard_cli import launch_cli_dashboard
from .dashboard_tk import launch_dashboard as launch_tk_dashboard
from .dashboard_web import launch_web_dashboard
from .updater import update_dashboard
