# Rend le package 'dashboard' importable et expose les interfaces principales

try:
    from .cli_dashboard import launch as launch_cli_dashboard  # type: ignore
except Exception as e:
    def launch_cli_dashboard():  # pragma: no cover
        import traceback
        import sys
        print("CLI dashboard non disponible : erreur détaillée ci-dessous :")
        traceback.print_exc(file=sys.stdout)
        print("Erreur :", e)

from .dashboard_web import launch as dashboard_web

__all__ = ['launch_cli_dashboard', 'dashboard_web']
