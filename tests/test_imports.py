def test_core_network_imports():
    from core.network import Transport, select_best_interface
    assert callable(select_best_interface)

def test_dashboard_imports():
    from dashboard import launch_cli_dashboard
    assert callable(launch_cli_dashboard)
