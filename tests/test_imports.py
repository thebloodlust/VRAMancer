def test_core_network_imports():
    from core.network import Transport
    assert Transport is not None

def test_dashboard_imports():
    from dashboard import launch_cli_dashboard
    assert callable(launch_cli_dashboard)
