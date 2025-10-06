"""Runner de tests smoke pour CI accélérée.
Sélectionne un sous-ensemble rapide des tests critiques.
Usage:
    python -m tests.smoke
Ou:
    pytest -q -k 'smoke'
"""
import sys
import pytest

# Liste de fichiers considérés smoke (rapides, sans multi-process ou stress massif)
SMOKE = [
    "tests/test_imports.py",
    "tests/test_integration_flask.py",
    "tests/test_scheduler.py",
    "tests/test_fastpath_endpoints.py",
    "tests/test_metrics_promotions.py",
]

if __name__ == "__main__":
    sys.exit(pytest.main(["-q", *SMOKE]))
