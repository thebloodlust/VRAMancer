#!/usr/bin/env python3
"""D2.9 — tests de `enforce_startup_checks()` (aucune couverture avant).

La fonction est le garde-fou de démarrage en production : elle doit REFUSER de
laisser l'API démarrer si les identifiants par défaut, un token manquant ou une
variable de test dangereuse sont présents. On vérifie les 4 refus, et surtout
qu'en mode non-production elle ne bloque rien (sinon plus personne ne peut
développer).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.auth_strong import _USERS, create_user
from core.security.startup_checks import authenticate, enforce_startup_checks


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Environnement de prod minimal VALIDE ; chaque test casse une seule chose."""
    for var in ("VRM_MINIMAL_TEST", "VRM_TEST_RELAX_SECURITY", "VRM_TEST_BYPASS_HA",
                "VRM_TEST_MODE", "VRM_TEST_ALL_OPEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("VRM_API_TOKEN", "un-token-de-test")
    monkeypatch.setenv("VRM_AUTH_SECRET", "a" * 64)
    saved = dict(_USERS)
    _USERS.clear()
    yield
    _USERS.clear()
    _USERS.update(saved)


def _make_default_admin():
    """Recrée le cas dangereux : compte admin avec le mot de passe admin."""
    _USERS.pop("admin", None)
    create_user("admin", "admin", role="admin")


# ---------------------------------------------------------------- production

def test_prod_refuses_default_admin_credentials(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    _make_default_admin()
    with pytest.raises(RuntimeError, match="admin"):
        enforce_startup_checks()


def test_prod_refuses_missing_api_token(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.delenv("VRM_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="VRM_API_TOKEN"):
        enforce_startup_checks()


def test_prod_refuses_missing_auth_secret(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.delenv("VRM_AUTH_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="VRM_AUTH_SECRET"):
        enforce_startup_checks()


@pytest.mark.parametrize("var", ["VRM_MINIMAL_TEST", "VRM_TEST_RELAX_SECURITY",
                                 "VRM_TEST_BYPASS_HA"])
def test_prod_refuses_test_bypass_vars(monkeypatch, var):
    """Une variable de test qui fuit en prod désactive la sécurité : refus."""
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv(var, "1")
    with pytest.raises(RuntimeError, match=var):
        enforce_startup_checks()


def test_prod_passes_with_clean_config(monkeypatch):
    """Config de prod saine : aucun refus (sinon l'API ne démarrerait jamais)."""
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    create_user("operateur", "un-mot-de-passe-solide", role="admin")
    enforce_startup_checks()      # ne doit pas lever


# ------------------------------------------------------------ non-production

def test_dev_mode_allows_everything(monkeypatch):
    """Hors production, aucun de ces cas ne bloque le démarrage."""
    monkeypatch.delenv("VRM_PRODUCTION", raising=False)
    monkeypatch.delenv("VRM_API_TOKEN", raising=False)
    monkeypatch.delenv("VRM_AUTH_SECRET", raising=False)
    monkeypatch.setenv("VRM_TEST_RELAX_SECURITY", "1")
    _make_default_admin()
    enforce_startup_checks()      # ne doit pas lever


# ------------------------------------------------------------- authenticate()

def test_authenticate_refuses_default_admin_in_prod(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    _make_default_admin()
    assert authenticate(object()) is False


def test_authenticate_warns_but_allows_in_dev(monkeypatch):
    monkeypatch.delenv("VRM_PRODUCTION", raising=False)
    _make_default_admin()
    assert authenticate(object()) is True


def test_authenticate_ok_without_default_admin(monkeypatch):
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    create_user("operateur", "un-mot-de-passe-solide", role="admin")
    assert authenticate(object()) is True


# ----------------------------------- verify_request : les bypass sont bloqués en prod

def test_prod_blocks_relax_security_bypass(monkeypatch):
    """VRM_TEST_RELAX_SECURITY ne doit RIEN ouvrir quand VRM_PRODUCTION=1."""
    from core.security import verify_request
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_TEST_RELAX_SECURITY", "1")
    res = verify_request("secret", "POST", "/api/generate", {}, b"")
    assert res is not None and res[1] == 401


def test_prod_blocks_ha_bypass(monkeypatch):
    from core.security import verify_request
    monkeypatch.setenv("VRM_PRODUCTION", "1")
    monkeypatch.setenv("VRM_TEST_BYPASS_HA", "1")
    res = verify_request("secret", "POST", "/api/ha/apply", {}, b"")
    assert res is not None and res[1] == 401
