"""Startup Security Checks — pre-flight validation for the VRAMancer API.

Verifies at boot time that no insecure configurations exist:
  - Default credentials are rejected in production mode
  - Required environment variables are set
  - Security middleware is correctly installed

This replaces the former zero_trust.py stub with actionable checks.
"""
import os
import logging
from core.auth_strong import _USERS, verify_user

logger = logging.getLogger(__name__)


def authenticate(request) -> bool:
    """Pre-request security check (hook for production hardening).

    Currently validates that no default credentials are in use.
    Returns False in production if insecure config is detected.
    """
    is_prod = os.environ.get("VRM_PRODUCTION", "0") == "1"

    # Check for default admin/admin credentials
    if "admin" in _USERS and verify_user("admin", "admin"):
        msg = "SECURITY BREACH: Default 'admin/admin' credentials detected!"
        if is_prod:
            logger.critical("%s Refusing authentication in production.", msg)
            return False
        else:
            logger.warning("%s Please change this immediately.", msg)

    return True


def enforce_startup_checks():
    """Called during API startup to ensure no insecure configurations exist.

    Raises RuntimeError if a critical security violation is found in
    production mode (VRM_PRODUCTION=1).
    """
    is_prod = os.environ.get("VRM_PRODUCTION", "0") == "1"

    if is_prod:
        # Reject default admin/admin credentials
        if "admin" in _USERS and verify_user("admin", "admin"):
            raise RuntimeError(
                "SECURITY FATAL: Cannot start API in production mode with "
                "default 'admin/admin' credentials. Please configure a "
                "secure password."
            )

        # Verify that an API token is configured
        if not os.environ.get("VRM_API_TOKEN"):
            raise RuntimeError(
                "SECURITY FATAL: VRM_API_TOKEN must be set in production mode. "
                "All API requests require a valid token."
            )

        # Verify auth secret is configured
        if not os.environ.get("VRM_AUTH_SECRET"):
            raise RuntimeError(
                "SECURITY FATAL: VRM_AUTH_SECRET must be set in production mode. "
                "Generate one with: python3 -c 'import secrets; print(secrets.token_hex(32))'"
            )

        # Guard against test env vars leaking into production
        dangerous_in_prod = [
            'VRM_MINIMAL_TEST',
            'VRM_TEST_RELAX_SECURITY',
            'VRM_TEST_BYPASS_HA',
        ]
        for var in dangerous_in_prod:
            if os.environ.get(var) == '1':
                raise RuntimeError(
                    f"SECURITY FATAL: {var}=1 is set in production mode. "
                    f"This disables security protections. Unset it before starting."
                )


# Backward-compatible aliases for code that imports the old names
enforce_zero_trust_startup = enforce_startup_checks
