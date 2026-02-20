"""Zero Trust Security Module.

This module implements strict zero-trust policies for the VRAMancer API.
It verifies that no default or insecure credentials are used in production.
"""
import os
import logging
from core.auth_strong import _USERS, verify_user

logger = logging.getLogger(__name__)

def authenticate(request) -> bool:
    """
    Zero-trust authentication check.
    In production, this enforces strict checks.
    """
    # If not in production, we can be more lenient, but still warn
    is_prod = os.environ.get("VRM_PRODUCTION", "0") == "1"
    
    # Check for default admin/admin credentials
    if "admin" in _USERS and verify_user("admin", "admin"):
        msg = "SECURITY BREACH: Default 'admin/admin' credentials detected!"
        if is_prod:
            logger.critical(f"{msg} Refusing authentication in production.")
            return False
        else:
            logger.warning(f"{msg} Please change this immediately.")
            
    # Implement actual token/MFA verification here based on request headers
    # For now, we rely on the existing security middleware for token validation,
    # but this hook can be expanded for deeper zero-trust checks (e.g., device posture).
    
    return True

def enforce_zero_trust_startup():
    """
    Called during API startup to ensure no insecure configurations exist.
    Raises RuntimeError if a critical security violation is found in production.
    """
    is_prod = os.environ.get("VRM_PRODUCTION", "0") == "1"
    
    if is_prod:
        if "admin" in _USERS and verify_user("admin", "admin"):
            raise RuntimeError(
                "SECURITY FATAL: Cannot start API in production mode with "
                "default 'admin/admin' credentials. Please configure a secure password."
            )
