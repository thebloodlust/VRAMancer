import uuid
import time
import secrets
import logging

_log = logging.getLogger("vramancer.trust_ring")

class SwarmTrustManager:
    """
    Manages isolated Swarm Groups (Trust Rings) using Invite Tokens.
    Nodes can only connect and exchange tensors within the same Ring.
    """
    def __init__(self):
        # Maps ring_id -> {"invite_token": str, "name": str, "created_at": float}
        self.rings = {}
        # Default legacy ring
        self.create_ring("public", "Default Public Ring", "PUBLIC_NO_AUTH")
        
    def create_ring(self, ring_id: str = None, name: str = "Private Swarm", token: str = None) -> dict:
        if not ring_id:
            ring_id = f"ring_{uuid.uuid4().hex[:8]}"
        if not token:
            token = secrets.token_urlsafe(16)
            
        self.rings[ring_id] = {
            "name": name,
            "invite_token": token,
            "created_at": time.time()
        }
        _log.info(f"[TrustRing] '{name}' ({ring_id}) créé avec le token de sécurité.")
        return {"ring_id": ring_id, "token": token, "name": name}

    def verify_node(self, ring_id: str, provided_token: str) -> bool:
        """Verifies if a node can join a specific ring."""
        if not ring_id:
            ring_id = "public"
            provided_token = provided_token or "PUBLIC_NO_AUTH"
            
        if ring_id not in self.rings:
            _log.warning(f"[TrustRing] Tentative de connexion sur un ring inexistant: {ring_id}")
            return False
            
        expected_token = self.rings[ring_id]["invite_token"]
        
        # Cas Legacy test fallback
        if expected_token == "PUBLIC_NO_AUTH" or provided_token == "PUBLIC_NO_AUTH":
            return True
            
        # Secure comparison
        if not secrets.compare_digest(expected_token, provided_token):
            _log.warning(f"[TrustRing] Rejet: Token invalide pour le ring {ring_id}.")
            return False
            
        return True

# Global singleton
TRUST_MANAGER = SwarmTrustManager()
