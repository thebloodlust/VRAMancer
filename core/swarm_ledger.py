import os
import sqlite3
import threading
import uuid
import secrets
import hashlib
from typing import Dict, Any, Optional

try:
    from core.logger import LoggerAdapter
    _logger = LoggerAdapter("ledger")
except Exception:
    import logging
    _logger = logging.getLogger("ledger")

_lock = threading.Lock()
_LEDGER_PATH = os.environ.get("VRM_LEDGER_PATH", "vramancer_ledger.db")

class SwarmLedger:
    """
    Système de base de données (Ledger) pour gérer l'économie du Swarm P2P.
    Gère les Clés API (sk-VRAM-xxx), les soldes de crédits VRAM (fournis vs consommés).
    """
    def __init__(self):
        self._ensure_tables()

    def _get_conn(self):
        return sqlite3.connect(_LEDGER_PATH, check_same_thread=False)

    def _ensure_tables(self):
        with _lock:
            conn = self._get_conn()
            curr = conn.cursor()
            # Table Utilisateurs & Solde
            curr.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    api_key_hash TEXT UNIQUE,
                    alias TEXT,
                    vram_credits REAL DEFAULT 100.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Table des Nœuds du Swarm (Qui donne quoi)
            curr.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    host TEXT,
                    contribution_mb REAL,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            conn.commit()
            conn.close()

    def create_user(self, alias: str) -> tuple[str, str]:
        """Crée un utilisateur et lui retourne sa clé secrète (sk-VRAM-...) qui ne sera affichée qu'une fois."""
        user_id = str(uuid.uuid4())
        raw_secret = secrets.token_urlsafe(32)
        api_key = f"sk-VRAM-{raw_secret}"
        # On ne stocke que le hash de la clé (sécurité industrielle)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        with _lock:
            conn = self._get_conn()
            curr = conn.cursor()
            curr.execute("INSERT INTO users (id, api_key_hash, alias) VALUES (?, ?, ?)", 
                         (user_id, key_hash, alias))
            conn.commit()
            conn.close()
            
        _logger.info(f"Création d'un nouveau compte Swarm pour: {alias}")
        return user_id, api_key

    def verify_and_get_user(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Sécurité Zero-Trust: Vérifie la clé API et renvoie l'utilisateur si elle est valide."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        with _lock:
            conn = self._get_conn()
            curr = conn.cursor()
            curr.execute("SELECT id, alias, vram_credits FROM users WHERE api_key_hash = ?", (key_hash,))
            row = curr.fetchone()
            conn.close()
            
        if row:
            return {"id": row[0], "alias": row[1], "vram_credits": row[2]}
        return None

    def consume_credits(self, user_id: str, tokens_generated: int) -> bool:
        """Débite les crédits d'un utilisateur pour une inférence."""
        cost = tokens_generated * 0.001 # Tarif arbitraire (ex: 1 crédit = 1000 tokens)
        with _lock:
            conn = self._get_conn()
            curr = conn.cursor()
            curr.execute("SELECT vram_credits FROM users WHERE id = ?", (user_id,))
            row = curr.fetchone()
            if not row or row[0] < cost:
                conn.close()
                return False # Pas assez de crédits / Over quota
            
            curr.execute("UPDATE users SET vram_credits = vram_credits - ? WHERE id = ?", (cost, user_id))
            conn.commit()
            conn.close()
            return True

    def reward_node(self, node_id: str, blocks_processed: int):
        """Donne des crédits à un utilisateur dont le PC a aidé les autres."""
        reward = blocks_processed * 0.0005
        with _lock:
            conn = self._get_conn()
            curr = conn.cursor()
            # Trouver à quel user appartient ce node
            curr.execute("SELECT user_id FROM nodes WHERE node_id = ?", (node_id,))
            row = curr.fetchone()
            if row:
                user_id = row[0]
                curr.execute("UPDATE users SET vram_credits = vram_credits + ? WHERE id = ?", (reward, user_id))
                conn.commit()
            conn.close()

# Singleton du Ledger
ledger = SwarmLedger()
