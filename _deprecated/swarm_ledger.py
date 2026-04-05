"""Swarm Ledger — P2P VRAM credit system (EXPERIMENTAL / NOT INTEGRATED).

This module is fully implemented but not wired into the orchestrator.
See core/network/cluster_discovery.py for the active discovery path.
"""
import os
import sqlite3
import threading
import uuid
import secrets
import hashlib
import hmac as _hmac
from typing import Dict, Any, Optional, List

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
    _stub_warned = False

    def __init__(self):
        if not SwarmLedger._stub_warned:
            SwarmLedger._stub_warned = True
            _logger.warning("STUB: swarm_ledger — orphan module, not integrated "
                           "with orchestrator (Grade D+)")
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
            # ── Private Groups (Cercles de Confiance) ──
            curr.execute("""
                CREATE TABLE IF NOT EXISTS groups (
                    group_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    invite_token_hash TEXT UNIQUE NOT NULL,
                    owner_id TEXT NOT NULL,
                    max_members INTEGER DEFAULT 50,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(owner_id) REFERENCES users(id)
                )
            """)
            curr.execute("""
                CREATE TABLE IF NOT EXISTS group_members (
                    group_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    node_id TEXT,
                    joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(group_id, user_id),
                    FOREIGN KEY(group_id) REFERENCES groups(group_id),
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

    # ══════════════════════════════════════════════════════════════════
    # Private Groups (Cercles de Confiance)
    # ══════════════════════════════════════════════════════════════════

    def create_group(self, name: str, owner_id: str, max_members: int = 50) -> tuple:
        """Create a private swarm group. Returns (group_id, invite_token).

        The invite_token is shown only once — share it with trusted members.
        Nodes must present this token to join the group.
        """
        group_id = str(uuid.uuid4())
        raw_token = secrets.token_urlsafe(32)
        invite_token = f"grp-{raw_token}"
        token_hash = hashlib.sha256(invite_token.encode()).hexdigest()

        with _lock:
            conn = self._get_conn()
            curr = conn.cursor()
            curr.execute(
                "INSERT INTO groups (group_id, name, invite_token_hash, owner_id, max_members) "
                "VALUES (?, ?, ?, ?, ?)",
                (group_id, name, token_hash, owner_id, max_members),
            )
            # Owner is automatically a member
            curr.execute(
                "INSERT INTO group_members (group_id, user_id) VALUES (?, ?)",
                (group_id, owner_id),
            )
            conn.commit()
            conn.close()

        _logger.info("Private group '%s' created by %s (max=%d)", name, owner_id, max_members)
        return group_id, invite_token

    def join_group(self, invite_token: str, user_id: str, node_id: Optional[str] = None) -> Optional[str]:
        """Join a private group using the invite token. Returns group_id or None."""
        token_hash = hashlib.sha256(invite_token.encode()).hexdigest()
        with _lock:
            conn = self._get_conn()
            curr = conn.cursor()
            curr.execute("SELECT group_id, max_members FROM groups WHERE invite_token_hash = ?", (token_hash,))
            row = curr.fetchone()
            if not row:
                conn.close()
                return None
            group_id, max_members = row

            # Check membership count
            curr.execute("SELECT COUNT(*) FROM group_members WHERE group_id = ?", (group_id,))
            count = curr.fetchone()[0]
            if count >= max_members:
                conn.close()
                _logger.warning("Group %s full (%d/%d)", group_id, count, max_members)
                return None

            # Idempotent join
            curr.execute(
                "INSERT OR IGNORE INTO group_members (group_id, user_id, node_id) VALUES (?, ?, ?)",
                (group_id, user_id, node_id),
            )
            conn.commit()
            conn.close()

        _logger.info("User %s joined group %s", user_id, group_id)
        return group_id

    def is_group_member(self, group_id: str, user_id: str) -> bool:
        """Check whether a user belongs to a group."""
        with _lock:
            conn = self._get_conn()
            curr = conn.cursor()
            curr.execute(
                "SELECT 1 FROM group_members WHERE group_id = ? AND user_id = ?",
                (group_id, user_id),
            )
            found = curr.fetchone() is not None
            conn.close()
        return found

    def get_group_members(self, group_id: str) -> List[Dict[str, Any]]:
        """List members of a group."""
        with _lock:
            conn = self._get_conn()
            curr = conn.cursor()
            curr.execute(
                "SELECT gm.user_id, u.alias, gm.node_id, gm.joined_at "
                "FROM group_members gm LEFT JOIN users u ON gm.user_id = u.id "
                "WHERE gm.group_id = ?",
                (group_id,),
            )
            rows = curr.fetchall()
            conn.close()
        return [
            {"user_id": r[0], "alias": r[1], "node_id": r[2], "joined_at": r[3]}
            for r in rows
        ]

    def get_user_groups(self, user_id: str) -> List[Dict[str, Any]]:
        """List groups a user belongs to."""
        with _lock:
            conn = self._get_conn()
            curr = conn.cursor()
            curr.execute(
                "SELECT g.group_id, g.name, g.owner_id "
                "FROM groups g JOIN group_members gm ON g.group_id = gm.group_id "
                "WHERE gm.user_id = ?",
                (user_id,),
            )
            rows = curr.fetchall()
            conn.close()
        return [{"group_id": r[0], "name": r[1], "owner_id": r[2]} for r in rows]

    def validate_group_token(self, invite_token: str) -> Optional[str]:
        """Validate an invite token without joining. Returns group_id or None."""
        token_hash = hashlib.sha256(invite_token.encode()).hexdigest()
        with _lock:
            conn = self._get_conn()
            curr = conn.cursor()
            curr.execute("SELECT group_id FROM groups WHERE invite_token_hash = ?", (token_hash,))
            row = curr.fetchone()
            conn.close()
        return row[0] if row else None

# Singleton du Ledger
ledger = SwarmLedger()
