"""
Module d’auditabilité/logs/fairness/XAI pour VRAMancer
Explications, audit, fairness, logs IA, conformité.
"""
import time

class AuditLog:
    def __init__(self):
        self.logs = []

    def log_action(self, node, action, explanation=None, fairness=None):
        entry = {
            "node": node,
            "action": action,
            "timestamp": time.time(),
            "explanation": explanation,
            "fairness": fairness
        }
        self.logs.append(entry)
        return entry

    def get_logs(self):
        return self.logs

    def audit(self):
        """Retourne un rapport d’audit (actions, fairness, explications)."""
        return {
            "total": len(self.logs),
            "actions": [l["action"] for l in self.logs],
            "fairness": [l["fairness"] for l in self.logs if l["fairness"] is not None],
            "explications": [l["explanation"] for l in self.logs if l["explanation"] is not None]
        }