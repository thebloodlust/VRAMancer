# DEPRECATED: Ce fichier est un doublon obsolète.
# Le vrai code vit dans core/monitor.py — ce fichier sera supprimé.
# Re-export pour compatibilité.
try:
    from core.monitor import GPUMonitor  # noqa: F401
except ImportError:
    class GPUMonitor:  # pragma: no cover
        pass

# ---- Ancien code supprimé (98 lignes) ----
# Voir core/monitor.py pour la version canonique.
_DEPRECATED = True

