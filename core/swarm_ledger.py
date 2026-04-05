# Backward-compat redirect — module moved to _deprecated/swarm_ledger.py
from _deprecated.swarm_ledger import *  # noqa: F401,F403
try:
    from _deprecated.swarm_ledger import (  # noqa: F401
        ledger, SwarmLedger, _LEDGER_PATH,
    )
except ImportError:
    ledger = None
    SwarmLedger = None
    _LEDGER_PATH = None
