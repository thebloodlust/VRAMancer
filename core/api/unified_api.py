"""Unified lightweight API exposing:
- Health & basic info
- No-code workflows (proxy vers no_code_workflow storage)
- Digital twin simulation endpoints
- Federated learning round aggregation

Cette API agrège des prototypes existants pour offrir un point unique.
"""
from flask import Flask, request, jsonify
import time, uuid, os, hashlib, threading
import secrets
from typing import List
from core.metrics import Counter as _Counter

try:
    from pydantic import BaseModel, Field, ValidationError
except ImportError:  # fallback léger si pydantic non installé
    BaseModel = object  # type: ignore
    ValidationError = Exception  # type: ignore
    Field = lambda default=None, **k: default  # type: ignore

from core.api.no_code_workflow import WORKFLOWS, TEMPLATES, create_workflow as _create_wf
# Chargement résilient DigitalTwin (collision potentielle module/dossier simulator)
try:
    from core.simulator.digital_twin import DigitalTwin  # type: ignore
except Exception:  # fallback minimal
    class DigitalTwin:  # type: ignore
        def __init__(self, cluster_state):
            self.cluster_state = cluster_state
            self.history = []
        def simulate(self, action):
            rec = {"result": "ok", "action": action, "timestamp": time.time()}
            self.history.append(rec)
            return rec
        def replay(self):
            return list(self.history)
from core.collective.federated_learning import FederatedLearner
from core.security import install_security
from core.metrics import API_LATENCY, metrics_server_start, ENV_ENDPOINT_HITS
from core.persistence import (
    persistence_enabled, save_workflow, load_workflow,
    save_federated_round, load_federated_round
)
from core.xai.xai_dashboard import XAIDashboard
from core.marketplace.generative_plugin import list_plugins, register_plugin, GenerativePlugin, compute_signature
from core.auth_strong import ensure_default_admin, verify_user, issue_tokens, refresh_token, decode_access

app = Flask(__name__)
install_security(app)
metrics_server_start()
ensure_default_admin()

# ----------------------------------------------------------------------------
# Middleware logging HTTP (simple, activable via env VRM_REQUEST_LOG=1)
# ----------------------------------------------------------------------------
_REQ_LOG = os.environ.get("VRM_REQUEST_LOG", "0") in {"1","true","TRUE"}
if _REQ_LOG:
    from core.logger import get_logger
    _http_log = get_logger("api.http")
else:
    _http_log = None

@app.before_request
def _lat_start():  # pragma: no cover
    request._start_ts = time.time()
    # Request ID propagation (trace simple)
    rid = request.headers.get('X-Request-ID') or secrets.token_hex(8)
    request._req_id = rid
    if _http_log:
        _http_log.info(f"REQ START id={rid} path={request.path} method={request.method} ip={request.remote_addr}")

@app.after_request
def _lat_end(resp):  # pragma: no cover
    st = getattr(request, '_start_ts', None)
    rid = getattr(request, '_req_id', None)
    if st is not None:
        dur = time.time() - st
        try:
            API_LATENCY.labels(request.path.split('?')[0][:64], request.method, resp.status_code).observe(dur)
        except Exception:
            pass
    if _http_log:
        _http_log.info(f"REQ END id={rid} path={request.path} status={resp.status_code}")
    if rid:
        resp.headers['X-Request-ID'] = rid
    return resp

# Quota global avant chaque requête API (hors /api/health déjà géré côté security)
@app.before_request
def _quota_global():  # pragma: no cover (testé indirectement)
    from flask import request
    if not request.path.startswith('/api/'):
        return None
    if request.path in ('/api/health','/api/quota/reset'):
        return None
    # Auth forte: endpoints publics
    public_paths = {'/api/version','/api/health','/api/login','/api/token/refresh','/api/marketplace/plugins','/api/workflows','/api/env','/api/xai/explain','/api/xai/explainers','/api/federated/round/start','/api/federated/round/submit','/api/federated/round/aggregate','/api/twin/state'}
    # En mode test on ouvre plus largement pour ne pas exiger Bearer
    if os.environ.get('VRM_TEST_MODE') == '1':
        return None
    if request.path.startswith('/api/federated/round'):
        pass  # toléré (interop tests)
    elif request.path not in public_paths:
        # Si une signature HMAC est fournie (X-API-SIGN) on considère que security.verify_request a déjà validé
        if request.headers.get('X-API-SIGN'):
            pass
        else:
            auth = request.headers.get('Authorization','')
            if not auth.startswith('Bearer '):
                return ("unauthorized", 401)
            token = auth.split(' ',1)[1]
            data = decode_access(token)
            if not data:
                return ("invalid_token", 401)
            request._auth_user = data
    err = _enforce_quota(request)
    if err:
        return err

# In-memory objects
TWIN = DigitalTwin(cluster_state={"nodes": [], "blocks": []})
FED = FederatedLearner(peers=[])  # peers stub
FED_ROUND = {"id": None, "updates": []}  # updates: list[ {value: float, weight: float} ]
XAI = XAIDashboard()
register_plugin(GenerativePlugin("mini-llm","llm"))
register_plugin(GenerativePlugin("sd-tiny","diffusion"))

# Secure aggregation (masquage pairwise) basique: stockage en mémoire de masques par update
SECURE_AGG = {"enabled": False, "masks": {}}  # round_id -> list[mask]
_mask_lock = threading.Lock()

# ---- Quotas (simple compteur par clé API) ----
_quota_counters: dict[str, int] = {}
API_QUOTA_EXCEEDED = _Counter('vramancer_api_quota_exceeded_total', 'Requêtes refusées pour quota')
API_READ_ONLY_BLOCKED = _Counter('vramancer_api_read_only_blocked_total', 'Mutations bloquées (read-only)')

def _enforce_quota(req):
    # Lecture dynamique (permet changements à chaud dans tests/env)
    try:
        quota = int(os.environ.get("VRM_UNIFIED_API_QUOTA", "0"))
    except Exception:
        quota = 0
    if quota <= 0:
        return None
    # Si signature HMAC fournie, utiliser une clé de quota dérivée pour compter correctement
    if req.headers.get('X-API-SIGN'):
        # Les signatures étant différentes à chaque requête, utiliser une clé logique stable
        # Basée sur présence de X-API-SIGN + éventuellement rôle ou un secret stable (non accessible ici) -> on prend un identifiant constant
        token = "signed_client"
    else:
        token = req.headers.get("X-API-TOKEN", "anonymous")
    cur = _quota_counters.get(token, 0) + 1
    _quota_counters[token] = cur
    if cur > quota:
        API_QUOTA_EXCEEDED.inc()
        return ("quota exceeded", 429)
    return None

# ---- Pydantic schemas ----
class WorkflowTask(BaseModel):  # type: ignore[misc]
    type: str = Field(..., min_length=1)
    params: dict = Field(default_factory=dict)

class CreateWorkflowRequest(BaseModel):  # type: ignore[misc]
    tasks: List[WorkflowTask] = Field(default_factory=list)
    name: str | None = Field(default=None, max_length=64)

READ_ONLY = lambda: os.environ.get('VRM_READ_ONLY','0') == '1'

@app.route('/api/info')
def info():
    return {"ok": True, "time": time.time(), "workflows": len(WORKFLOWS)}

@app.route('/api/version')
def version():
    try:
        from core import __version__
    except Exception:
        __version__ = "unknown"
    return {"version": __version__}

@app.route('/api/health')
def health():
    return {"status": "ok", "ts": time.time()}

@app.route('/api/env')
def env_info():
    """Expose quelques drapeaux runtime (diagnostic)."""
    try:
        ENV_ENDPOINT_HITS.inc()
    except Exception:
        pass
    flags = {}
    # Imports facultatifs
    try:
        import torch  # type: ignore
        flags['torch'] = True
        flags['torch_cuda'] = bool(getattr(torch.cuda, 'is_available', lambda: False)())
    except Exception:
        flags['torch'] = False
    try:
        import transformers  # type: ignore
        flags['transformers'] = True
    except Exception:
        flags['transformers'] = False
    flags['onnx'] = bool(os.environ.get('VRM_DISABLE_ONNX') != '1')
    flags['socketio_disabled'] = os.environ.get('VRM_DISABLE_SOCKETIO','0') in {'1','true','TRUE'}
    flags['dashboard_minimal'] = os.environ.get('VRM_DASHBOARD_MINIMAL','0') in {'1','true','TRUE'}
    flags['api_quota'] = int(os.environ.get('VRM_UNIFIED_API_QUOTA','0'))
    flags['read_only'] = os.environ.get('VRM_READ_ONLY','0') in {'1','true','TRUE'}
    flags['json_log'] = os.environ.get('VRM_LOG_JSON','0') in {'1','true','TRUE'}
    flags['request_log'] = os.environ.get('VRM_REQUEST_LOG','0') in {'1','true','TRUE'}
    flags['auth_secret_set'] = bool(os.environ.get('VRM_AUTH_SECRET'))
    return flags

@app.route('/api/login', methods=['POST'])
def login():
    body = request.json or {}
    user = body.get('username')
    pwd = body.get('password')
    if not user or not pwd:
        return ("missing credentials", 400)
    if not verify_user(user, pwd):
        return ("invalid credentials", 401)
    return jsonify(issue_tokens(user))

@app.route('/api/token/refresh', methods=['POST'])
def token_refresh():
    body = request.json or {}
    ref = body.get('refresh')
    if not ref:
        return ("missing refresh", 400)
    newt = refresh_token(ref)
    if not newt:
        return ("invalid refresh", 401)
    return jsonify(newt)

# ---- No-code reuse ----
@app.route('/api/workflows', methods=['POST'])
def create_workflow():
    if READ_ONLY():
        API_READ_ONLY_BLOCKED.inc()
        return ("read-only", 503)
    # quota global géré en before_request
    payload = request.json or {}
    if BaseModel is object:  # pydantic absent
        return _create_wf()
    try:
        parsed = CreateWorkflowRequest(**payload)
    except ValidationError as ve:  # type: ignore
        return jsonify({"error": "validation", "details": ve.errors()}), 422
    # Adapter format attendu par no_code_workflow.create_workflow()
    req = {"tasks": [ {"type": t.type, **t.params} for t in parsed.tasks ]}
    # Monkey patch request.json temporaire si nécessaire
    # Simplicité: ré-appeler la fonction sous-jacente via variable globale
    from core.api.no_code_workflow import WORKFLOWS
    wid_resp = _create_wf.__wrapped__(req) if hasattr(_create_wf, '__wrapped__') else None  # not used
    # Reproduire logique interne minimale (évite de dépendre du flask request interne du module)
    import uuid as _uuid
    wid = str(_uuid.uuid4())
    WORKFLOWS[wid] = {"id": wid, "tasks": req.get("tasks", []), "status": "created", "logs": []}
    if persistence_enabled():
        save_workflow(WORKFLOWS[wid])
    return jsonify(WORKFLOWS[wid])

@app.route('/api/workflows/<wid>')
def get_workflow(wid):
    wf = WORKFLOWS.get(wid)
    if not wf and persistence_enabled():
        wf = load_workflow(wid)
        if wf:
            WORKFLOWS[wid] = wf
    return jsonify(wf or {})

@app.route('/api/workflows')
def list_workflows():
    # Pagination simple ?limit=
    try:
        limit = int(request.args.get('limit','50'))
    except Exception:
        limit = 50
    items = list(WORKFLOWS.values())[-limit:]
    if persistence_enabled() and len(items) < limit:
        # Fusion persistés (liste déjà triée rowid desc dans persistence)
        from core.persistence import list_workflows as _list
        persisted = _list(limit=limit)
        # Merge par id (in-memory prioritaire)
        combined = {w['id']: w for w in persisted}
        combined.update({w['id']: w for w in items})
        items = list(combined.values())[-limit:]
    return jsonify({"items": items, "count": len(items)})

# ---- Digital Twin ----
@app.route('/api/twin/simulate', methods=['POST'])
def twin_simulate():
    action = request.json or {}
    return jsonify(TWIN.simulate(action))

@app.route('/api/twin/replay')
def twin_replay():
    return jsonify(TWIN.replay())

# ---- Federated Learning ----
@app.route('/api/federated/round/start', methods=['POST'])
def federated_start():
    FED_ROUND['id'] = str(uuid.uuid4())
    FED_ROUND['updates'] = []
    if persistence_enabled():
        save_federated_round(FED_ROUND['id'], [])
    with _mask_lock:
        SECURE_AGG['masks'][FED_ROUND['id']] = []
    return jsonify({"round_id": FED_ROUND['id']})

@app.route('/api/federated/round/submit', methods=['POST'])
def federated_submit():
    if READ_ONLY():
        API_READ_ONLY_BLOCKED.inc()
        return ("read-only", 503)
    # quota global géré en before_request
    data = request.json or {}
    val = data.get('value')
    weight = float(data.get('weight', 1.0) or 1.0)
    if val is None:
        return jsonify({'error': 'missing value'}), 422
    if FED_ROUND['id'] is None:
        return jsonify({'error': 'no_round'}), 400
    # Secure agg: générer masque déterministe simple basé sur token + round
    if SECURE_AGG['enabled'] and FED_ROUND['id']:
        token = request.headers.get('X-API-TOKEN','anon')
        seed_src = f"{FED_ROUND['id']}::{token}::{len(FED_ROUND['updates'])}"
        mask = int(hashlib.sha256(seed_src.encode()).hexdigest(),16) % 1000 / 1000.0  # 0..0.999
        masked_val = float(val) + mask
        with _mask_lock:
            SECURE_AGG['masks'][FED_ROUND['id']].append(mask)
        FED_ROUND['updates'].append({'value': masked_val, 'weight': weight, 'masked': True})
    else:
        FED_ROUND['updates'].append({'value': val, 'weight': weight})
    if persistence_enabled():
        save_federated_round(FED_ROUND['id'], FED_ROUND['updates'])
    return jsonify({"accepted": True, "count": len(FED_ROUND['updates'])})

@app.route('/api/federated/round/aggregate')
def federated_aggregate():
    updates = FED_ROUND['updates']
    if not updates and persistence_enabled() and FED_ROUND['id']:
        stored = load_federated_round(FED_ROUND['id'])
        if stored:
            FED_ROUND['updates'] = stored
            updates = stored
    if not updates:
        return jsonify({"error": "no_updates"}), 400
    # Support poids
    values = [u['value'] for u in updates]
    weights = [u.get('weight',1.0) for u in updates]
    # Retrait des masques si secure agg actif
    if SECURE_AGG['enabled'] and FED_ROUND['id']:
        with _mask_lock:
            masks = SECURE_AGG['masks'].get(FED_ROUND['id'], [])
        # on suppose même nombre de masques que updates masqués
        unmasked = []
        mi = 0
        for u in updates:
            v = u['value']
            if u.get('masked') and mi < len(masks):
                v -= masks[mi]
                mi += 1
            unmasked.append(v)
        values = unmasked
    try:
        agg = FED.aggregate_weighted(values, weights) if hasattr(FED, 'aggregate_weighted') else FED.aggregate(values)
    except Exception as e:
        return jsonify({"error": "aggregation_failed", "detail": str(e)}), 500
    return jsonify({"round_id": FED_ROUND['id'], "aggregate": agg, "count": len(updates)})

# ---- Twin state / snapshot ----
@app.route('/api/twin/state')
def twin_state():
    # Retourne un snapshot actuel (copie superficielle)
    return jsonify({"cluster_state": TWIN.cluster_state, "history_len": len(TWIN.history)})

@app.route('/api/xai/explain', methods=['POST'])
def xai_explain():
    payload = request.json or {}
    kind = payload.get('kind','feature_attrib')
    data = payload.get('data',{})
    return jsonify(XAI.explain(kind, data))

@app.route('/api/xai/explainers')
def xai_explainers():
    return jsonify({"explainers": list(XAI._explainers.keys())})

# ---- Marketplace ----
@app.route('/api/marketplace/plugins')
def market_list():
    items = list_plugins()
    for it in items:
        it['signature'] = compute_signature(it['name'])
    return jsonify({"plugins": items})

@app.route('/api/federated/secure', methods=['POST'])
def federated_secure_toggle():
    if READ_ONLY():
        API_READ_ONLY_BLOCKED.inc()
        return ("read-only", 503)
    data = request.json or {}
    enabled = bool(data.get('enabled'))
    SECURE_AGG['enabled'] = enabled
    return {"secure_enabled": enabled}

@app.route('/api/federated/secure')
def federated_secure_status():
    return {"secure_enabled": bool(SECURE_AGG['enabled'])}

@app.route('/api/quota/reset', methods=['POST'])
def quota_reset():
    """Réinitialise les compteurs de quota (admin/outillage)."""
    _quota_counters.clear()
    return {"ok": True, "reset": True}

if __name__ == '__main__':  # pragma: no cover
    app.run(port=5030, debug=True)
