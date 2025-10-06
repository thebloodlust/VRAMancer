"""
API et protocole custom VRAMancer pour supervision avancée des nœuds.
- État en ligne/hors ligne, type, icône, CPU, RAM, GPU, OS, connexion (USB4, Ethernet, WiFi)
- Historique, alertes, actions distantes
- REST + WebSocket
"""
from flask import Flask, jsonify, request, Response
import os
try:
    from flask_socketio import SocketIO, emit
except Exception:  # fallback tests minimal
    class _DummySock:
        def __init__(self, app,*a,**k): self.app=app
        def on(self,*a,**k):
            def deco(f): return f
            return deco
        def run(self, app, port=0, debug=False):
            app.run(port=port, debug=debug)
    def emit(*a,**k): pass
    SocketIO = _DummySock  # type: ignore
import time
import random
import psutil
from core.security import install_security
from core.telemetry import encode_stream, format_text_line, decode_stream
from core.task_scheduler import get_global_scheduler
from core.metrics import (
    TELEMETRY_PACKETS,
    publish_device_info,
    publish_task_percentiles,
    API_LATENCY,
    FASTPATH_IF_LATENCY,
    HA_JOURNAL_ROTATIONS,
    HA_JOURNAL_SIZE,
)
from core.utils import enumerate_devices
from core.tracing import start_tracing
from core.hierarchical_memory import HierarchicalMemoryManager
from core.network.fibre_fastpath import (
    open_low_latency_channel,
    detect_fast_interfaces,
    benchmark_interfaces,
)
from core.ha_replication import start_replication_loop, _journal_append, _register_nonce, _derive_secret
import zlib, json, hmac, hashlib, base64

# Démarrage tracing si demandé
start_tracing()

app = Flask(__name__)

@app.before_request
def _api_timer_start():  # pragma: no cover - mesure
    from flask import g, request
    g._start_ts = time.perf_counter()

@app.after_request
def _api_timer_stop(resp):  # pragma: no cover - mesure
    from flask import g, request
    st = getattr(g, '_start_ts', None)
    if st is not None and request.path.startswith('/api/'):
        dur = time.perf_counter() - st
        try:
            API_LATENCY.labels(request.path, request.method, str(resp.status_code)).observe(dur)
        except Exception:
            pass
    return resp
socketio = SocketIO(app, cors_allowed_origins="*")
install_security(app)
scheduler = get_global_scheduler()
HMM = HierarchicalMemoryManager()  # instance locale pour exposition API (lot B)
start_replication_loop(HMM)
if not os.environ.get('VRM_MINIMAL_TEST'):
    try:
        publish_device_info(enumerate_devices())
    except Exception:  # pragma: no cover - environnement minimal sans torch
        pass

NODES = [
    {"id": "raspberrypi", "type": "edge", "icon": "raspberry.svg", "status": "online", "cpu": "ARM", "ram": 1024, "gpu": "None", "os": "Linux", "conn": "WiFi", "usb4": False, "last_seen": time.time()},
    {"id": "jetson", "type": "edge", "icon": "jetson.svg", "status": "offline", "cpu": "ARM", "ram": 4096, "gpu": "Nvidia", "os": "Linux", "conn": "Ethernet", "usb4": True, "last_seen": time.time()-3600},
    {"id": "workstation", "type": "standard", "icon": "standard.svg", "status": "online", "cpu": "x86_64", "ram": 32768, "gpu": "RTX 4090", "os": "Windows", "conn": "USB4", "usb4": True, "last_seen": time.time()},
]

HISTORY = []
_HEARTBEAT_TIMEOUT = 30  # secondes

@socketio.on('heartbeat')
def heartbeat(msg):  # pragma: no cover - I/O simple
    nid = msg.get('id')
    node = next((n for n in NODES if n['id']==nid), None)
    if node:
        node['last_seen'] = time.time()
        emit('heartbeat_ack', {'id': nid})

def _purge_inactive():  # thread minimal
    while True:  # pragma: no cover - boucle
        now = time.time()
        for n in NODES:
            if n.get('status') == 'online' and (now - n.get('last_seen', now)) > _HEARTBEAT_TIMEOUT:
                n['status'] = 'offline'
        time.sleep(5)

import threading as _thr
_thr.Thread(target=_purge_inactive, daemon=True).start()

@app.route("/api/nodes")
def get_nodes():
    # Mise à jour dynamique CPU load & free cores
    for n in NODES:
        if n["status"] == "online":
            n["cpu_load_pct"] = psutil.cpu_percent(interval=0.0)
            n["free_cores"] = max(0, psutil.cpu_count(logical=True) - psutil.cpu_count(logical=False))
    return jsonify(NODES)

@app.route("/api/nodes/<node_id>")
def get_node(node_id):
    node = next((n for n in NODES if n["id"] == node_id), None)
    return jsonify(node or {})

@app.route("/api/nodes/<node_id>/action", methods=["POST"])
def node_action(node_id):
    action = request.json.get("action")
    HISTORY.append({"node": node_id, "action": action, "timestamp": time.time()})
    # Simuler une action distante
    return jsonify({"ok": True, "action": action})

@app.route("/api/edge/report", methods=["POST"])
def edge_report():
    data = request.json or {}
    nid = data.get("id")
    load = data.get("cpu_load")
    free = data.get("free_cores")
    # Enregistrement / mise à jour
    node = next((n for n in NODES if n['id']==nid), None)
    if node:
        node["cpu_load_pct"] = load
        node["free_cores"] = free
        node["last_seen"] = time.time()
    else:
        NODES.append({"id": nid, "type": data.get("type","edge"), "icon": data.get("icon","edge.svg"),
                      "status": "online", "cpu": data.get("cpu","?"), "ram": data.get("ram",0), "gpu": data.get("gpu","?"),
                      "os": data.get("os","?"), "conn": data.get("conn","?"), "usb4": data.get("usb4", False),
                      "cpu_load_pct": load, "free_cores": free, "last_seen": time.time()})
    return jsonify({"ok": True})

@app.route("/api/history")
def get_history():
    return jsonify(HISTORY)

@app.route("/api/tasks/submit", methods=["POST"])
def submit_task():
    data = request.json or {}
    kind = data.get("kind", "noop")
    priority = data.get("priority", "NORMAL")
    # Définir quelques tâches symboliques
    if kind == "warmup":
        def fn():
            import torch
            x = torch.randn(2048,2048)
            (x @ x).sum().item()
        scheduler.submit(fn, priority=priority, tags=[kind])
    elif kind == "compress":
        def fn():
            import zlib, os
            payload = os.urandom(2_000_000)
            zlib.compress(payload)
        scheduler.submit(fn, priority=priority, tags=[kind])
    else:  # noop
        scheduler.submit(lambda: time.sleep(0.2), priority=priority, tags=[kind])
    return jsonify({"ok": True, "queued": kind, "priority": priority})

@app.route("/api/tasks/status")
def tasks_status():
    act = {rid: {"priority": t.priority, "id": t.id} for rid, t in scheduler.active.items()}
    return jsonify({"active": act, "queue_size": scheduler.tasks.qsize()})

@app.route("/api/tasks/cancel", methods=["POST"])
def tasks_cancel():
    tid = (request.json or {}).get('id')
    if not tid:
        return jsonify({'error':'missing id'}), 400
    ok = scheduler.cancel(tid)
    return jsonify({'ok': ok})

@app.route("/api/tasks/submit_batch", methods=["POST"])
def tasks_submit_batch():
    payload = request.json or {}
    kinds = payload.get('tasks', [])
    mapping = {
        'warmup': lambda: (lambda: ( (__import__('torch').randn(512,512) @ __import__('torch').randn(512,512)).sum().item() )),
        'compress': lambda: (lambda: (__import__('zlib').compress(__import__('os').urandom(500000)))) ,
        'noop': lambda: (lambda: __import__('time').sleep(0.05)),
    }
    batch = []
    for spec in kinds:
        kind = spec.get('kind','noop')
        pr   = spec.get('priority','NORMAL')
        est  = spec.get('est_runtime_s',0.0)
        fn = mapping.get(kind, mapping['noop'])()
        batch.append({'fn': fn, 'priority': pr, 'tags':[kind], 'est_runtime_s': est})
    ids = scheduler.submit_batch(batch)
    return jsonify({'ok': True, 'ids': ids})

@app.route("/api/tasks/history")
def tasks_history():
    return jsonify(scheduler.history[-200:])

@app.route("/api/tasks/metrics")
def tasks_metrics():
    pct = scheduler.compute_percentiles()
    publish_task_percentiles(pct)
    return jsonify(pct)

@app.route("/api/memory/evict", methods=["POST"])
def memory_evict_cycle():
    payload = request.json or {}
    vram_pressure = payload.get('vram_pressure')  # float entre 0 et 1
    res = HMM.eviction_cycle(vram_pressure=vram_pressure)
    return jsonify({"evicted": res, "count": len(res), "vram_pressure": vram_pressure})

@app.route('/api/memory/summary')
def memory_summary():
    return jsonify(HMM.summary())

@app.route('/api/fastpath/capabilities')
def fastpath_caps():
    ch = open_low_latency_channel()
    try:
        interfaces = detect_fast_interfaces()
    except Exception:
        interfaces = []
    selected = interfaces[0] if interfaces else None
    caps = ch.capabilities()
    caps.update({'interfaces': interfaces, 'selected': selected})
    return jsonify(caps)

@app.route('/api/fastpath/interfaces')
def fastpath_interfaces():
    """Liste les interfaces détectées + benchmark rapide simulé.
    Publie les latences dans la Gauge FASTPATH_IF_LATENCY.
    """
    interfaces = detect_fast_interfaces()
    benches = benchmark_interfaces()
    for b in benches:
        iface = b.get('interface')
        kind = b.get('kind')
        lat = b.get('latency_s', 0.0)
        try:
            FASTPATH_IF_LATENCY.labels(iface, kind).set(lat)
        except Exception:
            pass
    selected = interfaces[0] if interfaces else None
    return jsonify({
        'interfaces': interfaces,
        'benchmarks': benches,
        'selected': selected,
    })

@app.route('/api/fastpath/select', methods=['POST'])
def fastpath_select():
    """Sélectionne (priorise) une interface fastpath via variable d'env.
    Payload: {"interface": "eth0"} ou {"kind": "usb4"}
    """
    data = request.json or {}
    name = data.get('interface') or data.get('kind')
    if not name:
        return jsonify({'error': 'missing interface'}), 400
    os.environ['VRM_FASTPATH_IF'] = name
    interfaces = detect_fast_interfaces()
    # Auto re-benchmark après sélection pour feedback immédiat
    benches = benchmark_interfaces()
    for b in benches:
        iface = b.get('interface')
        kind = b.get('kind')
        lat = b.get('latency_s', 0.0)
        try:
            FASTPATH_IF_LATENCY.labels(iface, kind).set(lat)
        except Exception:
            pass
    selected = interfaces[0] if interfaces else None
    return jsonify({'ok': True, 'selected': selected, 'interfaces': interfaces, 'benchmarks': benches})

@app.route('/api/ha/apply', methods=['POST'])
def ha_apply():
    meta_b64 = request.headers.get('X-HA-META')
    secret = os.environ.get('VRM_HA_SECRET')
    added = 0
    if not meta_b64:
        return jsonify({'error':'missing meta'}), 400
    try:
        meta = json.loads(base64.b64decode(meta_b64).decode())
    except Exception:
        return jsonify({'error':'bad meta'}), 400
    payload = request.get_data() or b''
    ts = meta.get('ts')
    nonce = meta.get('nonce')
    if secret:
        if ts is None or nonce is None:
            return jsonify({'error':'missing ts/nonce'}), 400
        # Anti-rejeu fenêtre
        if not _register_nonce(nonce, ts):
            return jsonify({'error':'replay'}), 409
        derived = _derive_secret(secret, ts)
        base = f"{int(ts)}:{nonce}:{meta.get('hash','')}".encode() + payload
        calc = hmac.new(derived.encode(), base, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(calc, meta.get('sig','')):
            return jsonify({'error':'bad sig'}), 401
    # Interprétation payload
    if payload:
        try:
            if meta.get('compressed'):
                algo = meta.get('algo','zlib')
                if algo == 'zstd':
                    import zstandard as zstd  # type: ignore
                    d = zstd.ZstdDecompressor()
                    raw = d.decompress(payload)
                elif algo == 'lz4':
                    import lz4.frame as lz4f  # type: ignore
                    raw = lz4f.decompress(payload)
                elif algo == 'zlib':
                    raw = zlib.decompress(payload)
                else:
                    return jsonify({'error':'unknown comp algo'}), 400
            else:
                raw = payload
            obj = json.loads(raw)
        except Exception:
            return jsonify({'error':'decode fail'}), 400
        if obj.get('full'):
            reg = obj['state']['registry']
            for bid, m in reg.items():
                if bid not in HMM.registry:
                    HMM.registry[bid] = {
                        'tier': m.get('tier','L3'),
                        'size_mb': m.get('size',0),
                        'ts': time.time(),
                        'access': m.get('acc',0),
                        'last_access': None,
                        'meta': {},
                    }
                    added += 1
        else:
            # delta structurel
            for bid, m in obj.get('add', {}).items():
                if bid not in HMM.registry:
                    HMM.registry[bid] = {
                        'tier': m.get('tier','L3'),
                        'size_mb': m.get('size',0),
                        'ts': time.time(),
                        'access': m.get('acc',0),
                        'last_access': None,
                        'meta': {},
                    }
                    added += 1
            for bid in obj.get('remove', []):
                HMM.registry.pop(bid, None)
    _journal_append({"dir":"in","hash": meta.get('hash'), "delta": meta.get('delta'), "added": added})
    # Mettre à jour métrique taille journal si présent
    jpath = os.environ.get('VRM_HA_JOURNAL')
    if jpath and os.path.exists(jpath):
        try:
            HA_JOURNAL_SIZE.set(os.path.getsize(jpath))
        except Exception:
            pass
    # Rotation journal (taille max en octets)
    max_size = int(os.environ.get('VRM_HA_JOURNAL_MAX','5242880'))  # 5MB
    if jpath and os.path.exists(jpath) and os.path.getsize(jpath) > max_size:
        # compresser archive et repartir sur nouveau
        import gzip, shutil
        ts = int(time.time())
        gz_path = f"{jpath}.{ts}.gz"
        with open(jpath,'rb') as fin, gzip.open(gz_path,'wb') as fout:
            shutil.copyfileobj(fin,fout)
        with open(jpath,'w') as f:  # reset
            f.write('')
        try:
            HA_JOURNAL_ROTATIONS.inc()
            HA_JOURNAL_SIZE.set(0)
        except Exception:
            pass
    # Politique cluster globale: si trop de blocs L1 cumulés > seuil, éviction locale
    l1_blocks = sum(1 for v in HMM.registry.values() if v['tier']=='L1')
    if l1_blocks > int(os.environ.get('VRM_CLUSTER_L1_THRESHOLD','500')):
        HMM.eviction_cycle(vram_pressure=0.91)
    return jsonify({'ok': True, 'added': added, 'delta': meta.get('delta')})

@app.route('/api/tasks/estimator/install', methods=['POST'])
def tasks_estimator_install():
    """Installe dynamiquement un estimator très simple basé sur tags.
    Exemple de payload: {"map": {"warmup": 4.0, "compress": 2.0}}
    """
    payload = request.json or {}
    mapping = payload.get('map', {})
    def estimator(task):  # pragma: no cover - simple logique
        for t in task.tags:
            if t in mapping:
                return mapping[t]
        return 1.0
    scheduler.runtime_estimator = estimator
    return jsonify({"ok": True, "installed": True})

@app.route("/api/tasks/estimator/example")
def task_estimator_example():
    return {"hint": "Fournir une fonction runtime_estimator= lambda task: est_secs"}

@app.route("/api/devices")
def api_devices():
    devs = enumerate_devices()
    # republier (idempotent) – support hot-plug futur
    publish_device_info(devs)
    return jsonify(devs)

@app.route("/api/telemetry.bin")
def telemetry_bin():
    # Flux binaire concaténé
    payload = encode_stream(NODES)
    TELEMETRY_PACKETS.labels('out').inc()
    return Response(payload, mimetype="application/octet-stream")

@app.route("/api/telemetry.txt")
def telemetry_txt():
    lines = [format_text_line(n) for n in NODES]
    TELEMETRY_PACKETS.labels('out').inc()
    return Response("\n".join(lines)+"\n", mimetype="text/plain")

@app.route("/api/telemetry/stream")
def telemetry_stream():
    def gen():
        while True:
            lines = [format_text_line(n) for n in NODES]
            TELEMETRY_PACKETS.labels('out').inc()
            yield "data: " + "|".join(lines) + "\n\n"
            time.sleep(2)
    return Response(gen(), mimetype='text/event-stream')

@app.route('/api/telemetry/multicast')
def telemetry_multicast():
    """Envoie un datagramme UDP multicast (lot D) avec l'état condensé.
    Variables d'env: VRM_MC_ADDR (default 239.12.12.12), VRM_MC_PORT (default 45123)
    """
    import socket, os, json
    group = os.environ.get('VRM_MC_ADDR', '239.12.12.12')
    port = int(os.environ.get('VRM_MC_PORT', '45123'))
    state = [{k: n.get(k) for k in ('id','cpu_load_pct','free_cores','last_seen')} for n in NODES]
    payload = json.dumps(state, separators=(',',':')).encode()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    try:
        ttl = 1
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)
        sock.sendto(payload, (group, port))
    finally:
        sock.close()
    TELEMETRY_PACKETS.labels('out').inc()
    return jsonify({'ok': True, 'bytes': len(payload), 'group': group, 'port': port})

@app.route('/api/telemetry/ingest', methods=['POST'])
def telemetry_ingest():
    # Ingestion d'un paquet binaire (edge → concentrateur)
    blob = request.get_data() or b''
    if not blob:
        return jsonify({'error':'empty'}), 400
    TELEMETRY_PACKETS.labels('in').inc()
    # Parse & merge
    try:
        for entry in decode_stream(blob):
            node = next((n for n in NODES if n['id']==entry['id']), None)
            if node:
                node['cpu_load_pct'] = entry.get('cpu_load_pct')
                node['free_cores'] = entry.get('free_cores')
                node['last_seen'] = time.time()
            else:
                NODES.append({
                    'id': entry['id'], 'type':'edge', 'icon':'edge.svg', 'status':'online',
                    'cpu': 'unknown', 'ram': 0, 'gpu': '?', 'os': 'Unknown', 'conn': 'unknown', 'usb4': False,
                    'cpu_load_pct': entry.get('cpu_load_pct'), 'free_cores': entry.get('free_cores'), 'last_seen': time.time()
                })
    except Exception:
        pass
    return jsonify({'ok': True, 'bytes': len(blob), 'merged': True})

@socketio.on("subscribe")
def handle_subscribe(data):
    emit("nodes", NODES)

@socketio.on("ping")
def handle_ping(data):
    node_id = data.get("node_id")
    # Simuler un ping
    emit("pong", {"node_id": node_id, "status": random.choice(["online", "offline"])})

if __name__ == "__main__":
    port = int(os.environ.get('VRM_API_PORT','5010'))
    socketio.run(app, port=port, debug=True)
