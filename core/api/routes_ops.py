"""VRAMancer ops/health/system routes — Flask Blueprint.

Extracted from production_api.py for maintainability.
These routes don't require inference state (_run_with_timeout, queue, etc.).
"""

import os
import time

from flask import Blueprint, jsonify, Response

from core.logger import get_logger

logger = get_logger('api.ops')

ops_bp = Blueprint('ops', __name__)

# Lazy reference to the registry — set by register_ops_blueprint()
_registry_ref = None
_api_port = int(os.environ.get('VRM_API_PORT', '5030'))


def register_ops_blueprint(app, registry):
    """Register the ops blueprint on a Flask app.

    Parameters
    ----------
    app : Flask
        The Flask application.
    registry : PipelineRegistry
        The shared pipeline registry instance.
    """
    global _registry_ref
    _registry_ref = registry
    app.register_blueprint(ops_bp)


# ====================================================================
# Health / readiness / liveness
# ====================================================================

@ops_bp.route('/health')
def health():
    """Health check (Kubernetes/Docker ready)."""
    checks = {'process': True}
    overall = 'healthy'
    try:
        from core.monitor import GPUMonitor  # noqa: F401
        checks['monitor'] = True
    except Exception:
        checks['monitor'] = False
    try:
        from core.config import get_config  # noqa: F401
        checks['config'] = True
    except Exception:
        checks['config'] = False
    try:
        import psutil
        mem = psutil.virtual_memory()
        checks['memory_available_mb'] = int(mem.available / 1024 / 1024)
        if mem.available < 500 * 1024 * 1024:
            overall = 'degraded'
            checks['memory_warning'] = True
    except ImportError:
        checks['memory_available_mb'] = -1
    try:
        from core import __version__
        version = __version__
    except Exception:
        version = '0.2.4'
    checks['pipeline_loaded'] = _registry_ref.is_loaded() if _registry_ref else False
    if not all(checks.get(k, False) for k in ('monitor', 'config')):
        overall = 'degraded'
    return jsonify({
        'status': overall, 'service': 'vramancer-api',
        'version': version, 'checks': checks,
    }), 200


@ops_bp.route('/ready')
def ready():
    """Readiness check."""
    try:
        from core.utils import detect_backend, enumerate_devices
        backend = detect_backend()
        devices = enumerate_devices()
        has_gpu = any(d['backend'] in ('cuda', 'rocm', 'mps') for d in devices)
        return jsonify({
            'status': 'ready', 'backend': backend,
            'devices': len(devices), 'has_gpu': has_gpu,
            'model_loaded': _registry_ref.is_loaded() if _registry_ref else False,
            'service': 'vramancer-api',
        })
    except Exception as e:
        logger.error("Readiness check failed: %s", e)
        return jsonify({'status': 'not_ready', 'error': str(e)}), 503


@ops_bp.route('/live')
def live():
    """Liveness probe."""
    return jsonify({'status': 'alive'}), 200


@ops_bp.route('/metrics')
def metrics_endpoint():
    """Expose Prometheus metrics."""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
    except ImportError:
        return Response("# prometheus_client not installed\n",
                        mimetype="text/plain"), 501


# ====================================================================
# System / GPU / Nodes info
# ====================================================================

@ops_bp.route('/api/status')
def api_status():
    """Detailed API status with all endpoints."""
    try:
        from core import __version__
        version = __version__
    except Exception:
        version = '0.2.4'
    return jsonify({
        'backend': 'running', 'version': version,
        'api_base': os.environ.get('VRM_API_BASE', f'http://localhost:{_api_port}'),
        'endpoints': {
            'generate': 'POST /v1/completions',
            'generate_alt': 'POST /api/generate',
            'chat': 'POST /v1/chat/completions',
            'batch': 'POST /v1/batch/completions',
            'infer': 'POST /api/infer',
            'models': 'GET /api/models',
            'load_model': 'POST /api/models/load',
            'pipeline': 'GET /api/pipeline/status',
            'queue': 'GET /api/queue/status',
            'health': 'GET /health',
            'ready': 'GET /ready',
            'live': 'GET /live',
            'metrics': 'GET /metrics',
            'gpu': 'GET /api/gpu',
            'system': 'GET /api/system',
            'nodes': 'GET /api/nodes',
        },
    })


@ops_bp.route('/api/gpu/info')
@ops_bp.route('/api/gpu')
def gpu_info():
    """GPU information."""
    try:
        import torch
        if not torch.cuda.is_available():
            return jsonify({'cuda_available': False, 'devices': [],
                            'message': 'CUDA not available'})
        devices = []
        for i in range(torch.cuda.device_count()):
            try:
                props = torch.cuda.get_device_properties(i)
                alloc = torch.cuda.memory_allocated(i)
                total = props.total_memory
                devices.append({
                    'id': i, 'name': props.name,
                    'memory_used': alloc, 'memory_total': total,
                    'memory_free': total - alloc,
                    'memory_usage_percent': round(
                        (alloc / total) * 100, 2
                    ) if total else 0,
                    'compute_capability': f"{props.major}.{props.minor}",
                })
            except Exception as e:
                devices.append({'id': i, 'error': str(e)})
        return jsonify({
            'cuda_available': True, 'device_count': len(devices),
            'devices': devices, 'cuda_version': torch.version.cuda,
        })
    except ImportError:
        return jsonify({'cuda_available': False, 'devices': [],
                        'message': 'PyTorch not installed'}), 503
    except Exception as e:
        return jsonify({'error': str(e), 'cuda_available': False}), 500


@ops_bp.route('/api/system')
def system_info():
    """System information."""
    try:
        import psutil
        import platform
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        freq = psutil.cpu_freq()
        return jsonify({
            'platform': platform.system(),
            'architecture': platform.machine(),
            'cpu': {
                'count_logical': psutil.cpu_count(logical=True),
                'count_physical': psutil.cpu_count(logical=False),
                'percent': round(psutil.cpu_percent(interval=0.1), 2),
                'frequency_mhz': round(freq.current) if freq else None,
            },
            'memory': {
                'total_gb': round(mem.total / (1024**3), 2),
                'available_gb': round(mem.available / (1024**3), 2),
                'percent': round(mem.percent, 2),
            },
            'disk': {
                'total_gb': round(disk.total / (1024**3), 2),
                'free_gb': round(disk.free / (1024**3), 2),
                'percent': round(disk.percent, 2),
            },
        })
    except ImportError:
        return jsonify({'error': 'psutil not installed'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@ops_bp.route('/api/nodes')
def nodes_info():
    """Cluster node information."""
    try:
        import psutil
        import platform
        mem = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=0.1)
        uptime_s = time.time() - psutil.boot_time()
        gpu_data = {'name': 'No GPU', 'vram': 0, 'count': 0}
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                gpu_data = {
                    'name': props.name,
                    'vram': round(props.total_memory / (1024**2)),
                    'count': torch.cuda.device_count(),
                }
        except Exception:
            pass
        discovered = []
        try:
            if _registry_ref:
                nodes = _registry_ref.get_nodes()
                for hostname, info in nodes.items():
                    discovered.append({'hostname': hostname, **info})
        except Exception:
            pass
        local_node = {
            'host': 'localhost',
            'name': f'VRAMancer-{platform.node()}',
            'status': 'active', 'role': 'master',
            'os': f"{platform.system()} {platform.release()}",
            'cpu': psutil.cpu_count(logical=True),
            'memory': round(mem.total / (1024**3), 1),
            'vram': gpu_data['vram'], 'gpu_name': gpu_data['name'],
            'gpu_count': gpu_data['count'],
            'ip': '127.0.0.1', 'port': _api_port,
            'uptime': f"{int(uptime_s / 3600)}h {int((uptime_s % 3600) / 60)}m",
            'load': round(cpu_pct, 1),
            'memory_percent': round(mem.percent, 1),
        }
        all_nodes = [local_node] + discovered
        return jsonify({'nodes': all_nodes, 'count': len(all_nodes),
                        'timestamp': time.time()})
    except Exception as e:
        return jsonify({
            'nodes': [{'host': 'localhost', 'status': 'error', 'error': str(e)}],
            'count': 1,
        }), 500
