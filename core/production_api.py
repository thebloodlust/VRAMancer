#!/usr/bin/env python3
"""
VRAMancer API - Version Production
Remplace api_simple.py avec logging structuré, validation, et error handling robuste
"""
import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, jsonify, request
from core.logger import get_logger
from typing import Dict, Any, Optional

# Logger structuré
logger = get_logger('api.production')

# Configuration depuis environnement
API_HOST = os.environ.get('VRM_API_HOST', '0.0.0.0')
API_PORT = int(os.environ.get('VRM_API_PORT', '5030'))
API_DEBUG = os.environ.get('VRM_API_DEBUG', '0') in {'1', 'true', 'TRUE'}

# Définir la base URL
os.environ.setdefault('VRM_API_BASE', f'http://localhost:{API_PORT}')

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


# ============================================================================
# Middleware et error handlers
# ============================================================================

@app.before_request
def log_request():
    """Log toutes les requêtes entrantes"""
    if API_DEBUG:
        logger.debug(
            f"Request: {request.method} {request.path}",
            extra={'context': {
                'method': request.method,
                'path': request.path,
                'remote_addr': request.remote_addr,
                'user_agent': request.headers.get('User-Agent', 'Unknown')
            }}
        )


@app.after_request
def log_response(response):
    """Log toutes les réponses"""
    if API_DEBUG:
        logger.debug(
            f"Response: {response.status_code} for {request.path}",
            extra={'context': {
                'status_code': response.status_code,
                'path': request.path
            }}
        )
    return response


@app.errorhandler(404)
def not_found(error):
    """Handler pour erreurs 404"""
    logger.warning(f"404 Not Found: {request.path}")
    return jsonify({
        'error': 'Not Found',
        'message': f'Endpoint {request.path} does not exist',
        'status': 404
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handler pour erreurs 500"""
    logger.error(f"500 Internal Error: {error}", exc_info=True)
    return jsonify({
        'error': 'Internal Server Error',
        'message': 'An unexpected error occurred',
        'status': 500
    }), 500


# ============================================================================
# Endpoints de santé et status
# ============================================================================

@app.route('/health')
def health():
    """Endpoint de health check (Kubernetes/Docker ready)"""
    return jsonify({
        'status': 'healthy',
        'service': 'vramancer-api',
        'version': '1.0.0'
    })


@app.route('/ready')
def ready():
    """Endpoint de readiness check"""
    # Vérifier que les services essentiels sont disponibles
    try:
        # Test basique : import torch
        import torch
        cuda_available = torch.cuda.is_available()
        
        return jsonify({
            'status': 'ready',
            'cuda_available': cuda_available,
            'service': 'vramancer-api'
        })
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return jsonify({
            'status': 'not_ready',
            'error': str(e)
        }), 503


@app.route('/api/status')
def status():
    """Status détaillé de l'API"""
    logger.info("Status check requested")
    
    try:
        return jsonify({
            'backend': 'running',
            'version': '1.0.0',
            'api_base': os.environ.get('VRM_API_BASE', f'http://localhost:{API_PORT}'),
            'endpoints': {
                'health': '/health',
                'ready': '/ready',
                'gpu': '/api/gpu',
                'system': '/api/system',
                'nodes': '/api/nodes'
            }
        })
    except Exception as e:
        logger.error(f"Status endpoint error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Endpoints GPU
# ============================================================================

@app.route('/api/gpu/info')
@app.route('/api/gpu')
def gpu_info():
    """Informations GPU détaillées"""
    logger.debug("GPU info requested")
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available")
            return jsonify({
                'cuda_available': False,
                'devices': [],
                'message': 'CUDA not available',
                'backend': 'CPU-only'
            })
        
        devices = []
        for i in range(torch.cuda.device_count()):
            try:
                device_name = torch.cuda.get_device_name(i)
                device_props = torch.cuda.get_device_properties(i)
                
                # Initialiser CUDA si nécessaire
                if not torch.cuda.is_initialized():
                    torch.cuda.init()
                
                memory_allocated = torch.cuda.memory_allocated(i)
                memory_reserved = torch.cuda.memory_reserved(i)
                memory_total = device_props.total_memory
                
                device_info = {
                    'id': i,
                    'name': device_name,
                    'backend': 'PyTorch CUDA',
                    'memory_used': memory_allocated,
                    'memory_total': memory_total,
                    'memory_reserved': memory_reserved,
                    'memory_free': memory_total - memory_allocated,
                    'memory_usage_percent': round((memory_allocated / memory_total) * 100, 2) if memory_total > 0 else 0,
                    'compute_capability': f"{device_props.major}.{device_props.minor}",
                    'multi_processor_count': device_props.multi_processor_count
                }
                
                devices.append(device_info)
                logger.debug(f"GPU {i}: {device_name} - {memory_allocated / (1024**3):.2f}GB used")
                
            except Exception as e:
                logger.error(f"Error getting info for GPU {i}: {e}", exc_info=True)
                devices.append({
                    'id': i,
                    'error': str(e),
                    'status': 'error'
                })
        
        return jsonify({
            'cuda_available': True,
            'device_count': len(devices),
            'devices': devices,
            'cuda_version': torch.version.cuda
        })
        
    except ImportError:
        logger.error("PyTorch not installed")
        return jsonify({
            'cuda_available': False,
            'devices': [],
            'message': 'PyTorch not installed',
            'error': 'ImportError'
        }), 503
    except Exception as e:
        logger.error(f"GPU endpoint error: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'cuda_available': False
        }), 500


# ============================================================================
# Endpoints System
# ============================================================================

@app.route('/api/system')
def system_info():
    """Informations système"""
    logger.debug("System info requested")
    
    try:
        import psutil
        import platform
        
        cpu_count = psutil.cpu_count(logical=True)
        cpu_count_physical = psutil.cpu_count(logical=False)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return jsonify({
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'cpu': {
                'count_logical': cpu_count,
                'count_physical': cpu_count_physical,
                'percent': round(cpu_percent, 2),
                'frequency_mhz': round(cpu_freq.current) if cpu_freq else None
            },
            'memory': {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'percent': round(memory.percent, 2)
            },
            'disk': {
                'total_gb': round(disk.total / (1024**3), 2),
                'used_gb': round(disk.used / (1024**3), 2),
                'free_gb': round(disk.free / (1024**3), 2),
                'percent': round(disk.percent, 2)
            },
            'backend': 'VRAMancer Production'
        })
        
    except ImportError:
        logger.error("psutil not installed")
        return jsonify({
            'error': 'psutil not installed',
            'platform': 'Unknown',
            'backend': 'Limited'
        }), 503
    except Exception as e:
        logger.error(f"System endpoint error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ============================================================================
# Endpoints Nodes (clustering)
# ============================================================================

@app.route('/api/nodes')
def nodes_info():
    """Informations des nodes du cluster"""
    logger.debug("Nodes info requested")
    
    try:
        import psutil
        import platform
        import time
        
        # Informations du node local
        cpu_count = psutil.cpu_count(logical=True)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        boot_time = psutil.boot_time()
        uptime_seconds = time.time() - boot_time
        
        # Informations GPU
        gpu_info = {'name': 'No GPU', 'vram': 0, 'count': 0}
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                device_props = torch.cuda.get_device_properties(0)
                gpu_info = {
                    'name': device_name,
                    'vram': round(device_props.total_memory / (1024**2)),  # MB
                    'count': gpu_count
                }
        except Exception as e:
            logger.debug(f"Could not get GPU info: {e}")
        
        # Calculer uptime formaté
        uptime_hours = int(uptime_seconds / 3600)
        uptime_minutes = int((uptime_seconds % 3600) / 60)
        
        node_data = {
            'host': 'localhost',
            'name': f'VRAMancer-{platform.node()}',
            'status': 'active',
            'role': 'master',
            'os': f"{platform.system()} {platform.release()}",
            'cpu': cpu_count,
            'memory': round(memory.total / (1024**3), 1),  # GB
            'vram': gpu_info['vram'],
            'gpu_name': gpu_info['name'],
            'gpu_count': gpu_info['count'],
            'backend': 'VRAMancer Production',
            'ip': '127.0.0.1',
            'port': API_PORT,
            'uptime': f"{uptime_hours}h {uptime_minutes}m",
            'uptime_seconds': int(uptime_seconds),
            'load': round(cpu_percent, 1),
            'memory_percent': round(memory.percent, 1),
            'info': f"CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}% | Active"
        }
        
        return jsonify({
            'nodes': [node_data],
            'count': 1,
            'timestamp': time.time()
        })
        
    except Exception as e:
        logger.error(f"Nodes endpoint error: {e}", exc_info=True)
        return jsonify({
            'nodes': [{
                'host': 'localhost',
                'name': 'Error Node',
                'status': 'error',
                'error': str(e)
            }],
            'count': 1
        }), 500


# ============================================================================
# Main
# ============================================================================

def main():
    """Point d'entrée principal"""
    logger.info("=" * 60)
    logger.info("VRAMancer API - Production Mode")
    logger.info("=" * 60)
    logger.info(f"Host: {API_HOST}")
    logger.info(f"Port: {API_PORT}")
    logger.info(f"Debug: {API_DEBUG}")
    logger.info(f"API Base: {os.environ.get('VRM_API_BASE')}")
    logger.info("=" * 60)
    logger.info("Endpoints disponibles:")
    logger.info("  • Health:  http://localhost:{}/health".format(API_PORT))
    logger.info("  • Ready:   http://localhost:{}/ready".format(API_PORT))
    logger.info("  • Status:  http://localhost:{}/api/status".format(API_PORT))
    logger.info("  • GPU:     http://localhost:{}/api/gpu".format(API_PORT))
    logger.info("  • System:  http://localhost:{}/api/system".format(API_PORT))
    logger.info("  • Nodes:   http://localhost:{}/api/nodes".format(API_PORT))
    logger.info("=" * 60)
    
    try:
        # Ne JAMAIS utiliser debug=True en production
        app.run(
            host=API_HOST,
            port=API_PORT,
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.info("Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.critical(f"Erreur fatale: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
