#!/usr/bin/env python3
"""
API VRAMancer - Version ultra simple
"""
import os
import sys

# Définir les variables d'environnement
os.environ['VRM_API_BASE'] = 'http://localhost:5030'
os.environ['VRM_API_PORT'] = '5030'

print("🚀 Démarrage API VRAMancer simple...")
print(f"VRM_API_BASE = {os.environ['VRM_API_BASE']}")

try:
    from flask import Flask, jsonify
    print("✅ Flask disponible")
except ImportError:
    print("❌ Flask non disponible, installation...")
    os.system("python -m pip install flask")
    from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "ok", "service": "vramancer"})

@app.route('/api/status')
def status():
    return jsonify({
        "backend": "running", 
        "version": "1.0",
        "api_base": os.environ.get('VRM_API_BASE', 'http://localhost:5030')
    })

@app.route('/api/gpu/info')
def gpu_info():
    return jsonify({
        "gpus": [{"id": 0, "name": "Default GPU", "memory": "Detecting..."}],
        "message": "API fonctionnelle - GPU detection en cours"
    })

@app.route('/api/gpu')
def gpu():
    """Endpoint GPU pour dashboards"""
    try:
        import torch
        if torch.cuda.is_available():
            devices = []
            for i in range(torch.cuda.device_count()):
                device_name = torch.cuda.get_device_name(i)
                device_props = torch.cuda.get_device_properties(i)
                
                memory_allocated = torch.cuda.memory_allocated(i) if torch.cuda.is_initialized() else 0
                memory_reserved = torch.cuda.memory_reserved(i) if torch.cuda.is_initialized() else 0
                memory_total = device_props.total_memory
                
                devices.append({
                    "id": i,
                    "name": device_name,
                    "backend": "PyTorch CUDA",
                    "memory_used": memory_allocated,
                    "memory_total": memory_total,
                    "memory_reserved": memory_reserved,
                    "compute_capability": f"{device_props.major}.{device_props.minor}"
                })
            
            return jsonify({"devices": devices, "cuda_available": True})
        else:
            return jsonify({"devices": [], "cuda_available": False, "message": "CUDA non disponible"})
    except ImportError:
        return jsonify({"devices": [], "cuda_available": False, "message": "PyTorch non installé"})

@app.route('/api/system')
def system():
    """Endpoint System pour dashboards"""
    try:
        import psutil, platform
        
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return jsonify({
            "platform": platform.system(),
            "cpu_count": cpu_count,
            "cpu_percent": cpu_percent,
            "memory_gb": round(memory.total / (1024**3), 2),
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_percent": memory.percent,
            "backend": "VRAMancer Simple"
        })
    except ImportError:
        return jsonify({
            "platform": "Unknown",
            "cpu_count": 0,
            "cpu_percent": 0,
            "memory_gb": 0,
            "memory_used_gb": 0,
            "memory_percent": 0,
            "backend": "No psutil"
        })

@app.route('/api/nodes')
def nodes():
    """Endpoint Nodes pour dashboard web avancé"""
    try:
        import psutil, platform, time
        
        # Informations du node local
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        boot_time = psutil.boot_time()
        uptime = time.time() - boot_time
        
        # Informations GPU
        gpu_info = {"name": "No GPU", "vram": 0}
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                device_props = torch.cuda.get_device_properties(0)
                gpu_info = {
                    "name": device_name,
                    "vram": round(device_props.total_memory / (1024**2))  # MB
                }
        except:
            pass
            
        node_data = {
            "host": "localhost",
            "name": "VRAMancer Local Node",
            "status": "active",
            "os": f"{platform.system()} {platform.release()}",
            "cpu": cpu_count,
            "memory": round(memory.total / (1024**3), 1),  # GB
            "vram": gpu_info["vram"],
            "gpu_name": gpu_info["name"],
            "backend": "VRAMancer Simple",
            "ip": "127.0.0.1",
            "port": 5030,
            "uptime": f"{int(uptime/3600)}h {int((uptime%3600)/60)}m",
            "load": round(cpu_percent, 1),
            "info": f"CPU: {cpu_percent:.1f}% | RAM: {memory.percent:.1f}% | Actif"
        }
        
        return jsonify({"nodes": [node_data], "count": 1})
        
    except Exception as e:
        return jsonify({
            "nodes": [{
                "host": "localhost",
                "name": "Error Node",
                "status": "error",
                "info": f"Erreur: {str(e)}"
            }],
            "count": 1
        })

if __name__ == '__main__':
    print("✅ API démarrée sur http://localhost:5030")
    print("   • Health: http://localhost:5030/health")
    print("   • Status: http://localhost:5030/api/status") 
    print("   • GPU: http://localhost:5030/api/gpu/info")
    print("\n🎮 Vous pouvez maintenant lancer les interfaces!")
    
    try:
        app.run(host='0.0.0.0', port=5030, debug=False)
    except Exception as e:
        print(f"❌ Erreur: {e}")
        input("Appuyez sur Entrée...")