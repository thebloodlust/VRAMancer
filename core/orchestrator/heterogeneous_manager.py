"""
Gestionnaire de nœuds hétérogènes VRAMancer.
Support CUDA, ROCm, Apple Silicon MPS, CPU, IoT/Edge.
"""
from __future__ import annotations
import platform
import psutil
import socket
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from core.logger import LoggerAdapter
from core.utils import enumerate_devices, detect_backend

@dataclass
class NodeCapabilities:
    """Capacités détaillées d'un nœud hétérogène."""
    hostname: str
    architecture: str  # x86_64, arm64, aarch64
    platform: str      # Linux, Darwin, Windows
    cpu_model: str
    cpu_cores: int
    ram_gb: float
    
    # GPU/Accélérateurs
    gpus: List[Dict[str, Any]]
    primary_backend: str  # cuda, rocm, mps, cpu
    
    # Réseau
    network_interfaces: List[str]
    bandwidth_mbps: Optional[float] = None
    latency_ms: Optional[float] = None
    
    # Edge/IoT markers
    is_edge: bool = False
    power_profile: str = "standard"  # standard, low_power, high_perf
    
    # Performance estimée
    compute_score: float = 1.0
    memory_score: float = 1.0
    network_score: float = 1.0

class HeterogeneousManager:
    """Orchestrateur pour clusters multi-architecture."""
    
    def __init__(self):
        self.log = LoggerAdapter("hetero")
        self.nodes: Dict[str, NodeCapabilities] = {}
        self.load_balancing_weights: Dict[str, float] = {}
        
    def detect_local_capabilities(self) -> NodeCapabilities:
        """Détecte les capacités du nœud local."""
        hostname = socket.gethostname()
        
        # Architecture et plateforme
        arch = platform.machine()
        plat = platform.system()
        cpu_model = platform.processor() or "Unknown"
        cpu_cores = psutil.cpu_count(logical=False) or 1
        
        # RAM
        ram_bytes = psutil.virtual_memory().total
        ram_gb = ram_bytes / (1024**3)
        
        # GPU/Accélérateurs
        gpus = self._detect_gpus()
        backend = detect_backend()
        
        # Réseau
        network_ifs = self._get_network_interfaces()
        
        # Edge detection
        is_edge = self._is_edge_device(arch, cpu_model, ram_gb)
        power_profile = self._estimate_power_profile(arch, cpu_cores, ram_gb)
        
        # Scores de performance
        compute_score = self._estimate_compute_score(backend, gpus, cpu_cores)
        memory_score = self._estimate_memory_score(ram_gb, gpus)
        network_score = 1.0  # À mesurer dynamiquement
        
        caps = NodeCapabilities(
            hostname=hostname,
            architecture=arch,
            platform=plat,
            cpu_model=cpu_model,
            cpu_cores=cpu_cores,
            ram_gb=ram_gb,
            gpus=gpus,
            primary_backend=backend,
            network_interfaces=network_ifs,
            is_edge=is_edge,
            power_profile=power_profile,
            compute_score=compute_score,
            memory_score=memory_score,
            network_score=network_score
        )
        
        self.log.info(f"Détection local: {hostname} | {arch} | {backend} | {len(gpus)} GPU(s) | {ram_gb:.1f}GB RAM")
        return caps
    
    def _detect_gpus(self) -> List[Dict[str, Any]]:
        """Détecte tous les GPUs/accélérateurs disponibles."""
        devices = enumerate_devices()
        gpus = []
        
        for device in devices:
            if device['backend'] in ['cuda', 'rocm', 'mps']:
                gpu_info = {
                    'id': device['id'],
                    'backend': device['backend'],
                    'name': device['name'],
                    'memory_mb': device.get('total_memory', 0) // (1024**2) if device.get('total_memory') else 0,
                    'architecture': self._get_gpu_architecture(device['name'])
                }
                gpus.append(gpu_info)
        
        return gpus
    
    def _get_gpu_architecture(self, name: str) -> str:
        """Détermine l'architecture GPU à partir du nom."""
        name_lower = name.lower()
        
        # NVIDIA
        if 'rtx 40' in name_lower or 'ada' in name_lower:
            return 'Ada Lovelace'
        elif 'rtx 30' in name_lower or 'ampere' in name_lower:
            return 'Ampere'
        elif 'rtx 20' in name_lower or 'turing' in name_lower:
            return 'Turing'
        elif 'gtx' in name_lower or 'pascal' in name_lower:
            return 'Pascal'
        
        # AMD  
        elif 'mi' in name_lower and ('50' in name_lower or '60' in name_lower):
            return 'CDNA 1/2'
        elif 'rx 7' in name_lower or 'rdna3' in name_lower:
            return 'RDNA 3'
        elif 'rx 6' in name_lower or 'rdna2' in name_lower:
            return 'RDNA 2'
        
        # Apple
        elif 'apple' in name_lower or 'mps' in name_lower:
            return 'Apple Silicon'
            
        return 'Unknown'
    
    def _get_network_interfaces(self) -> List[str]:
        """Liste les interfaces réseau disponibles."""
        interfaces = []
        try:
            import netifaces
            for iface in netifaces.interfaces():
                if iface != 'lo' and not iface.startswith('docker'):
                    interfaces.append(iface)
        except ImportError:
            # Fallback sans netifaces
            interfaces = ['eth0', 'wlan0']  # Estimation
        
        return interfaces
    
    def _is_edge_device(self, arch: str, cpu_model: str, ram_gb: float) -> bool:
        """Détermine si c'est un device edge/IoT."""
        # ARM généralement edge
        if arch.lower() in ['arm64', 'aarch64', 'armv7l']:
            return True
        
        # RAM faible = edge probable
        if ram_gb < 8:
            return True
            
        # Patterns edge connus
        edge_patterns = ['raspberry', 'jetson', 'coral', 'atom']
        cpu_lower = cpu_model.lower()
        return any(pattern in cpu_lower for pattern in edge_patterns)
    
    def _estimate_power_profile(self, arch: str, cores: int, ram_gb: float) -> str:
        """Estime le profil de consommation."""
        if self._is_edge_device(arch, "", ram_gb):
            return "low_power"
        elif cores >= 16 and ram_gb >= 64:
            return "high_perf"
        else:
            return "standard"
    
    def _estimate_compute_score(self, backend: str, gpus: List[Dict], cpu_cores: int) -> float:
        """Score de capacité de calcul (0.1 à 10.0)."""
        base_score = cpu_cores / 8.0  # 8 cores = score 1.0
        
        # Bonus GPU
        for gpu in gpus:
            if backend == 'cuda':
                # RTX 4090 = +8, RTX 3090 = +6, etc.
                if 'rtx 4090' in gpu['name'].lower():
                    base_score += 8.0
                elif 'rtx 4080' in gpu['name'].lower():
                    base_score += 6.0
                elif 'rtx 3090' in gpu['name'].lower():
                    base_score += 6.0
                elif 'rtx 3080' in gpu['name'].lower():
                    base_score += 4.0
                elif '4060 ti' in gpu['name'].lower():
                    base_score += 3.0
                else:
                    base_score += 2.0
            elif backend == 'rocm':
                # AMD MI series
                if 'mi50' in gpu['name'].lower():
                    base_score += 5.0
                elif 'mi60' in gpu['name'].lower():
                    base_score += 7.0
                else:
                    base_score += 2.0
            elif backend == 'mps':
                # Apple Silicon
                if 'm4' in gpu['name'].lower():
                    base_score += 4.0
                elif 'm3' in gpu['name'].lower():
                    base_score += 3.0
                elif 'm1' in gpu['name'].lower():
                    base_score += 2.0
        
        return min(base_score, 10.0)
    
    def _estimate_memory_score(self, ram_gb: float, gpus: List[Dict]) -> float:
        """Score de capacité mémoire (0.1 à 10.0)."""
        base_score = ram_gb / 32.0  # 32GB = score 1.0
        
        # Bonus VRAM
        total_vram_gb = sum(gpu.get('memory_mb', 0) for gpu in gpus) / 1024
        base_score += total_vram_gb / 16.0  # 16GB VRAM = +1.0
        
        return min(base_score, 10.0)
    
    def register_node(self, node_caps: NodeCapabilities):
        """Enregistre un nœud dans le cluster."""
        self.nodes[node_caps.hostname] = node_caps
        self.log.info(f"Nœud enregistré: {node_caps.hostname} | Score: {node_caps.compute_score:.1f}")
    
    def get_optimal_placement(self, task_requirements: Dict[str, Any]) -> List[str]:
        """Détermine le placement optimal pour une tâche."""
        required_memory = task_requirements.get('memory_mb', 1024)
        required_compute = task_requirements.get('compute_score', 1.0)
        preferred_backend = task_requirements.get('backend', 'any')
        
        candidates = []
        
        for hostname, caps in self.nodes.items():
            # Filtrage backend
            if preferred_backend != 'any' and caps.primary_backend != preferred_backend:
                continue
            
            # Vérification mémoire
            total_memory_mb = caps.ram_gb * 1024 + sum(gpu.get('memory_mb', 0) for gpu in caps.gpus)
            if total_memory_mb < required_memory:
                continue
            
            # Score composite
            score = (caps.compute_score * 0.5 + 
                    caps.memory_score * 0.3 + 
                    caps.network_score * 0.2)
            
            candidates.append((hostname, score))
        
        # Tri par score décroissant
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [hostname for hostname, _ in candidates]
    
    def balance_load(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """Répartit les tâches sur les nœuds selon leurs capacités."""
        allocation = {hostname: [] for hostname in self.nodes.keys()}
        
        for task in tasks:
            candidates = self.get_optimal_placement(task)
            if candidates:
                # Attribution au nœud le moins chargé parmi les candidats
                best_node = min(candidates[:3], key=lambda h: len(allocation[h]))
                allocation[best_node].append(task)
                self.log.debug(f"Tâche {task.get('id', 'unknown')} -> {best_node}")
        
        return allocation
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Résumé du cluster hétérogène."""
        if not self.nodes:
            return {"nodes": 0, "total_gpus": 0, "total_ram_gb": 0}
        
        total_gpus = sum(len(caps.gpus) for caps in self.nodes.values())
        total_ram = sum(caps.ram_gb for caps in self.nodes.values())
        
        backend_counts = {}
        arch_counts = {}
        
        for caps in self.nodes.values():
            backend_counts[caps.primary_backend] = backend_counts.get(caps.primary_backend, 0) + 1
            arch_counts[caps.architecture] = arch_counts.get(caps.architecture, 0) + 1
        
        return {
            "nodes": len(self.nodes),
            "total_gpus": total_gpus,
            "total_ram_gb": total_ram,
            "backends": backend_counts,
            "architectures": arch_counts,
            "edge_nodes": sum(1 for caps in self.nodes.values() if caps.is_edge)
        }

# Singleton global
_heterogeneous_manager = HeterogeneousManager()

def get_heterogeneous_manager() -> HeterogeneousManager:
    """Retourne l'instance globale du gestionnaire hétérogène."""
    return _heterogeneous_manager

__all__ = ["HeterogeneousManager", "NodeCapabilities", "get_heterogeneous_manager"]