import time
class MemoryBenchmarker:
    """
    Mesure la latence et la bande passante de chaque niveau mémoire (DRAM, NVMe, Réseau).
    """
    def __init__(self, nvme_dir, remote_nodes=None):
        self.nvme_dir = nvme_dir
        self.remote_nodes = remote_nodes or []
        self.results = {}

    def bench_dram(self, block):
        start = time.perf_counter()
        _ = block.cpu()
        end = time.perf_counter()
        return end - start

    def bench_nvme(self, block):
        import uuid
        path = os.path.join(self.nvme_dir, f"bench_{uuid.uuid4()}.pt")
        start = time.perf_counter()
        save_block_to_disk(block.cpu(), path)
        _ = load_block_from_disk(path)
        end = time.perf_counter()
        os.remove(path)
        return end - start

    def bench_network(self, block):
        # Simulation : envoyer/recevoir un bloc sur le premier nœud
        if not self.remote_nodes:
            return float('inf')
        node = self.remote_nodes[0]
        start = time.perf_counter()
        try:
            send_block(block, target_device=node)
            # TODO : recevoir le bloc (simulateur)
        except Exception:
            return float('inf')
        end = time.perf_counter()
        return end - start

    def bench_all(self, block):
        self.results = {
            'dram': self.bench_dram(block),
            'nvme': self.bench_nvme(block),
            'network': self.bench_network(block),
        }
        return self.results

    def get_fastest_level(self):
        if not self.results:
            return 'dram'
        return min(self.results, key=self.results.get)
import logging
from core.memory_balancer import MemoryBalancer
from core.stream_manager import StreamManager
from core.monitor import GPUMonitor
from core.metrics import ORCH_PLACEMENTS, ORCH_MIGRATIONS, ORCH_REBALANCE, ORCH_HIERARCHY_MOVE


from core.storage_manager import load_block_from_disk, save_block_to_disk
from core.network.vramancer_link import send_block, start_client
import os

class BlockOrchestrator:
    """
    Orchestrateur intelligent pour la gestion automatique des blocs sur multi-GPU/accélérateurs.
    Hiérarchie mémoire : VRAM principale, VRAM secondaires (cache), DRAM (RAM), NVMe (L4), Réseau (L5).
    Optimise la répartition selon l’utilisation et la vitesse mémoire.
    """
    def __init__(self, scheduler, logger=None, cache_size=2, nvme_dir="/tmp/vramancer_blocks", remote_nodes=None):
        self.logger = logger or logging.getLogger("BlockOrchestrator")
        self.monitor = GPUMonitor()
        self.balancer = MemoryBalancer(scheduler, self.logger, cache_size=cache_size)
        self.stream_manager = StreamManager(scheduler, self.logger)
        self.scheduler = scheduler
        self.dram_cache = {}  # {block_id: block}  # L4
        self.nvme_dir = nvme_dir
        os.makedirs(self.nvme_dir, exist_ok=True)
        self.remote_nodes = remote_nodes or []  # Liste d'adresses de nœuds distants (L6)
        self.remote_connected = False
        self.benchmarker = MemoryBenchmarker(self.nvme_dir, self.remote_nodes)

    def place_block(self, block, preferred_gpus=None):
        """
        Place un bloc sur le GPU optimal selon la charge et le cache.
        """
        usage = self.monitor.status()
        candidates = preferred_gpus or list(usage.keys())
        best_gpu = min(candidates, key=lambda g: float(usage[str(g)].replace('% VRAM','').replace('%','')))
        res = self.balancer.allocate_block(block, int(best_gpu))
        try:
            ORCH_PLACEMENTS.labels("vram").inc()
        except Exception:
            pass
        return res

    def release_block(self, block, gpu_id):
        self.balancer.release_block(block, gpu_id)

    def migrate_block(self, block, src_gpu, dst_gpu):
        res = self.balancer.migrate_block(block, src_gpu, dst_gpu)
        try:
            ORCH_MIGRATIONS.inc()
        except Exception:
            pass
        return res

    def rebalance(self, blocks, threshold=90.0, dram_limit=10):
        """
        Rééquilibre automatiquement tous les blocs selon la charge live.
        Blocs les moins utilisés migrés vers DRAM, NVMe, ou réseau si besoin.
        """
        usage = {int(k): float(v.replace('% VRAM','').replace('%','')) for k,v in self.monitor.status().items()}
        self.balancer.balance(blocks, usage, threshold=threshold)
        try:
            ORCH_REBALANCE.inc()
        except Exception:
            pass
        # Si VRAM saturée, migrer les blocs les moins utilisés vers le niveau le plus rapide disponible
        if any(u > threshold for u in usage.values()):
            sorted_blocks = sorted(blocks, key=lambda b: getattr(b, 'last_access', 0))
            for block in sorted_blocks:
                bench = self.benchmarker.bench_all(block)
                fastest = self.benchmarker.get_fastest_level()
                if fastest == 'dram' and len(self.dram_cache) < dram_limit:
                    self.logger.info(f"[Orchestrator] Bloc {block.id} migré en DRAM (L4)")
                    self.dram_cache[block.id] = block.cpu()
                    try: ORCH_HIERARCHY_MOVE.labels("dram").inc()
                    except Exception: pass
                elif fastest == 'nvme':
                    path = os.path.join(self.nvme_dir, f"{block.id}.pt")
                    save_block_to_disk(block.cpu(), path)
                    self.logger.info(f"[Orchestrator] Bloc {block.id} migré sur NVMe (L5) : {path}")
                    try: ORCH_HIERARCHY_MOVE.labels("nvme").inc()
                    except Exception: pass
                elif fastest == 'network':
                    for node in self.remote_nodes:
                        try:
                            send_block(block, target_device=node)
                            self.logger.info(f"[Orchestrator] Bloc {block.id} migré sur le réseau (L6) : {node}")
                            try: ORCH_HIERARCHY_MOVE.labels("network").inc()
                            except Exception: pass
                            break
                        except Exception as e:
                            self.logger.warning(f"[Orchestrator] Erreur migration réseau {node} : {e}")

    def fetch_block(self, block_id):
        """
        Récupère un bloc depuis la hiérarchie (DRAM, NVMe, ou réseau).
        """
        if block_id in self.dram_cache:
            return self.dram_cache[block_id]
        path = os.path.join(self.nvme_dir, f"{block_id}.pt")
        if os.path.exists(path):
            return load_block_from_disk(path)
        # L6 : récupération réseau custom (SFP+ sans TCP/IP)
        for node in self.remote_nodes:
            try:
                if not self.remote_connected:
                    start_client(node)
                    self.remote_connected = True
                # Ici, on suppose que le protocole custom permet de demander un bloc par son id
                # send_block peut être adapté pour demander/réceptionner un bloc
                self.logger.info(f"[Orchestrator] Demande bloc {block_id} à {node} (L6)")
                # TODO : implémenter la logique de requête/réception réelle selon le protocole custom
                # Ex : send_block({"request_block": block_id}, target_device=node)
                # Bloc fictif pour l'exemple :
                # received_block = receive_block(block_id, node)
                # return received_block
            except Exception as e:
                self.logger.warning(f"[Orchestrator] Erreur accès L6 {node} : {e}")
        self.logger.warning(f"[Orchestrator] Bloc {block_id} introuvable dans la hiérarchie (L1-L6)")
        return None

    def auto_manage(self, layers, lookahead=2, threshold=90.0, dram_limit=10):
        """
        Orchestration complète : préchargement, swap, rééquilibrage, libération, migration hiérarchique.
        """
        self.stream_manager.prefetch_layers(layers, lookahead=lookahead)
        loaded_blocks = [self.stream_manager.loaded_layers[name] for name in self.stream_manager.loaded_layers]
        self.rebalance(loaded_blocks, threshold=threshold, dram_limit=dram_limit)
        self.stream_manager.swap_if_needed()
        self.stream_manager.unload_unused(layers)

    def get_state(self):
        return {
            "memory": self.balancer.get_memory_state(),
            "cache": self.balancer.get_cache_state(),
            "dram": list(self.dram_cache.keys()),
            "nvme": os.listdir(self.nvme_dir),
            "loaded": list(self.stream_manager.loaded_layers.keys()),
            "remote_nodes": self.remote_nodes,
        }
