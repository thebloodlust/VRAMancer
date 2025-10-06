from __future__ import annotations
import os, time, logging
from core.memory_balancer import MemoryBalancer
from core.stream_manager import StreamManager
from core.monitor import GPUMonitor
from core.metrics import ORCH_PLACEMENTS, ORCH_MIGRATIONS, ORCH_REBALANCE, ORCH_HIERARCHY_MOVE
from core.storage_manager import load_block_from_disk, save_block_to_disk
from core.network.vramancer_link import send_block, start_client

class MemoryBenchmarker:
    def __init__(self, nvme_dir, remote_nodes=None):
        self.nvme_dir = nvme_dir
        self.remote_nodes = remote_nodes or []
        self.results = {}
    def bench_dram(self, block):
        start = time.perf_counter(); _ = block.cpu(); return time.perf_counter()-start
    def bench_nvme(self, block):
        import uuid
        path = os.path.join(self.nvme_dir, f"bench_{uuid.uuid4()}.pt")
        start = time.perf_counter(); save_block_to_disk(block.cpu(), path); _ = load_block_from_disk(path); os.remove(path); return time.perf_counter()-start
    def bench_network(self, block):
        if not self.remote_nodes: return float('inf')
        node = self.remote_nodes[0]
        start = time.perf_counter()
        try: send_block(block, target_device=node)
        except Exception: return float('inf')
        return time.perf_counter()-start
    def bench_all(self, block):
        self.results = { 'dram': self.bench_dram(block), 'nvme': self.bench_nvme(block), 'network': self.bench_network(block) }
        return self.results
    def get_fastest_level(self):
        if not self.results: return 'dram'
        return min(self.results, key=self.results.get)

class BlockOrchestrator:
    def __init__(self, scheduler, logger=None, cache_size=2, nvme_dir="/tmp/vramancer_blocks", remote_nodes=None):
        self.logger = logger or logging.getLogger("BlockOrchestrator")
        self.monitor = GPUMonitor()
        self.balancer = MemoryBalancer(scheduler, self.logger, cache_size=cache_size)
        self.stream_manager = StreamManager(scheduler, self.logger)
        self.scheduler = scheduler
        self.dram_cache = {}
        self.nvme_dir = nvme_dir; os.makedirs(self.nvme_dir, exist_ok=True)
        self.remote_nodes = remote_nodes or []
        self.remote_connected = False
        self.benchmarker = MemoryBenchmarker(self.nvme_dir, self.remote_nodes)
    def place_block(self, block, preferred_gpus=None):
        usage = self.monitor.status(); candidates = preferred_gpus or list(usage.keys())
        best_gpu = min(candidates, key=lambda g: float(usage[str(g)].replace('% VRAM','').replace('%','')))
        res = self.balancer.allocate_block(block, int(best_gpu))
        try: ORCH_PLACEMENTS.labels("vram").inc()
        except Exception: pass
        return res
    def release_block(self, block, gpu_id):
        self.balancer.release_block(block, gpu_id)
    def migrate_block(self, block, src_gpu, dst_gpu):
        res = self.balancer.migrate_block(block, src_gpu, dst_gpu)
        try: ORCH_MIGRATIONS.inc()
        except Exception: pass
        return res
    def rebalance(self, blocks, threshold=90.0, dram_limit=10):
        usage = {int(k): float(v.replace('% VRAM','').replace('%','')) for k,v in self.monitor.status().items()}
        self.balancer.balance(blocks, usage, threshold=threshold)
        try: ORCH_REBALANCE.inc()
        except Exception: pass
        if any(u > threshold for u in usage.values()):
            sorted_blocks = sorted(blocks, key=lambda b: getattr(b, 'last_access', 0))
            for block in sorted_blocks:
                self.benchmarker.bench_all(block)
                fastest = self.benchmarker.get_fastest_level()
                if fastest == 'dram' and len(self.dram_cache) < dram_limit:
                    self.logger.info(f"[Orchestrator] Bloc {block.id} -> DRAM")
                    self.dram_cache[block.id] = block.cpu();
                    try: ORCH_HIERARCHY_MOVE.labels("dram").inc()
                    except Exception: pass
                elif fastest == 'nvme':
                    path = os.path.join(self.nvme_dir, f"{block.id}.pt")
                    save_block_to_disk(block.cpu(), path)
                    self.logger.info(f"[Orchestrator] Bloc {block.id} -> NVMe {path}")
                    try: ORCH_HIERARCHY_MOVE.labels("nvme").inc()
                    except Exception: pass
                elif fastest == 'network':
                    for node in self.remote_nodes:
                        try:
                            send_block(block, target_device=node)
                            self.logger.info(f"[Orchestrator] Bloc {block.id} -> réseau {node}")
                            try: ORCH_HIERARCHY_MOVE.labels("network").inc()
                            except Exception: pass
                            break
                        except Exception as e:
                            self.logger.warning(f"Erreur migration réseau {node}: {e}")
    def fetch_block(self, block_id):
        if block_id in self.dram_cache: return self.dram_cache[block_id]
        path = os.path.join(self.nvme_dir, f"{block_id}.pt")
        if os.path.exists(path): return load_block_from_disk(path)
        for node in self.remote_nodes:
            try:
                if not self.remote_connected:
                    start_client(node); self.remote_connected = True
                self.logger.info(f"Demande bloc {block_id} à {node}")
            except Exception as e:
                self.logger.warning(f"Accès L6 {node} : {e}")
        return None
    def auto_manage(self, layers, lookahead=2, threshold=90.0, dram_limit=10):
        self.stream_manager.prefetch_layers(layers, lookahead=lookahead)
        loaded = [self.stream_manager.loaded_layers[n] for n in self.stream_manager.loaded_layers]
        self.rebalance(loaded, threshold=threshold, dram_limit=dram_limit)
        self.stream_manager.swap_if_needed(); self.stream_manager.unload_unused(layers)
    def get_state(self):
        return {
            'memory': self.balancer.get_memory_state(),
            'cache': self.balancer.get_cache_state(),
            'dram': list(self.dram_cache.keys()),
            'nvme': os.listdir(self.nvme_dir),
            'loaded': list(self.stream_manager.loaded_layers.keys()),
            'remote_nodes': self.remote_nodes,
        }

__all__ = ["BlockOrchestrator"]