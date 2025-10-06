"""Planificateur de tâches opportuniste.

Objectif: Réutiliser automatiquement les ressources GPU / CPU inactives pour lancer des
travaux secondaires (pré-chargement de blocs, compression, export, warmup, etc.).

Approche:
 - Détection périodique des devices disponibles (via enumerate_devices())
 - File de tâches (priorité simple: HIGH > NORMAL > LOW)
 - Attribution: on réserve une ressource si elle n'a pas dépassé un seuil d'utilisation mémoire
   ou si c'est un CPU idle.
 - Exécution asynchrone via threads légers (compatibles I/O / jobs courts). Pour des jobs
   compute lourds, extension possible vers multiprocessing.

API:
    scheduler = OpportunisticScheduler()
    scheduler.submit(func, priority='NORMAL', tags=['warmup'])

Tâches: fonction sans argument ou fonction avec kwargs pré-liés (utiliser functools.partial).

Sécurité: Exécution best-effort; exceptions capturées et comptabilisées.
"""
from __future__ import annotations
import time, threading, queue, traceback
import os
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Dict, Any, Deque, Tuple
from collections import defaultdict, deque
import inspect
import uuid, psutil
try:
    import torch
except Exception:  # pragma: no cover
    torch = None
from core.utils import enumerate_devices
from core.metrics import (
    TASKS_SUBMITTED, TASKS_COMPLETED, TASKS_FAILED,
    TASKS_RUNNING, TASKS_PER_RESOURCE, TASK_DURATION
)

PRIORITY_ORDER = {"HIGH":0, "NORMAL":1, "LOW":2}

@dataclass(order=True)
class ScheduledTask:
    sort_index: int = field(init=False, repr=False)
    priority: str
    fn: Callable
    submitted_at: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    est_runtime_s: float = 0.0  # renseigné pour adaptation priorité
    cancelled: bool = False

    def __post_init__(self):
        self.sort_index = PRIORITY_ORDER.get(self.priority.upper(), 1)

class OpportunisticScheduler:
    def __init__(self, poll_interval: float = 0.1, max_threads: int = 8, gpu_mem_threshold: float = 0.85, adaptive_spill: bool = True,
                 adaptive_priority: bool = True, history_path: Optional[str] = None,
                 runtime_estimator: Optional[Callable[[ScheduledTask], float]] = None,
                 reservoir_size: int = 500):
        self.poll_interval = poll_interval
        self.gpu_mem_threshold = gpu_mem_threshold
        self.adaptive_spill = adaptive_spill
        self.adaptive_priority = adaptive_priority
        self.tasks: "queue.PriorityQueue[ScheduledTask]" = queue.PriorityQueue()
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self.max_threads = max_threads
        self.active = {}
        self.history: List[Dict[str, Any]] = []
        self.history_path = history_path
        self.runtime_estimator = runtime_estimator
        # Reservoirs pour percentiles (clé = (priority,status))
        self._dur_samples: Dict[Tuple[str,str], Deque[float]] = defaultdict(lambda: deque(maxlen=reservoir_size))
        # Cancellation events par tâche active
        self._cancel_events: Dict[str, threading.Event] = {}
        self._wake = threading.Event()
        self._dispatcher_thread = threading.Thread(target=self._loop, daemon=True)
        self._dispatcher_thread.start()
        # Persistence simple (recharge historique si fourni)
        if self.history_path and os.path.exists(self.history_path):
            try:
                import json
                with open(self.history_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        self.history.append(json.loads(line))
            except Exception:
                pass

    # --------------------------------------------------------- API publique
    def submit(self, fn: Callable, priority: str = "NORMAL", tags: Optional[List[str]] = None, est_runtime_s: float = 0.0):
        task = ScheduledTask(priority=priority.upper(), fn=fn, tags=tags or [], est_runtime_s=est_runtime_s)
        if self.runtime_estimator and est_runtime_s == 0.0:
            try:
                task.est_runtime_s = float(self.runtime_estimator(task))
            except Exception:
                pass
        self.tasks.put(task)
        TASKS_SUBMITTED.inc()
        # Réveiller boucle pour éviter d'attendre poll_interval complet
        self._wake.set()
        return task

    def submit_batch(self, tasks: List[Dict[str, Any]]):
        ids = []
        for t in tasks:
            fn = t["fn"]
            pr = t.get("priority", "NORMAL")
            tags = t.get("tags", [])
            est = t.get("est_runtime_s", 0.0)
            ids.append(self.submit(fn, pr, tags, est).id)
        return ids

    def cancel(self, task_id: str) -> bool:
        # Marquage lazy: on reconstruit une nouvelle queue
        newq: "queue.PriorityQueue[ScheduledTask]" = queue.PriorityQueue()
        found = False
        while not self.tasks.empty():
            t: ScheduledTask = self.tasks.get()
            if t.id == task_id:
                t.cancelled = True
                found = True
            else:
                newq.put(t)
        self.tasks = newq
        # Cooperative: si déjà en cours, marquer flag sur objet actif
        for rid, atask in list(self.active.items()):
            if atask.id == task_id:
                atask.cancelled = True
                ev = self._cancel_events.get(task_id)
                if ev:
                    ev.set()
        return found

    def stop(self):
        self._stop.set()

    # --------------------------------------------------------- Boucle interne
    def _loop(self):
        while not self._stop.is_set():
            self._spawn_if_possible()
            # Attendre réveil ou timeout poll
            self._wake.wait(self.poll_interval)
            self._wake.clear()

    def _spawn_if_possible(self):
        # Nettoyage threads morts
        self._threads = [t for t in self._threads if t.is_alive()]
        TASKS_RUNNING.set(len(self._threads))
        # Limite globale
        if len(self._threads) >= self.max_threads:
            return
        # Pas de tâche prête
        if self.tasks.empty():
            return
        # Détecter ressources (simplifié: on ne vérifie pas usage mémoire runtime ici)
        devices = enumerate_devices()
        # Naïf: on prend la première tâche + la première ressource libre (ou CPU si rien)
        task: ScheduledTask = self.tasks.get()
        if task.cancelled:
            return
        # Admission control: VRAM/CPU
        if not self._admit():
            # Requeue plus tard
            self.tasks.put(task)
            return
        if self.adaptive_priority and task.est_runtime_s > 5.0 and task.priority == 'NORMAL':
            # Long running → basse priorité (réinjection)
            task.priority = 'LOW'
            task.sort_index = PRIORITY_ORDER['LOW']
            self.tasks.put(task)
            return
        resource_id = 'cpu:0'
        if self.adaptive_spill:
            # Priorité CUDA > ROCm > MPS > CPU
            order = []
            for d in devices:
                order.append((d['backend'], d['id']))
            pref = ['cuda', 'rocm', 'mps', 'cpu']
            for backend in pref:
                for b, rid in order:
                    if b == backend and rid not in self.active:
                        resource_id = rid
                        break
                if resource_id != 'cpu:0':  # trouvé mieux
                    break
        else:
            for d in devices:
                rid = d['id']
                if rid not in self.active:  # ressource libre
                    resource_id = rid
                    break
        self._launch(task, resource_id)

    def _launch(self, task: ScheduledTask, resource_id: str):
        def runner():
            self.active[resource_id] = task
            TASKS_PER_RESOURCE.labels(resource_id).inc()
            start = time.time()
            cancel_event = threading.Event()
            self._cancel_events[task.id] = cancel_event
            try:
                # Cooperative cancellation check wrappers
                if task.cancelled:
                    self._record_history(task, resource_id, 'cancelled')
                else:
                    # Injection contexte si fonction accepte arg
                    ctx = TaskContext(task_id=task.id, cancel_event=cancel_event)
                    try:
                        if len(inspect.signature(task.fn).parameters) >= 1:
                            task.fn(ctx)
                        else:
                            task.fn()
                    except CancelledError:
                        task.cancelled = True
                    except Exception:
                        raise
                    if task.cancelled:
                        self._record_history(task, resource_id, 'cancelled')
                        TASK_DURATION.labels(task.priority, 'cancelled').observe(time.time()-start)
                    else:
                        TASKS_COMPLETED.inc()
                        self._record_history(task, resource_id, 'completed')
                        TASK_DURATION.labels(task.priority, 'completed').observe(time.time()-start)
            except Exception:
                TASKS_FAILED.inc()
                traceback.print_exc()
                self._record_history(task, resource_id, 'failed')
                TASK_DURATION.labels(task.priority, 'failed').observe(time.time()-start)
            finally:
                TASKS_PER_RESOURCE.labels(resource_id).dec()
                self.active.pop(resource_id, None)
                self._cancel_events.pop(task.id, None)
        t = threading.Thread(target=runner, daemon=True)
        self._threads.append(t)
        t.start()

    # --------------------------------------------------------- Admission & History
    def _admit(self) -> bool:
        # CPU load: si > 95% on retarde
        try:
            if psutil.cpu_percent(interval=0.0) > 95:
                return False
        except Exception:
            pass
        # VRAM (si torch + cuda): si mem_alloc / mem_total > seuil => refuse
        if torch and torch.cuda.is_available():
            try:
                total = torch.cuda.get_device_properties(0).total_memory
                used = torch.cuda.memory_allocated(0)
                if used / total > self.gpu_mem_threshold:
                    return False
            except Exception:
                pass
        return True

    def _record_history(self, task: ScheduledTask, resource: str, status: str):
        entry = {
            'id': task.id,
            'priority': task.priority,
            'tags': task.tags,
            'resource': resource,
            'status': status,
            'submitted_at': task.submitted_at,
            'completed_at': time.time(),
        }
        self.history.append(entry)
        dur = entry['completed_at'] - task.submitted_at
        self._dur_samples[(task.priority, status)].append(dur)
        if self.history_path:
            try:
                import json
                with open(self.history_path, 'a') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

    # --------------------------------------------------------- Metrics export
    def compute_percentiles(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for (prio, status), samples in self._dur_samples.items():
            if not samples:
                continue
            arr = sorted(samples)
            def pct(p: float):
                if not arr:
                    return 0.0
                k = min(len(arr)-1, int(round(p/100.0*(len(arr)-1))))
                return arr[k]
            key = f"{prio.lower()}_{status}"
            out[key] = {
                'count': len(arr),
                'p50': pct(50),
                'p95': pct(95),
                'p99': pct(99),
                'min': arr[0],
                'max': arr[-1],
                'mean': sum(arr)/len(arr)
            }
        return out

class CancelledError(RuntimeError):
    pass

class TaskContext:
    def __init__(self, task_id: str, cancel_event: threading.Event):
        self.task_id = task_id
        self.cancel_event = cancel_event
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()
    def check(self):
        if self.cancelled():
            raise CancelledError("Task cancelled")
    def sleep(self, seconds: float, step: float = 0.1):
        end = time.time() + seconds
        while time.time() < end:
            if self.cancelled():
                raise CancelledError
            time.sleep(min(step, end - time.time()))

# Singleton léger optionnel
_global_scheduler: OpportunisticScheduler | None = None

def get_global_scheduler() -> OpportunisticScheduler:
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = OpportunisticScheduler()
    return _global_scheduler

__all__ = ["OpportunisticScheduler", "get_global_scheduler"]
