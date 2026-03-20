"""Lightweight Edge REST API for smartphones and IoT devices.

Provides a minimal HTTP API that edge devices can use to:
  1. Register as swarm workers (report capabilities)
  2. Pull small compute tasks (quantized tensor shards)
  3. Push results back
  4. Report battery/thermal status

This is designed for devices that can't maintain persistent WebSocket
connections (battery constraints, intermittent connectivity).

Usage:
    from core.network.edge_api import create_edge_app
    app = create_edge_app()
    app.run(host="0.0.0.0", port=5035)

Or integrated into the main production API via Blueprint.
"""

import os
import time
import uuid
import json
import logging
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

_edge_registry: Dict[str, Dict[str, Any]] = {}
_task_inbox: Dict[str, list] = {}  # device_id -> [tasks]
_results: Dict[str, bytes] = {}  # task_id -> result
_lock = threading.Lock()

# Max age for an edge device before it's considered offline
EDGE_MAX_AGE_S = int(os.environ.get("VRM_EDGE_MAX_AGE", "120"))


def register_edge_device(device_id: str, capabilities: dict) -> dict:
    """Register or update an edge device in the swarm."""
    with _lock:
        is_new = device_id not in _edge_registry
        _edge_registry[device_id] = {
            "capabilities": capabilities,
            "last_seen": time.time(),
            "status": "active",
            "tasks_completed": _edge_registry.get(device_id, {}).get("tasks_completed", 0),
        }
        if device_id not in _task_inbox:
            _task_inbox[device_id] = []

    action = "registered" if is_new else "updated"
    logger.info(
        f"[Edge] Device {device_id} {action}: "
        f"{capabilities.get('device_type', '?')}, "
        f"battery={capabilities.get('battery', '?')}%, "
        f"gpu={capabilities.get('gpu', 'none')}"
    )
    return {"status": "ok", "action": action, "device_id": device_id}


def get_active_edge_devices(max_age: float = None) -> Dict[str, dict]:
    """Return all active edge devices."""
    cutoff = time.time() - (max_age or EDGE_MAX_AGE_S)
    with _lock:
        return {
            did: info
            for did, info in _edge_registry.items()
            if info["last_seen"] > cutoff
        }


def submit_edge_task(device_id: str, task_id: str, payload: bytes, metadata: dict = None):
    """Submit a compute task for an edge device to pick up."""
    with _lock:
        if device_id not in _task_inbox:
            _task_inbox[device_id] = []
        _task_inbox[device_id].append({
            "task_id": task_id,
            "payload": payload,
            "metadata": metadata or {},
            "created_at": time.time(),
        })


def pull_edge_task(device_id: str) -> Optional[dict]:
    """Pull the next pending task for this device (FIFO)."""
    with _lock:
        tasks = _task_inbox.get(device_id, [])
        if not tasks:
            return None
        task = tasks.pop(0)
        # Update last_seen
        if device_id in _edge_registry:
            _edge_registry[device_id]["last_seen"] = time.time()
        return task


def submit_edge_result(device_id: str, task_id: str, result_data: bytes):
    """Submit the result of a completed task."""
    with _lock:
        _results[task_id] = result_data
        if device_id in _edge_registry:
            _edge_registry[device_id]["tasks_completed"] = (
                _edge_registry[device_id].get("tasks_completed", 0) + 1
            )
            _edge_registry[device_id]["last_seen"] = time.time()


def create_edge_app():
    """Create a Flask Blueprint for the edge REST API."""
    try:
        from flask import Blueprint, request, jsonify
    except ImportError:
        logger.warning("[Edge API] Flask not available, edge API disabled")
        return None

    bp = Blueprint("edge_api", __name__, url_prefix="/api/edge")

    @bp.route("/register", methods=["POST"])
    def api_register():
        data = request.get_json(silent=True) or {}
        device_id = data.get("device_id") or uuid.uuid4().hex[:12]
        caps = data.get("capabilities", {})
        # Ensure required fields
        caps.setdefault("device_type", "unknown")
        caps.setdefault("battery", 100)
        caps.setdefault("gpu", "none")
        caps.setdefault("compute_tflops", 0)
        caps.setdefault("ram_mb", 0)
        result = register_edge_device(device_id, caps)
        return jsonify(result)

    @bp.route("/heartbeat", methods=["POST"])
    def api_heartbeat():
        data = request.get_json(silent=True) or {}
        device_id = data.get("device_id", "")
        with _lock:
            if device_id in _edge_registry:
                _edge_registry[device_id]["last_seen"] = time.time()
                # Update battery if provided
                if "battery" in data:
                    _edge_registry[device_id]["capabilities"]["battery"] = data["battery"]
                if "thermal" in data:
                    _edge_registry[device_id]["capabilities"]["thermal"] = data["thermal"]
                return jsonify({"status": "ok", "pending_tasks": len(_task_inbox.get(device_id, []))})
        return jsonify({"status": "unknown_device"}), 404

    @bp.route("/task/pull", methods=["POST"])
    def api_pull_task():
        data = request.get_json(silent=True) or {}
        device_id = data.get("device_id", "")
        task = pull_edge_task(device_id)
        if task is None:
            return jsonify({"status": "no_task"})
        # encode payload as base64 for JSON transport
        import base64
        return jsonify({
            "status": "ok",
            "task_id": task["task_id"],
            "payload": base64.b64encode(task["payload"]).decode(),
            "metadata": task["metadata"],
        })

    @bp.route("/task/result", methods=["POST"])
    def api_submit_result():
        data = request.get_json(silent=True) or {}
        device_id = data.get("device_id", "")
        task_id = data.get("task_id", "")
        import base64
        result_b64 = data.get("result", "")
        try:
            result_data = base64.b64decode(result_b64)
        except Exception:
            result_data = result_b64.encode() if isinstance(result_b64, str) else b""
        submit_edge_result(device_id, task_id, result_data)
        return jsonify({"status": "ok"})

    @bp.route("/devices", methods=["GET"])
    def api_list_devices():
        devices = get_active_edge_devices()
        summary = {}
        for did, info in devices.items():
            summary[did] = {
                "capabilities": info["capabilities"],
                "last_seen": info["last_seen"],
                "tasks_completed": info.get("tasks_completed", 0),
            }
        return jsonify({"devices": summary, "count": len(summary)})

    return bp
