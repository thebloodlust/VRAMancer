"""Verify that importing core modules does not start threads, open sockets,
or trigger network broadcasts. This protects against silent regressions where
a try/except at module top-level accidentally calls a constructor with side effects.
"""
import os
import sys
import threading
import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("VRM_CLUSTER_AUTO_DISCOVER", raising=False)
    monkeypatch.delenv("VRM_FEATURE_AITP", raising=False)
    monkeypatch.setenv("VRM_MINIMAL_TEST", "1")
    monkeypatch.setenv("VRM_DISABLE_RATE_LIMIT", "1")
    yield


def _count_threads_named(prefix: str) -> int:
    return sum(1 for t in threading.enumerate() if t.name.startswith(prefix))


def test_registry_import_starts_no_discovery():
    for m in list(sys.modules):
        if m.startswith("core.api.registry") or m.startswith("experimental.cluster_discovery"):
            sys.modules.pop(m, None)
    from core.api.registry import PipelineRegistry
    r = PipelineRegistry()
    assert r.discovery is None
    assert _count_threads_named("ClusterDiscovery") == 0
    assert _count_threads_named("Heartbeat") == 0


def test_production_api_create_app_no_vtp():
    for m in list(sys.modules):
        if m.startswith("core.cross_node") or m.startswith("core.network.llm_transport"):
            sys.modules.pop(m, None)
    from core.production_api import create_app
    app = create_app()
    assert app is not None
    assert _count_threads_named("VTP") == 0
    assert _count_threads_named("vtp") == 0


def test_inference_pipeline_import_does_not_load_gpu_threads():
    for m in list(sys.modules):
        if m == "core.inference_pipeline":
            sys.modules.pop(m, None)
    from core.inference_pipeline import InferencePipeline  # noqa: F401
    assert _count_threads_named("GPUMonitor") == 0
