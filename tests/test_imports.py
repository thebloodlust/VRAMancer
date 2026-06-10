"""Validate that all core modules import without optional dependencies.

This test runs in CI with VRM_MINIMAL_TEST=1 and no GPU/CUDA.
It ensures the package is importable on a minimal Python install
(only the base pyproject.toml dependencies, `pip install -e .`).
"""
import importlib
import pytest


# Modules that are safe to import without heavy deps (torch, triton, etc.)
# They must handle missing deps gracefully with try/except.
CORE_MODULES = [
    "core.env_flags",
    "core.config",
    "core.logger",
    "core.utils",
    "core.metrics",
    "core.health",
    "core.scheduler",
    "core.block_router",
    "core.block_metadata",
    "core.compressor",
    "core.compute_engine",
    "core.monitor",
    "core.memory_block",
    "core.memory_balancer",
    "core.memory_monitor",
    "core.model_hub",
    "core.model_splitter",
    "core.stream_manager",
    "core.transfer_manager",
    "core.inference_pipeline",
    "core.backends",
    "core.backends_llamacpp",
    "core.backends_ollama",
    "core.backends_vllm",
    "core.gpu_interface",
    "core.gpu_fault_tolerance",
    "core.persistence",
    "core.swarm_ledger",
    "core.telemetry",
    "core.tracing",
    "core.tokenizer",
    "core.benchmark",
    "core.continuous_batcher",
    "core.paged_attention",
    "core.hetero_config",
    "core.cross_node",
    "core.speculative_decoding",
    "core.transport_factory",
    "core.webgpu_backend",
    "core.production_api",
]

# Modules in experimental/: unvalidated on real hardware, unstable APIs.
# Must still import cleanly (with VRM_EXPERIMENTAL=1 to silence the warning).
EXPERIMENTAL_MODULES = [
    "experimental.vram_lending",
    "experimental.hierarchical_memory",
    "experimental.cross_vendor_bridge",
    "experimental.wake_on_inference",
    "experimental.aitp_protocol",
    "experimental.aitp_fec",
    "experimental.nat_traversal",
    "experimental.fibre_fastpath",
    "experimental.cluster_discovery",
]

NETWORK_MODULES = [
    "core.network",
]

SECURITY_MODULES = [
    "core.security",
]


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_import(module):
    """Each core module must import without raising."""
    mod = importlib.import_module(module)
    assert mod is not None


@pytest.mark.parametrize("module", EXPERIMENTAL_MODULES)
def test_experimental_import(module, monkeypatch):
    """Experimental modules must still import cleanly behind the flag."""
    monkeypatch.setenv("VRM_EXPERIMENTAL", "1")
    mod = importlib.import_module(module)
    assert mod is not None


@pytest.mark.parametrize("module", NETWORK_MODULES)
def test_network_import(module):
    mod = importlib.import_module(module)
    assert mod is not None


@pytest.mark.parametrize("module", SECURITY_MODULES)
def test_security_import(module):
    mod = importlib.import_module(module)
    assert mod is not None


def test_env_flags_properties():
    """Verify env_flags facade is functional."""
    from core.env_flags import flags
    # Basic types
    assert isinstance(flags.MINIMAL_TEST, bool)
    assert isinstance(flags.API_PORT, int)
    assert isinstance(flags.GENERATE_TIMEOUT, float)
    assert isinstance(flags.PARALLEL_MODE, str)
    # repr doesn't crash
    assert "VRM" in repr(flags)


def test_dashboard_imports():
    """Dashboard CLI launcher must be importable."""
    try:
        from dashboard import launch_cli_dashboard
        assert callable(launch_cli_dashboard)
    except ImportError:
        pytest.skip("dashboard not installed")
