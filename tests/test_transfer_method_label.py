"""V5 P10.2 — TransferManager._get_method_for() returns correct labels.

Regression guard for V5 P2 (honest RUST_P2P label).
"""
import os
import pytest

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")
os.environ.setdefault("VRM_DISABLE_RATE_LIMIT", "1")


def test_method_for_returns_rust_p2p_when_gpu_pipeline_cached():
    """_get_method_for() returns 'RUST_P2P' when GpuPipeline is cached for a pair."""
    pytest.importorskip("torch")
    from core.transfer_manager import TransferManager

    tm = TransferManager()
    if not hasattr(tm, "_gpu_pipelines"):
        pytest.skip("TransferManager has no _gpu_pipelines attribute")

    # Inject a sentinel to simulate a cached Rust GpuPipeline pair.
    tm._gpu_pipelines[(0, 1)] = object()
    result = tm._get_method_for(0, 1)
    assert result == "RUST_P2P", (
        f"Expected 'RUST_P2P' when GpuPipeline cached, got '{result}'"
    )


def test_method_for_falls_back_to_known_labels():
    """Without a cached pipeline, label must be one of the known fallback values."""
    pytest.importorskip("torch")
    from core.transfer_manager import TransferManager

    tm = TransferManager()
    # Use a pair that will never have a cached pipeline in test environment.
    label = tm._get_method_for(2, 3)
    known = {"CUDA_P2P", "NCCL", "CPU_STAGED"}
    # Cross-vendor label is also acceptable if bridge is present.
    assert label in known or label.startswith("CROSS_VENDOR:"), (
        f"Expected one of {known} (or CROSS_VENDOR:*), got '{label}'"
    )
