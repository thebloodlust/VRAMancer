"""Minimal coverage tests for triton_sampling fused_sample paths (P8.3)."""
import pytest


def test_fused_sample_greedy():
    pytest.importorskip("torch")
    import torch
    from core.triton_sampling import fused_sample
    logits = torch.randn(1, 1000)
    out = fused_sample(logits, greedy=True)
    assert out.shape == (1, 1)


def test_fused_sample_topk():
    pytest.importorskip("torch")
    import torch
    from core.triton_sampling import fused_sample
    logits = torch.randn(2, 1000)
    out = fused_sample(logits, temperature=1.0, top_k=50, top_p=0.9)
    assert out.shape == (2, 1)
    assert (out >= 0).all() and (out < 1000).all()
