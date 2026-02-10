# tests/test_utils.py
import os
import pytest
import torch
from core import get_tokenizer, get_device_type, assign_block_to_device

_MINIMAL = os.environ.get('VRM_MINIMAL_TEST') in {'1', 'true', 'TRUE'}

@pytest.mark.skipif(_MINIMAL, reason="Requires real torch (not VRM_MINIMAL_TEST stubs)")
def test_get_tokenizer():
    tokenizer = get_tokenizer("gpt2")
    assert tokenizer is not None
    assert hasattr(tokenizer, "tokenize")

@pytest.mark.skipif(_MINIMAL, reason="Requires real torch.device")
def test_get_device_type():
    dev = get_device_type(0)
    assert isinstance(dev, torch.device)

@pytest.mark.skipif(_MINIMAL, reason="Requires real torch.nn and CUDA/CPU device")
def test_assign_block_to_device():
    model = torch.nn.Linear(10, 10)
    device = get_device_type(0)
    model.to(device)
    moved = assign_block_to_device(model, 0)
    assert moved is model
    assert moved.device == device
