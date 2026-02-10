# tests/test_scheduler.py
import os
import pytest
import torch
from core import SimpleScheduler

_MINIMAL = os.environ.get('VRM_MINIMAL_TEST') in {'1', 'true', 'TRUE'}

@pytest.mark.slow
@pytest.mark.network
@pytest.mark.skipif(_MINIMAL, reason="Requires real GPT-2 model download (slow, needs network)")
def test_scheduler_forward_and_predict():
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    scheduler = SimpleScheduler([model])

    x = torch.randint(0, 50257, (1, 10)).to("cpu")
    logits = scheduler.forward(x)
    assert logits.shape == (1, 10, 50257)

    pred = scheduler.predict(x)
    assert pred.shape == (1, 10)
