# tests/test_scheduler.py
import torch
from transformers import AutoModelForCausalLM
from core import SimpleScheduler

def test_scheduler_forward_and_predict():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    scheduler = SimpleScheduler([model])

    x = torch.randint(0, 50257, (1, 10)).to("cpu")
    logits = scheduler.forward(x)
    assert logits.shape == (1, 10, 50257)

    pred = scheduler.predict(x)
    assert pred.shape == (1, 10)
