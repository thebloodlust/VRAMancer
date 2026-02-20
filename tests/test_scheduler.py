# tests/test_scheduler.py
import os
import pytest
import torch
from unittest.mock import patch, MagicMock
from core import SimpleScheduler

_MINIMAL = os.environ.get('VRM_MINIMAL_TEST') in {'1', 'true', 'TRUE'}

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = 50257
        
    def forward(self, input_ids, **kwargs):
        batch_size, seq_len = input_ids.shape
        logits = torch.randn(batch_size, seq_len, self.vocab_size)
        return MagicMock(logits=logits)

@pytest.mark.slow
def test_scheduler_forward_and_predict():
    with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_from_pretrained:
        mock_from_pretrained.return_value = DummyModel()
        
        model = mock_from_pretrained("gpt2")
        scheduler = SimpleScheduler([model])

        x = torch.randint(0, 50257, (1, 10)).to("cpu")
        logits = scheduler.forward(x)
        assert logits.shape == (1, 10, 50257)

        pred = scheduler.predict(x)
        assert pred.shape == (1, 10)
