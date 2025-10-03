import os
import torch
from core.backends import select_backend

def test_vllm_stub():
    os.environ['VRM_BACKEND_ALLOW_STUB'] = '1'
    be = select_backend('vllm')
    model = be.load_model('dummy-model')
    out = be.infer(torch.randint(0, 100, (1,10))) if hasattr(be,'infer') else None
    assert model is not None

def test_ollama_stub():
    os.environ['VRM_BACKEND_ALLOW_STUB'] = '1'
    be = select_backend('ollama')
    model = be.load_model('dummy-ollama')
    out = be.infer(torch.randint(0, 100, (1,10))) if hasattr(be,'infer') else None
    assert model is not None
