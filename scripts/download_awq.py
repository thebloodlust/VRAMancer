#!/usr/bin/env python3
"""Download Qwen2.5-7B-Instruct-AWQ model."""
import os
os.environ['HF_HUB_OFFLINE'] = '0'

from huggingface_hub import snapshot_download

model_id = 'Qwen/Qwen2.5-7B-Instruct-AWQ'
print(f"Downloading {model_id}...")
path = snapshot_download(
    model_id,
    allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model']
)
print(f"OK: {path}")
