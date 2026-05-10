"""Tests for auto-detect helpers (recommend_backend + invalidate_device_cache)."""

from __future__ import annotations

import pytest

from core import auto_detect


@pytest.fixture
def all_available():
    return {"huggingface": True, "vllm": True, "llamacpp": True, "ollama": True}


def test_recommend_backend_gguf_extension(all_available):
    assert auto_detect.recommend_backend("path/model.gguf", available=all_available) == "llamacpp"


def test_recommend_backend_gguf_repo_name(all_available):
    assert auto_detect.recommend_backend("TheBloke/Mistral-7B-GGUF", available=all_available) == "llamacpp"


def test_recommend_backend_q4_path(all_available):
    assert auto_detect.recommend_backend("repo/q4_K_M.bin", available=all_available) == "llamacpp"


def test_recommend_backend_ollama_scheme(all_available):
    assert auto_detect.recommend_backend("ollama://llama3:8b", available=all_available) == "ollama"
    assert auto_detect.recommend_backend("ollama:mistral", available=all_available) == "ollama"


def test_recommend_backend_awq_uses_vllm(all_available):
    assert auto_detect.recommend_backend("TheBloke/Mistral-7B-AWQ", available=all_available) == "vllm"


def test_recommend_backend_gptq_uses_vllm(all_available):
    assert auto_detect.recommend_backend("TheBloke/Llama-2-GPTQ", available=all_available) == "vllm"


def test_recommend_backend_fp8_uses_vllm(all_available):
    assert auto_detect.recommend_backend("neuralmagic/Mistral-7B-FP8", available=all_available) == "vllm"


def test_recommend_backend_default_huggingface(all_available):
    assert auto_detect.recommend_backend("mistralai/Mistral-7B-v0.1", available=all_available) == "huggingface"


def test_recommend_backend_fallback_when_unavailable():
    # GGUF requested but llamacpp not installed → fall back to huggingface
    avail = {"huggingface": True, "vllm": False, "llamacpp": False, "ollama": False}
    assert auto_detect.recommend_backend("model.gguf", available=avail) == "huggingface"


def test_recommend_backend_fallback_chain_vllm_to_hf():
    avail = {"huggingface": True, "vllm": False, "llamacpp": False, "ollama": False}
    assert auto_detect.recommend_backend("repo/AWQ", available=avail) == "huggingface"
