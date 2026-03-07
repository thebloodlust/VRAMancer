"""Real inference test — GPT-2 end-to-end.

Downloads the smallest GPT-2 model and runs actual text generation
through the VRAMancer backend pipeline. Validates that the pipeline
produces real, non-garbage text output.

Requires: transformers, torch (CPU is fine)
Marked slow — not part of the default CI test suite.
Run explicitly: pytest tests/test_real_inference.py -m slow
"""

import os
import sys
import pytest

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Skip entirely in minimal test mode (no torch/transformers)
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("VRM_MINIMAL_TEST", "") == "1",
        reason="VRM_MINIMAL_TEST=1 — skip real model tests",
    ),
]

try:
    import torch
    import transformers  # noqa: F401
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


@pytest.fixture(scope="module")
def hf_backend():
    """Create a HuggingFaceBackend with GPT-2 loaded (shared across tests)."""
    if not _DEPS_OK:
        pytest.skip("torch/transformers not installed")
    from core.backends import HuggingFaceBackend
    backend = HuggingFaceBackend()
    backend.load_model("gpt2")
    return backend


class TestGPT2RealInference:
    """Test real GPT-2 inference through VRAMancer backends."""

    def test_model_loads_successfully(self, hf_backend):
        """Model and tokenizer should be loaded."""
        assert hf_backend.model is not None
        assert hf_backend.tokenizer is not None
        assert hf_backend.model_name == "gpt2"

    def test_generate_produces_text(self, hf_backend):
        """generate() should return a non-empty string."""
        result = hf_backend.generate("Hello, my name is", max_new_tokens=20)
        assert isinstance(result, str)
        assert len(result) > 0
        # Should produce more than just the prompt
        assert len(result) > len("Hello, my name is")

    def test_generate_respects_max_tokens(self, hf_backend):
        """Output should not exceed max_new_tokens significantly."""
        result = hf_backend.generate("The quick brown fox", max_new_tokens=5)
        assert isinstance(result, str)
        # Tokenize to check length (approximate)
        tokens = hf_backend.tokenizer.encode(result)
        prompt_tokens = hf_backend.tokenizer.encode("The quick brown fox")
        generated_count = len(tokens) - len(prompt_tokens)
        # Allow some leeway (tokenization is approximate)
        assert generated_count <= 10

    def test_infer_returns_logits(self, hf_backend):
        """infer() should return a logits tensor with correct shape."""
        input_ids = hf_backend.tokenizer.encode("Hello world", return_tensors="pt")
        output = hf_backend.infer(input_ids)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
        assert torch.is_tensor(logits)
        assert logits.dim() == 3  # [batch, seq_len, vocab_size]
        assert logits.shape[0] == 1  # batch=1
        assert logits.shape[1] == input_ids.shape[1]  # seq_len matches input

    def test_generate_stream_yields_tokens(self, hf_backend):
        """generate_stream() should yield non-empty string chunks."""
        chunks = list(hf_backend.generate_stream("Once upon a time", max_new_tokens=10))
        assert len(chunks) > 0
        full_text = "".join(chunks)
        assert len(full_text) > 0

    def test_split_model_single_gpu(self, hf_backend):
        """split_model(1) should return a single block."""
        blocks = hf_backend.split_model(num_gpus=1)
        assert len(blocks) == 1

    def test_generate_batch_produces_results(self, hf_backend):
        """generate_batch should return one result per prompt."""
        prompts = ["Hello", "World"]
        results = hf_backend.generate_batch(prompts, max_new_tokens=5)
        assert len(results) == 2
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

    def test_tokenizer_roundtrip(self, hf_backend):
        """Tokenizer encode/decode should be consistent."""
        text = "VRAMancer is a multi-GPU orchestrator"
        tokens = hf_backend.tokenizer.encode(text)
        decoded = hf_backend.tokenizer.decode(tokens)
        assert decoded == text

    def test_generate_different_prompts_different_outputs(self, hf_backend):
        """Different prompts should generally produce different outputs."""
        r1 = hf_backend.generate("The capital of France is", max_new_tokens=10)
        r2 = hf_backend.generate("def fibonacci(n):", max_new_tokens=10)
        # They should not be identical (extremely unlikely with a real model)
        assert r1 != r2
