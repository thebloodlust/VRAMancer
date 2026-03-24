"""Tests for speculative_decoding and holographic_memory modules.

Uses VRM_MINIMAL_TEST=1 from conftest.  Mocks torch where needed.
"""

import os
import pytest

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_API_TOKEN", "testtoken")


# ═══════════════════════════════════════════════════════════════════════
# Holographic Memory (pure Python — no torch needed)
# ═══════════════════════════════════════════════════════════════════════

class TestHolographicMemory:
    """XOR parity encode/heal tests."""

    def _make_kv(self):
        try:
            from core.holographic_memory import HolographicKVManager
        except ImportError:
            # Module moved to _deprecated/
            import sys, importlib.util
            spec = importlib.util.spec_from_file_location(
                "holographic_memory",
                os.path.join(os.path.dirname(__file__), '..', '_deprecated', 'holographic_memory.py'))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            HolographicKVManager = mod.HolographicKVManager
        return HolographicKVManager()

    def test_encode_decode_no_loss(self):
        kv = self._make_kv()
        data = b"ABCDEFGHIJKLMNOP" * 10  # 160 bytes
        shards, parity = kv.encode_hologram(data, num_shards=4)
        assert len(shards) == 4
        result = kv.heal_hologram(shards, parity)
        assert result[:len(data)] == data

    def test_heal_single_shard_loss(self):
        kv = self._make_kv()
        data = os.urandom(200)
        shards, parity = kv.encode_hologram(data, num_shards=4)
        original = list(shards)

        for i in range(4):
            test_shards = list(original)
            test_shards[i] = None
            result = kv.heal_hologram(test_shards, parity)
            # Result includes padding but must start with original data
            assert result[:len(data)] == data

    def test_heal_double_loss_fails(self):
        kv = self._make_kv()
        data = os.urandom(100)
        shards, parity = kv.encode_hologram(data, num_shards=4)
        shards[0] = None
        shards[1] = None
        result = kv.heal_hologram(shards, parity)
        assert result == b""  # cannot heal >1 loss

    def test_encode_uneven_split(self):
        """Data not evenly divisible by num_shards."""
        kv = self._make_kv()
        data = b"12345678901"  # 11 bytes, 3 shards
        shards, parity = kv.encode_hologram(data, num_shards=3)
        assert len(shards) == 3
        # All padded to same length
        assert all(len(s) == len(shards[0]) for s in shards)

    def test_store_and_heal_engram(self):
        kv = self._make_kv()
        data = os.urandom(128)
        info = kv.store_engram("test-1", data, num_shards=2)
        assert info["engram_id"] == "test-1"
        assert info["num_shards"] == 2

        result = kv.heal_engram("test-1", missing_idx=0)
        assert result is not None
        assert result[:len(data)] == data

    def test_heal_engram_missing(self):
        kv = self._make_kv()
        assert kv.heal_engram("nonexistent", missing_idx=0) is None

    def test_stats(self):
        kv = self._make_kv()
        s = kv.stats()
        assert "active_engrams" in s
        assert "native" in s

    def test_single_shard(self):
        """Edge case: 1 shard = parity is the data itself."""
        kv = self._make_kv()
        data = b"single"
        shards, parity = kv.encode_hologram(data, num_shards=1)
        assert len(shards) == 1
        # Parity of 1 shard = the shard itself
        assert parity == shards[0]


# ═══════════════════════════════════════════════════════════════════════
# Speculative Decoding
# ═══════════════════════════════════════════════════════════════════════

class TestSpeculativeDecoding:
    """Tests without real torch — mock tensor operations."""

    def test_import_module(self):
        """Module imports without torch available."""
        from core.speculative_decoding import SwarmSpeculativeDecoder
        assert SwarmSpeculativeDecoder is not None

    def test_create_draft_callable_minimal(self):
        """In minimal mode, factory returns None."""
        from core.speculative_decoding import create_draft_callable
        result = create_draft_callable(backend=None)
        assert result is None

    def test_decoder_init(self):
        from core.speculative_decoding import SwarmSpeculativeDecoder
        draft = lambda ids, n: ids
        verify = lambda ids: ids
        dec = SwarmSpeculativeDecoder(
            draft_model_callable=draft,
            swarm_verify_callable=verify,
            gamma=5,
            temperature=0.5,
        )
        assert dec.gamma == 5
        assert dec.temperature == 0.5
        assert dec.total_drafted == 0
        assert dec.total_accepted == 0

    @pytest.mark.skipif(
        os.environ.get("VRM_MINIMAL_TEST") == "1",
        reason="Requires real torch for tensor operations",
    )
    def test_generate_greedy(self):
        """With real torch: greedy speculative decoding."""
        import torch
        from core.speculative_decoding import SwarmSpeculativeDecoder

        vocab_size = 100
        seq_len = 10
        gamma = 3

        def mock_draft(input_ids, num_tokens):
            return torch.randint(0, vocab_size, (1, num_tokens))

        def mock_verify(speculated_ids):
            batch, seq = speculated_ids.shape
            return torch.randn(batch, seq, vocab_size)

        dec = SwarmSpeculativeDecoder(
            draft_model_callable=mock_draft,
            swarm_verify_callable=mock_verify,
            gamma=gamma,
            temperature=0.0,
        )
        input_ids = torch.zeros(1, seq_len, dtype=torch.long)
        result = dec.generate(input_ids, max_new_tokens=5)
        assert result.shape[1] >= seq_len + 5  # at least max_new_tokens added

    @pytest.mark.skipif(
        os.environ.get("VRM_MINIMAL_TEST") == "1",
        reason="Requires real torch for tensor operations",
    )
    def test_generate_stochastic(self):
        """With real torch: stochastic speculative decoding."""
        import torch
        from core.speculative_decoding import SwarmSpeculativeDecoder

        vocab_size = 50
        gamma = 3

        def mock_draft(input_ids, num_tokens):
            return torch.randint(0, vocab_size, (1, num_tokens))

        def mock_verify(speculated_ids):
            batch, seq = speculated_ids.shape
            return torch.randn(batch, seq, vocab_size)

        dec = SwarmSpeculativeDecoder(
            draft_model_callable=mock_draft,
            swarm_verify_callable=mock_verify,
            gamma=gamma,
            temperature=0.8,
        )
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        result = dec.generate(input_ids, max_new_tokens=4)
        assert result.shape[1] >= 9

    def test_metrics_init(self):
        from core.speculative_decoding import _init_spec_metrics
        _init_spec_metrics()  # should not raise
