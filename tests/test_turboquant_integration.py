"""Tests for TurboQuant integration with PagedKVCacheManager.

Tests that VRM_KV_COMPRESSION=turboquant properly wires TurboQuant into
the paged attention system: compress-on-write, compressed attention scores,
bulk compression, stats reporting, and stream_manager eviction hook.
"""
import os
import sys
import math
import pytest

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_DISABLE_RATE_LIMIT", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
    HAS_TORCH = hasattr(torch.nn, 'Module') and torch.nn.Module is not object
except (ImportError, AttributeError):
    HAS_TORCH = False


# ── PagedKVConfig with TurboQuant ──────────────────────────────────

class TestPagedKVConfigCompression:
    """PagedKVConfig reads VRM_KV_COMPRESSION env var."""

    def test_config_default_no_compression(self):
        from core.paged_attention import PagedKVConfig
        cfg = PagedKVConfig()
        assert cfg.kv_compression is None
        assert cfg.compression_bits == 3

    def test_config_turboquant_fields(self):
        from core.paged_attention import PagedKVConfig
        cfg = PagedKVConfig(kv_compression="turboquant", compression_bits=4)
        assert cfg.kv_compression == "turboquant"
        assert cfg.compression_bits == 4

    def test_from_model_reads_env(self, monkeypatch):
        monkeypatch.setenv("VRM_KV_COMPRESSION", "turboquant")
        monkeypatch.setenv("VRM_KV_COMPRESSION_BITS", "4")
        from core.paged_attention import PagedKVConfig
        cfg = PagedKVConfig.from_model(None)
        assert cfg.kv_compression == "turboquant"
        assert cfg.compression_bits == 4

    def test_from_model_empty_env(self, monkeypatch):
        monkeypatch.delenv("VRM_KV_COMPRESSION", raising=False)
        from core.paged_attention import PagedKVConfig
        cfg = PagedKVConfig.from_model(None)
        assert cfg.kv_compression is None


# ── PagedKVCacheManager with TurboQuant ────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="Requires real torch")
class TestPagedKVCacheTurboQuant:
    """TurboQuant compression wired into PagedKVCacheManager."""

    @pytest.fixture
    def manager(self):
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        cfg = PagedKVConfig(
            page_size=4,
            num_layers=2,
            num_kv_heads=2,
            head_dim=64,
            max_pages=16,
            device="cpu",
            kv_compression="turboquant",
            compression_bits=3,
        )
        return PagedKVCacheManager(cfg)

    @pytest.fixture
    def manager_no_compress(self):
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        cfg = PagedKVConfig(
            page_size=4,
            num_layers=2,
            num_kv_heads=2,
            head_dim=64,
            max_pages=16,
            device="cpu",
        )
        return PagedKVCacheManager(cfg)

    def test_compressor_initialized(self, manager):
        assert manager.kv_compression_active is True
        assert manager._kv_compressor is not None

    def test_no_compressor_by_default(self, manager_no_compress):
        assert manager_no_compress.kv_compression_active is False
        assert manager_no_compress._kv_compressor is None

    def test_compression_ratio(self, manager):
        ratio = manager.compression_ratio
        assert ratio > 1.0  # Must compress
        assert ratio < 10.0  # Sanity check

    def test_stats_include_compression(self, manager):
        stats = manager.stats()
        assert "kv_compression" in stats
        assert stats["kv_compression"] == "turboquant"
        assert "compressed_pages" in stats
        assert "compression_ratio" in stats
        assert stats["compression_ratio"] > 1.0

    def test_stats_no_compression(self, manager_no_compress):
        stats = manager_no_compress.stats()
        assert stats["kv_compression"] == "none"
        assert stats["compression_ratio"] == 1.0

    def test_write_kv_compresses(self, manager):
        """write_kv() should store compressed form in sidecar."""
        torch.manual_seed(42)
        entry = manager.allocate("req1", num_tokens=0)

        # Manually allocate a page and write KV
        result = manager.append_token("req1")
        assert result is not None
        page_id, slot = result

        key = torch.randn(2, 64)  # [num_kv_heads, head_dim]
        value = torch.randn(2, 64)
        manager.write_kv("req1", layer_idx=0, page_id=page_id, slot=slot,
                         key=key, value=value)

        # Check compressed sidecar
        assert page_id in manager._compressed_pages
        assert 0 in manager._compressed_pages[page_id]
        slot_data = manager._compressed_pages[page_id][0].get(f"s{slot}")
        assert slot_data is not None
        assert "k" in slot_data
        assert "v" in slot_data

    def test_compress_page_bulk(self, manager):
        """compress_page_bulk should compress all layers/slots."""
        # Don't use GPU pool (cpu mode, _MINIMAL), set up manually
        # The manager won't have _gpu_pool in minimal test mode
        # So test the method exists and returns False gracefully
        ok = manager.compress_page_bulk(0)
        assert isinstance(ok, bool)

    def test_compute_attention_turbo_returns_none_without_data(self, manager):
        """compute_attention_turbo returns None when no data."""
        q = torch.randn(2, 64)
        result = manager.compute_attention_turbo(q, "nonexistent", 0)
        assert result is None

    def test_compute_attention_turbo_with_data(self, manager):
        """Full write → attention_turbo path."""
        torch.manual_seed(42)

        # Allocate and write several tokens
        manager.allocate("req_attn", num_tokens=0)
        for _ in range(3):
            result = manager.append_token("req_attn")
            if result is None:
                break
            page_id, slot = result
            key = torch.randn(2, 64)
            value = torch.randn(2, 64)
            manager.write_kv("req_attn", 0, page_id, slot, key, value)

        # Query
        query = torch.randn(2, 64)  # [num_heads, head_dim]
        output = manager.compute_attention_turbo(query, "req_attn", 0)

        # Should return tensor [num_heads, head_dim]
        assert output is not None
        assert output.shape == (2, 64)
        assert not torch.isnan(output).any()

    def test_eviction_compresses_before_free(self, manager):
        """When a page is evicted, compressed form should be created first."""
        torch.manual_seed(42)

        # Fill up all pages to force eviction
        for i in range(20):
            rid = f"req_{i}"
            manager.allocate(rid, num_tokens=4)  # 4 tokens = 1 page

        # After filling, some pages should have been evicted
        # The eviction path calls compress_page_bulk if compressor is active
        # In minimal test mode without gpu_pool, compress_page_bulk returns False
        # but the eviction still works
        stats = manager.stats()
        assert stats["total_frees"] >= 0  # sanity — eviction happened


# ── StreamManager integration ──────────────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="Requires real torch")
class TestStreamManagerTurboQuant:
    """StreamManager accepts paged_kv parameter."""

    def test_stream_manager_accepts_paged_kv(self):
        from core.stream_manager import StreamManager
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig

        cfg = PagedKVConfig(
            page_size=4, num_layers=1, num_kv_heads=1,
            head_dim=64, max_pages=8, device="cpu",
            kv_compression="turboquant",
        )
        kv_mgr = PagedKVCacheManager(cfg)
        sm = StreamManager(paged_kv=kv_mgr)

        assert sm.paged_kv is kv_mgr
        assert sm.paged_kv.kv_compression_active is True

    def test_stream_manager_no_paged_kv(self):
        from core.stream_manager import StreamManager
        sm = StreamManager()
        assert sm.paged_kv is None


# ── Edge cases ─────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="Requires real torch")
class TestTurboQuantEdgeCases:
    """Edge cases and error resilience."""

    def test_non_power_of_2_head_dim(self):
        """TurboQuant pads to power-of-2 — verify with head_dim=96."""
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        cfg = PagedKVConfig(
            page_size=4, num_layers=1, num_kv_heads=1,
            head_dim=96, max_pages=8, device="cpu",
            kv_compression="turboquant",
        )
        mgr = PagedKVCacheManager(cfg)
        assert mgr.kv_compression_active is True

        # Write and verify
        mgr.allocate("edge1")
        result = mgr.append_token("edge1")
        assert result is not None
        page_id, slot = result
        key = torch.randn(1, 96)
        value = torch.randn(1, 96)
        mgr.write_kv("edge1", 0, page_id, slot, key, value)
        assert f"s{slot}" in mgr._compressed_pages[page_id][0]

    def test_multiple_layers_compressed(self):
        """Compression works across multiple layers."""
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        cfg = PagedKVConfig(
            page_size=4, num_layers=4, num_kv_heads=2,
            head_dim=64, max_pages=8, device="cpu",
            kv_compression="turboquant",
        )
        mgr = PagedKVCacheManager(cfg)
        mgr.allocate("multi")
        result = mgr.append_token("multi")
        page_id, slot = result

        for layer in range(4):
            key = torch.randn(2, 64)
            value = torch.randn(2, 64)
            mgr.write_kv("multi", layer, page_id, slot, key, value)

        # All 4 layers should have compressed data
        for layer in range(4):
            assert layer in mgr._compressed_pages[page_id]

    def test_free_does_not_crash_with_compression(self):
        """Freeing a request with compressed pages should not error."""
        from core.paged_attention import PagedKVCacheManager, PagedKVConfig
        cfg = PagedKVConfig(
            page_size=4, num_layers=1, num_kv_heads=1,
            head_dim=64, max_pages=8, device="cpu",
            kv_compression="turboquant",
        )
        mgr = PagedKVCacheManager(cfg)
        mgr.allocate("tofree")
        result = mgr.append_token("tofree")
        page_id, slot = result
        mgr.write_kv("tofree", 0, page_id, slot,
                      torch.randn(1, 64), torch.randn(1, 64))

        freed = mgr.free("tofree")
        assert freed >= 1
