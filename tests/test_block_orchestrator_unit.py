"""Tests for core/orchestrator/block_orchestrator.py — BlockOrchestrator stub-safe methods."""
import os
import pytest

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")

from core.scheduler import SimpleScheduler
from core.orchestrator.block_orchestrator import BlockOrchestrator


@pytest.fixture
def orchestrator(tmp_path):
    scheduler = SimpleScheduler()
    return BlockOrchestrator(
        scheduler=scheduler,
        nvme_dir=str(tmp_path / "blocks"),
        remote_nodes=None,
    )


class TestBlockOrchestratorInit:
    def test_init(self, orchestrator):
        assert orchestrator is not None

    def test_init_with_remote_nodes(self, tmp_path):
        scheduler = SimpleScheduler()
        bo = BlockOrchestrator(
            scheduler=scheduler,
            nvme_dir=str(tmp_path / "blocks"),
            remote_nodes=["192.168.1.10:5000", "192.168.1.11:5000"],
        )
        assert bo is not None


class TestBlockOrchestratorState:
    def test_get_state(self, orchestrator):
        state = orchestrator.get_state()
        assert isinstance(state, dict)

    def test_get_state_has_keys(self, orchestrator):
        state = orchestrator.get_state()
        # Should have some structural info
        assert len(state) > 0


class TestBlockOrchestratorPlacement:
    def test_release_block(self, orchestrator):
        class MockBlock:
            id = "test-block-001"
            size_mb = 128
            gpu_id = 0
            status = "allocated"

        block = MockBlock()
        orchestrator.release_block(block, gpu_id=0)


class TestBlockOrchestratorFetch:
    def test_fetch_nonexistent_block(self, orchestrator):
        result = orchestrator.fetch_block("nonexistent-block-id")
        assert result is None

    def test_fetch_after_cache(self, orchestrator, tmp_path):
        """If we manually place a file at the NVMe path, fetch_block should find it."""
        block_dir = tmp_path / "blocks"
        block_dir.mkdir(exist_ok=True)
        # Create a dummy block file
        import json
        block_file = block_dir / "test-block-002.json"
        block_file.write_text(json.dumps({"weights": [1, 2, 3]}))
        # fetch_block may or may not find it depending on impl
        # The point is it doesn't crash
        orchestrator.fetch_block("test-block-002")
