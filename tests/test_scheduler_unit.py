"""Tests for core/scheduler.py — SimpleScheduler stub-safe methods."""
import os
import pytest

os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")

from core.scheduler import SimpleScheduler


class TestSchedulerInit:
    def test_init_empty(self):
        s = SimpleScheduler()
        assert s is not None
        assert s.allocated_blocks() == []

    def test_init_with_blocks(self):
        blocks = [lambda x: x, lambda x: x]
        s = SimpleScheduler(blocks=blocks)
        assert len(s.blocks) == 2


class TestGPUDetection:
    def test_get_available_gpus(self):
        s = SimpleScheduler()
        gpus = s.get_available_gpus()
        assert isinstance(gpus, list)
        assert len(gpus) >= 1
        for gpu in gpus:
            assert "id" in gpu
            assert "free_vram_mb" in gpu or "free_mb" in gpu or "vram_free" in gpu

    def test_refresh_gpu_info(self):
        s = SimpleScheduler()
        s.refresh_gpu_info()
        gpus = s.get_available_gpus()
        assert len(gpus) >= 1


class TestBlockAllocation:
    def test_allocate_and_release(self):
        s = SimpleScheduler()
        block = s.allocate_block(size_mb=256, priority=5, layer_name="layer_0")
        assert block is not None
        assert block.size_mb == 256
        assert s.total_allocated_mb() >= 256
        s.release_block(block)
        assert s.total_allocated_mb() == 0

    def test_allocate_multiple(self):
        s = SimpleScheduler()
        b1 = s.allocate_block(size_mb=100, layer_name="layer_0")
        b2 = s.allocate_block(size_mb=200, layer_name="layer_1")
        assert len(s.allocated_blocks()) == 2
        assert s.total_allocated_mb() == 300
        s.release_block(b1)
        assert len(s.allocated_blocks()) == 1
        assert s.total_allocated_mb() == 200
        s.release_block(b2)

    def test_total_allocated_per_gpu(self):
        s = SimpleScheduler()
        b1 = s.allocate_block(size_mb=100, layer_name="layer_0")
        gpu_id = b1.gpu_id
        assert s.total_allocated_mb(gpu_id=gpu_id) >= 100
        s.release_block(b1)


class TestPrediction:
    def test_predict_next_layers_no_blocks(self):
        s = SimpleScheduler()
        # No blocks allocated -> returns empty or short list
        predicted = s.predict_next_layers([0, 1, 2], lookahead=3)
        assert isinstance(predicted, list)

    def test_predict_next_layers_empty_input(self):
        s = SimpleScheduler()
        predicted = s.predict_next_layers([], lookahead=2)
        assert isinstance(predicted, list)

    def test_find_alternate_gpu(self):
        s = SimpleScheduler()
        alt = s.find_alternate_gpu(exclude=0)
        assert isinstance(alt, int)
        assert alt != 0 or len(s.get_available_gpus()) == 1


class TestForward:
    def test_forward_identity_blocks(self):
        """Forward through identity blocks should return input."""
        blocks = [lambda x: x, lambda x: x]
        s = SimpleScheduler(blocks=blocks)
        result = s.forward("test_input")
        assert result == "test_input"

    def test_weighted_forward_identity(self):
        blocks = [lambda x: x]
        s = SimpleScheduler(blocks=blocks)
        result = s.weighted_forward("data")
        assert result == "data"
