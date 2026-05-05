"""Minimal coverage tests for TransferManager (P8.2)."""
import pytest


def test_transfer_manager_init():
    from core.transfer_manager import TransferManager
    tm = TransferManager()
    assert tm is not None
    assert hasattr(tm, '_transfer_streams')


def test_transfer_manager_method_for_invalid():
    from core.transfer_manager import TransferManager
    tm = TransferManager()
    method = tm._get_method_for(0, 99)
    assert method in ('CUDA_P2P', 'NCCL', 'CPU_STAGED') or method.startswith('CROSS_VENDOR')
