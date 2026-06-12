"""Explicit skip test for VTP_L3 stub.

See: docs/reports/TECHNICAL_DEBT.md — VTP_L3
csrc/vtp_core.cpp : L3–L7 layers return src.clone() (stub).
"""
import pytest


@pytest.mark.skip(
    reason="VTP_L3 router stub: returns src.clone() — no real RDMA transport. "
           "L1/L2 (vtp_cuda.cu cudaMemcpyPeerAsync) functional. "
           "See: docs/reports/TECHNICAL_DEBT.md#VTP_L3"
)
def test_vtp_l3_rdma_routing():
    """Placeholder: VTP L3+ routing is a stub in csrc/vtp_core.cpp."""
    pytest.fail("not implemented")
