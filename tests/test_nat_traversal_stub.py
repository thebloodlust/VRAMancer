"""Explicit skip tests for NAT hole punch and STUN.

See: docs/reports/TECHNICAL_DEBT.md — NAT_HOLE_PUNCH
core/network/nat_traversal.py: STUN RFC 5389 real; UDP hole punch + TURN relay = stubs.
"""
import pytest


@pytest.mark.skipif(
    True,
    reason="NAT hole punch / TURN relay non testé en WAN réel. "
           "STUN RFC 5389 fonctionnel. See TECHNICAL_DEBT.md#NAT_HOLE_PUNCH"
)
def test_nat_punch_wan():
    pytest.fail("not implemented")


def test_stun_basic():
    """STUN RFC 5389 implemented and reachable in LAN."""
    pytest.skip("requires reachable STUN server (binding test only in CI)")
