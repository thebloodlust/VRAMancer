"""Backward-compatibility shim — moved to core/network/network_transport.py."""
from core.network.network_transport import (  # noqa: F401
    open_low_latency_channel,
    FastHandle,
    detect_fast_interfaces,
    benchmark_interfaces,
    RDMATransport,
    GPUDirectTransport,
    ZeroCopyTCPTransport,
)

__all__ = [
    "open_low_latency_channel", "FastHandle", "detect_fast_interfaces",
    "benchmark_interfaces", "RDMATransport", "GPUDirectTransport", "ZeroCopyTCPTransport",
]
