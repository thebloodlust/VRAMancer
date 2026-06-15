#!/usr/bin/env python3
"""Tests unitaires ClusterRouter — partie SANS GPU (vendeur, variables, init, status).

Collectable par pytest (le spawn/inférence est testé séparément dans test_cluster_health
+ bench_cluster_router, qui nécessitent 2 GPU).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.cluster_router as cr


def test_detect_vendor_returns_known():
    v = cr.detect_gpu_vendor()
    assert v in ("cuda", "rocm", "mps", "cpu")


def test_visible_var_mapping():
    assert cr._VISIBLE_VAR["cuda"] == "CUDA_VISIBLE_DEVICES"
    assert cr._VISIBLE_VAR["rocm"] == "HIP_VISIBLE_DEVICES"
    assert cr._VISIBLE_VAR["mps"] is None


def test_router_default_vendors():
    r = cr.ClusterRouter("dummy-model", gpu_ids=[0, 1])
    assert len(r.vendors) == 2
    assert all(isinstance(v, str) for v in r.vendors)


def test_router_explicit_vendors():
    r = cr.ClusterRouter("dummy-model", gpu_ids=[0, 1], vendors=["cuda", "rocm"])
    assert r.vendors == ["cuda", "rocm"]


def test_status_before_start():
    r = cr.ClusterRouter("dummy-model", gpu_ids=[0])
    st = r.status()
    assert st["workers"] == 0 and st["alive"] == 0 and st["restarts"] == 0
    assert st["gpu_ids"] == [0]


def _run():
    fails = 0
    for fn in (test_detect_vendor_returns_known, test_visible_var_mapping,
               test_router_default_vendors, test_router_explicit_vendors,
               test_status_before_start):
        try:
            fn(); print(f"[OK ] {fn.__name__}")
        except AssertionError as e:
            fails += 1; print(f"[FAIL] {fn.__name__}: {e}")
    print("TOUS OK" if not fails else f"{fails} ÉCHECS")
    return fails


if __name__ == "__main__":
    sys.exit(1 if _run() else 0)
