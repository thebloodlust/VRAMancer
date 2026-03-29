#!/bin/bash
export VRM_MINIMAL_TEST=1 VRM_DISABLE_RATE_LIMIT=1 VRM_TEST_MODE=1
cd /home/jeremie/VRAMancer/VRAMancer
pytest tests/ -q --tb=no -p no:rich --override-ini="addopts=" \
  --ignore=tests/test_backend_stubs.py \
  --ignore=tests/test_real_inference.py \
  --ignore=tests/test_api_consolidation.py \
  --ignore=tests/test_api_latency_metric.py \
  --ignore=tests/test_devices_endpoint.py \
  --ignore=tests/test_fastpath.py \
  --ignore=tests/test_fastpath_endpoints.py \
  --ignore=tests/test_heterogeneous_cluster.py \
  --ignore=tests/test_integration_flask.py \
  --ignore=tests/test_llm_transport.py \
  --ignore=tests/test_memory_promotion.py \
  --ignore=tests/test_metrics_promotions.py \
  --ignore=tests/test_multiprocess_flask.py \
  --ignore=tests/test_persistence_and_rbac.py \
  --ignore=tests/test_read_only_mode.py \
  > /home/jeremie/pytest_result.log 2>&1
echo "EXIT=$?" >> /home/jeremie/pytest_result.log
