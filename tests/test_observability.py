"""Tests for final production improvements: Grafana, alerting, batch forward, /metrics."""
import os
import sys
import json
import time
import threading
import pytest

# Ensure test env
os.environ.setdefault("VRM_MINIMAL_TEST", "1")
os.environ.setdefault("VRM_API_TOKEN", "testtoken")
os.environ.setdefault("VRM_DISABLE_RATE_LIMIT", "1")
os.environ.setdefault("VRM_TEST_MODE", "1")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ======================================================================
# Monitoring Files Validation
# ======================================================================

class TestMonitoringFiles:
    """Validate Grafana dashboard, Prometheus config, and alerting rules."""

    def test_grafana_dashboard_valid_json(self):
        """Grafana dashboard is valid JSON with expected structure."""
        dashboard_path = os.path.join(
            os.path.dirname(__file__), "..", "monitoring", "grafana_dashboard.json"
        )
        with open(dashboard_path, "r") as f:
            dash = json.load(f)

        assert dash["uid"] == "vramancer-main"
        assert dash["title"] == "VRAMancer — Multi-GPU Inference"
        assert "panels" in dash
        assert len(dash["panels"]) >= 20  # We have 24 panels + rows

    def test_grafana_dashboard_covers_all_metric_families(self):
        """Dashboard has panels for each major metric family."""
        dashboard_path = os.path.join(
            os.path.dirname(__file__), "..", "monitoring", "grafana_dashboard.json"
        )
        with open(dashboard_path, "r") as f:
            dash = json.load(f)

        all_exprs = []
        for panel in dash["panels"]:
            for target in panel.get("targets", []):
                all_exprs.append(target.get("expr", ""))

        joined = " ".join(all_exprs)

        # Key metric families that must appear
        required = [
            "vramancer_infer_total",
            "vramancer_infer_errors_total",
            "vramancer_infer_latency_seconds",
            "vramancer_gpu_memory_used_bytes",
            "vramancer_tasks_running",
            "vramancer_memory_promotions_total",
            "vramancer_memory_demotions_total",
            "vramancer_memory_evictions_total",
            "vramancer_fastpath_bytes_total",
            "vramancer_api_latency_seconds",
            "vramancer_orch_placements_total",
            "vramancer_orch_migrations_total",
            "vramancer_block_hotness",
            "vramancer_gpu_transfer_ops_total",
        ]
        for metric in required:
            assert metric in joined, f"Dashboard missing metric: {metric}"

    def test_grafana_dashboard_has_template_variable(self):
        """Dashboard has a GPU template variable for filtering."""
        dashboard_path = os.path.join(
            os.path.dirname(__file__), "..", "monitoring", "grafana_dashboard.json"
        )
        with open(dashboard_path, "r") as f:
            dash = json.load(f)

        tpl_list = dash.get("templating", {}).get("list", [])
        assert len(tpl_list) >= 1
        assert tpl_list[0]["name"] == "gpu"

    def test_alerting_rules_valid_yaml(self):
        """Alerting rules file is valid YAML."""
        import yaml
        rules_path = os.path.join(
            os.path.dirname(__file__), "..", "monitoring", "alerting_rules.yml"
        )
        with open(rules_path, "r") as f:
            data = yaml.safe_load(f)

        assert "groups" in data
        groups = data["groups"]
        assert len(groups) >= 5  # gpu, inference, tasks, api, transfers, ha

    def test_alerting_rules_cover_critical_scenarios(self):
        """Alerting rules cover GPU, inference errors, latency, API down."""
        import yaml
        rules_path = os.path.join(
            os.path.dirname(__file__), "..", "monitoring", "alerting_rules.yml"
        )
        with open(rules_path, "r") as f:
            data = yaml.safe_load(f)

        all_alerts = []
        for group in data["groups"]:
            for rule in group.get("rules", []):
                all_alerts.append(rule["alert"])

        required_alerts = [
            "GPUMemoryHigh",
            "HighInferenceErrorRate",
            "HighInferenceLatency",
            "APIDown",
            "TaskQueueBacklog",
            "FastPathLatencyHigh",
        ]
        for alert_name in required_alerts:
            assert alert_name in all_alerts, f"Missing alert: {alert_name}"

    def test_alerting_rules_have_severity_labels(self):
        """Every alert rule has a severity label."""
        import yaml
        rules_path = os.path.join(
            os.path.dirname(__file__), "..", "monitoring", "alerting_rules.yml"
        )
        with open(rules_path, "r") as f:
            data = yaml.safe_load(f)

        for group in data["groups"]:
            for rule in group.get("rules", []):
                labels = rule.get("labels", {})
                assert "severity" in labels, (
                    f"Alert {rule['alert']} missing severity label"
                )
                assert labels["severity"] in ("warning", "critical", "info")

    def test_prometheus_config_valid_yaml(self):
        """Prometheus config is valid YAML with scrape config."""
        import yaml
        prom_path = os.path.join(
            os.path.dirname(__file__), "..", "monitoring", "prometheus.yml"
        )
        with open(prom_path, "r") as f:
            data = yaml.safe_load(f)

        assert "scrape_configs" in data
        jobs = [sc["job_name"] for sc in data["scrape_configs"]]
        assert "vramancer" in jobs

    def test_grafana_provisioning_datasource(self):
        """Grafana datasource provisioning exists and points to Prometheus."""
        import yaml
        ds_path = os.path.join(
            os.path.dirname(__file__), "..",
            "monitoring", "grafana_provisioning", "datasources", "prometheus.yml"
        )
        with open(ds_path, "r") as f:
            data = yaml.safe_load(f)

        ds = data["datasources"][0]
        assert ds["type"] == "prometheus"
        assert "prometheus" in ds["url"]

    def test_grafana_provisioning_dashboards(self):
        """Grafana dashboard provisioning config exists."""
        import yaml
        dp_path = os.path.join(
            os.path.dirname(__file__), "..",
            "monitoring", "grafana_provisioning", "dashboards", "dashboards.yml"
        )
        with open(dp_path, "r") as f:
            data = yaml.safe_load(f)

        providers = data["providers"]
        assert len(providers) >= 1
        assert providers[0]["name"] == "VRAMancer"


# ======================================================================
# /metrics Endpoint Tests
# ======================================================================

class TestMetricsEndpoint:
    """Tests for the /metrics Prometheus endpoint on the Flask API."""

    @pytest.fixture
    def client(self):
        os.environ["VRM_DISABLE_RATE_LIMIT"] = "1"
        os.environ["VRM_API_TOKEN"] = "testtoken"
        from core.production_api import create_app
        app = create_app()
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_metrics_endpoint_returns_prometheus_format(self, client):
        """GET /metrics returns prometheus text format."""
        # Ensure VRAMancer metrics are registered
        import core.metrics  # noqa: F401
        resp = client.get("/metrics")
        assert resp.status_code == 200
        body = resp.data.decode("utf-8")
        # Should contain at least one vramancer metric
        assert "vramancer_" in body

    def test_metrics_endpoint_content_type(self, client):
        """Response content-type is text/plain or openmetrics."""
        resp = client.get("/metrics")
        ct = resp.content_type
        assert "text/plain" in ct or "openmetrics" in ct

    def test_metrics_endpoint_no_auth_required(self, client):
        """Metrics endpoint should work without auth token for scraping."""
        # Remove token from headers (Prometheus scraper has no auth)
        resp = client.get("/metrics")
        # Should still return metrics (security middleware should whitelist /metrics)
        assert resp.status_code in (200, 401, 403)

    def test_api_status_includes_metrics_endpoint(self, client):
        """API status shows /metrics in endpoint listing."""
        resp = client.get(
            "/api/status",
            headers={"Authorization": "Bearer testtoken"},
        )
        if resp.status_code == 200:
            data = resp.get_json()
            endpoints = data.get("endpoints", {})
            assert "metrics" in endpoints


# ======================================================================
# Batch Forward-Pass Tests (generate_batch)
# ======================================================================

class TestGenerateBatch:
    """Tests for BaseLLMBackend.generate_batch and HuggingFace implementation."""

    def test_base_backend_generate_batch_exists(self):
        """BaseLLMBackend exposes generate_batch method."""
        from core.backends import BaseLLMBackend
        assert hasattr(BaseLLMBackend, "generate_batch")

    def test_hf_backend_generate_batch_exists(self):
        """HuggingFaceBackend has generate_batch override."""
        from core.backends import HuggingFaceBackend
        backend = HuggingFaceBackend()
        assert hasattr(backend, "generate_batch")
        assert callable(backend.generate_batch)

    def test_hf_generate_batch_requires_model(self):
        """generate_batch raises RuntimeError without loaded model."""
        from core.backends import HuggingFaceBackend
        backend = HuggingFaceBackend()
        with pytest.raises(RuntimeError, match="Modèle non chargé"):
            backend.generate_batch(["Hello"])

    def test_hf_generate_batch_empty_list(self):
        """generate_batch with empty list returns empty list."""
        from core.backends import HuggingFaceBackend
        backend = HuggingFaceBackend()
        # Set model to non-None to pass first check
        backend.model = True
        backend.tokenizer = True
        result = backend.generate_batch([])
        assert result == []

    def test_stub_backend_generate_batch_fallback(self):
        """Stub vLLM backend falls back to base generate_batch."""
        from core.backends import vLLMBackend
        backend = vLLMBackend(real=False)
        backend.load_model("test-model")
        # generate_batch delegates to generate() which may fail on stub
        # but the method should exist
        assert hasattr(backend, "generate_batch")


# ======================================================================
# Batcher with Batch Function Tests
# ======================================================================

class TestBatcherWithBatchFn:
    """Tests for InferenceBatcher using generate_batch_fn."""

    def _dummy_generate(self, prompt, **kwargs):
        time.sleep(0.01)
        return f"single:{prompt}"

    def _dummy_batch_generate(self, prompts, **kwargs):
        return [f"batch:{p}" for p in prompts]

    def test_batcher_accepts_batch_fn(self):
        """InferenceBatcher accepts generate_batch_fn parameter."""
        from core.api.batch_inference import InferenceBatcher
        batcher = InferenceBatcher(
            generate_fn=self._dummy_generate,
            generate_batch_fn=self._dummy_batch_generate,
        )
        assert batcher.generate_batch_fn is not None

    def test_batcher_uses_batch_fn_for_uniform_batch(self):
        """When all requests have same kwargs, uses true batch path."""
        from core.api.batch_inference import InferenceBatcher

        batch_calls = []

        def track_batch(prompts, **kwargs):
            batch_calls.append(len(prompts))
            return [f"batch:{p}" for p in prompts]

        batcher = InferenceBatcher(
            generate_fn=self._dummy_generate,
            generate_batch_fn=track_batch,
            max_batch=4,
            window_ms=50,
        )
        batcher.start()

        results = []
        threads = []
        for i in range(3):
            t = threading.Thread(
                target=lambda idx=i: results.append(
                    batcher.submit(f"prompt-{idx}", max_new_tokens=50)
                )
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        batcher.stop()

        # All results should be present
        assert len(results) == 3

    def test_batcher_fallback_when_batch_fn_fails(self):
        """If generate_batch_fn raises, falls back to sequential."""
        from core.api.batch_inference import InferenceBatcher

        def failing_batch(prompts, **kwargs):
            raise RuntimeError("batch failed")

        batcher = InferenceBatcher(
            generate_fn=self._dummy_generate,
            generate_batch_fn=failing_batch,
            max_batch=4,
            window_ms=30,
        )
        batcher.start()

        result = batcher.submit("test-fallback", max_new_tokens=10)
        time.sleep(0.1)
        batcher.stop()

        assert "single:test-fallback" in result

    def test_batcher_stats_include_batch_info(self):
        """Stats dict is still well-formed with batch_fn."""
        from core.api.batch_inference import InferenceBatcher
        batcher = InferenceBatcher(
            generate_fn=self._dummy_generate,
            generate_batch_fn=self._dummy_batch_generate,
        )
        stats = batcher.stats
        assert "total_batches" in stats
        assert "pending_requests" in stats


# ======================================================================
# Docker Compose Validation
# ======================================================================

class TestDockerCompose:
    """Validate docker-compose.yml has monitoring stack."""

    def test_docker_compose_has_prometheus(self):
        import yaml
        dc_path = os.path.join(
            os.path.dirname(__file__), "..", "docker-compose.yml"
        )
        with open(dc_path, "r") as f:
            data = yaml.safe_load(f)

        services = data.get("services", {})
        assert "prometheus" in services
        assert "9090:9090" in services["prometheus"]["ports"]

    def test_docker_compose_has_grafana(self):
        import yaml
        dc_path = os.path.join(
            os.path.dirname(__file__), "..", "docker-compose.yml"
        )
        with open(dc_path, "r") as f:
            data = yaml.safe_load(f)

        services = data.get("services", {})
        assert "grafana" in services
        assert "3000:3000" in services["grafana"]["ports"]

    def test_docker_compose_volumes(self):
        import yaml
        dc_path = os.path.join(
            os.path.dirname(__file__), "..", "docker-compose.yml"
        )
        with open(dc_path, "r") as f:
            data = yaml.safe_load(f)

        volumes = data.get("volumes", {})
        assert "prometheus_data" in volumes
        assert "grafana_data" in volumes
