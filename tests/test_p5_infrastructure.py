"""Tests for P5 — Infrastructure items.

P5.1: Rust CI (build-rust.yml) — validated by YAML structure, not runnable in unit tests.
P5.2: Schema versioning — SQLite migration system in persistence.py
P5.3: Config hot-reload hooks — register/unregister/notify in config.py
P5.4: Metrics lifecycle — reset_metrics() clears gauges
"""
import os
import sys
import json
import tempfile
import pytest

# ── P5.1: Rust CI YAML validation ────────────────────────────────────────

class TestRustCI:
    """Validate build-rust.yml structure."""

    def test_workflow_file_exists(self):
        ci_path = os.path.join(
            os.path.dirname(__file__), "..", ".github", "workflows", "build-rust.yml"
        )
        assert os.path.isfile(ci_path), "build-rust.yml missing"

    def test_workflow_has_lint_job(self):
        ci_path = os.path.join(
            os.path.dirname(__file__), "..", ".github", "workflows", "build-rust.yml"
        )
        content = open(ci_path).read()
        assert "cargo clippy" in content
        assert "cargo fmt" in content

    def test_workflow_has_test_job(self):
        ci_path = os.path.join(
            os.path.dirname(__file__), "..", ".github", "workflows", "build-rust.yml"
        )
        content = open(ci_path).read()
        assert "cargo test" in content

    def test_workflow_has_python_import_verify(self):
        ci_path = os.path.join(
            os.path.dirname(__file__), "..", ".github", "workflows", "build-rust.yml"
        )
        content = open(ci_path).read()
        assert "import vramancer_rust" in content

    def test_workflow_has_rust_toolchain(self):
        ci_path = os.path.join(
            os.path.dirname(__file__), "..", ".github", "workflows", "build-rust.yml"
        )
        content = open(ci_path).read()
        assert "dtolnay/rust-toolchain" in content


# ── P5.2: Schema versioning ──────────────────────────────────────────────

class TestSchemaVersioning:
    """Test persistence.py migration system."""

    def test_current_schema_version_constant(self):
        from core.persistence import CURRENT_SCHEMA_VERSION
        assert CURRENT_SCHEMA_VERSION >= 2

    def test_get_schema_version_no_db(self):
        """Without VRM_SQLITE_PATH, returns 0."""
        old = os.environ.pop("VRM_SQLITE_PATH", None)
        try:
            from core import persistence
            persistence._DB_PATH = None
            assert persistence.get_schema_version() == 0
        finally:
            if old:
                os.environ["VRM_SQLITE_PATH"] = old

    def test_fresh_db_migration(self):
        """A fresh database should auto-migrate to CURRENT_SCHEMA_VERSION."""
        import sqlite3
        from core import persistence

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        old_path = persistence._DB_PATH
        try:
            persistence._DB_PATH = db_path
            os.environ["VRM_SQLITE_PATH"] = db_path
            persistence._ensure()

            conn = sqlite3.connect(db_path)
            # schema_version table should exist
            cur = conn.cursor()
            cur.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
            row = cur.fetchone()
            assert row is not None
            assert row[0] == persistence.CURRENT_SCHEMA_VERSION

            # workflows table should have created_at column
            cur.execute("PRAGMA table_info(workflows)")
            columns = [r[1] for r in cur.fetchall()]
            assert "created_at" in columns
            assert "id" in columns
            assert "data" in columns
            conn.close()
        finally:
            persistence._DB_PATH = old_path
            if old_path:
                os.environ["VRM_SQLITE_PATH"] = old_path
            else:
                os.environ.pop("VRM_SQLITE_PATH", None)
            os.unlink(db_path)

    def test_legacy_v1_migration(self):
        """A v1 database (workflows only, no schema_version) should be migrated."""
        import sqlite3
        from core import persistence

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        # Create a v1-style database manually
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE workflows(id TEXT PRIMARY KEY, data TEXT)")
        conn.execute("INSERT INTO workflows(id, data) VALUES('w1', '{\"id\":\"w1\"}')")
        conn.commit()
        conn.close()

        old_path = persistence._DB_PATH
        try:
            persistence._DB_PATH = db_path
            os.environ["VRM_SQLITE_PATH"] = db_path
            persistence._ensure()

            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            # schema_version should now exist at v2
            cur.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1")
            assert cur.fetchone()[0] == 2

            # Old data should survive
            cur.execute("SELECT data FROM workflows WHERE id='w1'")
            row = cur.fetchone()
            assert row is not None
            assert json.loads(row[0])["id"] == "w1"

            # created_at column should exist
            cur.execute("PRAGMA table_info(workflows)")
            columns = [r[1] for r in cur.fetchall()]
            assert "created_at" in columns
            conn.close()
        finally:
            persistence._DB_PATH = old_path
            if old_path:
                os.environ["VRM_SQLITE_PATH"] = old_path
            else:
                os.environ.pop("VRM_SQLITE_PATH", None)
            os.unlink(db_path)

    def test_idempotent_migration(self):
        """Running _ensure() twice should not fail."""
        import sqlite3
        from core import persistence

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        old_path = persistence._DB_PATH
        try:
            persistence._DB_PATH = db_path
            os.environ["VRM_SQLITE_PATH"] = db_path
            persistence._ensure()
            persistence._ensure()  # second call should be no-op
            assert persistence.get_schema_version() == persistence.CURRENT_SCHEMA_VERSION
        finally:
            persistence._DB_PATH = old_path
            if old_path:
                os.environ["VRM_SQLITE_PATH"] = old_path
            else:
                os.environ.pop("VRM_SQLITE_PATH", None)
            os.unlink(db_path)


# ── P5.3: Config hot-reload hooks ────────────────────────────────────────

class TestConfigReloadHooks:
    """Test register/unregister/notify for config reload hooks."""

    def test_register_hook(self):
        from core.config import register_reload_hook, unregister_reload_hook, _reload_hooks
        calls = []
        def my_hook(old, new):
            calls.append((old, new))

        register_reload_hook(my_hook)
        try:
            assert my_hook in _reload_hooks
        finally:
            unregister_reload_hook(my_hook)
        assert my_hook not in _reload_hooks

    def test_hook_called_on_reload(self):
        from core.config import register_reload_hook, unregister_reload_hook, reload_config
        calls = []
        def my_hook(old, new):
            calls.append(("called", type(old).__name__, type(new).__name__))

        register_reload_hook(my_hook)
        try:
            reload_config()
            assert len(calls) == 1
            assert calls[0][0] == "called"
            assert calls[0][1] == "dict"
            assert calls[0][2] == "dict"
        finally:
            unregister_reload_hook(my_hook)

    def test_hook_receives_old_and_new(self):
        from core.config import register_reload_hook, unregister_reload_hook, reload_config
        configs = []
        def my_hook(old, new):
            configs.append({"old": old.copy(), "new": new.copy()})

        register_reload_hook(my_hook)
        try:
            reload_config()
            reload_config()
            assert len(configs) == 2
            # Second call should have old == first call's new
            assert configs[1]["old"] == configs[0]["new"]
        finally:
            unregister_reload_hook(my_hook)

    def test_failing_hook_does_not_block(self):
        from core.config import register_reload_hook, unregister_reload_hook, reload_config
        calls = []
        def bad_hook(old, new):
            raise RuntimeError("boom")

        def good_hook(old, new):
            calls.append("ok")

        register_reload_hook(bad_hook)
        register_reload_hook(good_hook)
        try:
            reload_config()  # should not raise
            assert calls == ["ok"]
        finally:
            unregister_reload_hook(bad_hook)
            unregister_reload_hook(good_hook)

    def test_duplicate_register_ignored(self):
        from core.config import register_reload_hook, unregister_reload_hook, _reload_hooks
        def my_hook(old, new): pass
        register_reload_hook(my_hook)
        register_reload_hook(my_hook)  # duplicate
        try:
            count = _reload_hooks.count(my_hook)
            assert count == 1
        finally:
            unregister_reload_hook(my_hook)

    def test_unregister_nonexistent_is_noop(self):
        from core.config import unregister_reload_hook
        def never_registered(old, new): pass
        unregister_reload_hook(never_registered)  # should not raise


# ── P5.4: Metrics lifecycle ──────────────────────────────────────────────

class TestMetricsLifecycle:
    """Test reset_metrics() clears gauge state."""

    def test_reset_metrics_import(self):
        from core.metrics import reset_metrics
        assert callable(reset_metrics)

    def test_reset_metrics_clears_simple_gauge(self):
        from core.metrics import TASKS_RUNNING, reset_metrics
        TASKS_RUNNING.set(42)
        reset_metrics()
        # After reset, gauge should be 0
        samples = TASKS_RUNNING.collect()[0].samples
        val = samples[0].value if samples else 0
        assert val == 0.0

    def test_reset_metrics_clears_labeled_gauge(self):
        from core.metrics import GPU_MEMORY_USED, reset_metrics
        GPU_MEMORY_USED.labels(gpu="test_gpu_99").set(12345)
        reset_metrics()
        # After reset, the label set should be gone
        samples = GPU_MEMORY_USED.collect()[0].samples
        test_samples = [s for s in samples if "test_gpu_99" in str(s.labels)]
        assert len(test_samples) == 0

    def test_reset_metrics_called_in_shutdown(self):
        """Verify reset_metrics is called in pipeline shutdown."""
        import inspect
        from core.inference_pipeline import InferencePipeline
        source = inspect.getsource(InferencePipeline.shutdown)
        assert "reset_metrics" in source

    def test_reset_metrics_in_exports(self):
        from core import metrics
        assert "reset_metrics" in metrics.__all__

    def test_labeled_gauges_tracked(self):
        from core.metrics import _LABELED_GAUGES, GPU_MEMORY_USED
        assert GPU_MEMORY_USED in _LABELED_GAUGES

    def test_simple_gauges_tracked(self):
        from core.metrics import _SIMPLE_GAUGES, TASKS_RUNNING
        assert TASKS_RUNNING in _SIMPLE_GAUGES
