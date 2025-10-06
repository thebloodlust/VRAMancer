"""Instrumentation OpenTelemetry (lot A) – optionnelle.

Ce module encapsule l'initialisation du tracing pour éviter de forcer
des dépendances runtime si l'utilisateur ne souhaite pas activer
OpenTelemetry. L'activation passe par la variable d'env
 VRM_TRACING=1

On exporte un helper `start_tracing()` idempotent. Les spans
utilisés ailleurs doivent toujours être créés via `get_tracer()`.

Si opentelemetry n'est pas installé, le module tombe en no-op.
"""
from __future__ import annotations
import os
from contextlib import contextmanager
import json

_started = False

try:  # Import lazy pour ne pas casser l'environnement minimal
    if os.environ.get("VRM_TRACING") == "1":
        from opentelemetry import trace  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace.export import (  # type: ignore
            BatchSpanProcessor, ConsoleSpanExporter,
        )
        # OTLP optionnel
        _OTLP_ENDPOINT = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        if _OTLP_ENDPOINT:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore
        else:
            OTLPSpanExporter = None  # type: ignore
    else:  # No tracing requested
        trace = None  # type: ignore
        OTLPSpanExporter = None  # type: ignore
except Exception:  # pragma: no cover - dépendances absentes
    trace = None  # type: ignore
    OTLPSpanExporter = None  # type: ignore


def start_tracing():  # pragma: no cover - simple initialisation
    global _started
    if _started or not trace:
        return
    # Resource attributes (service.name etc.)
    attrs = {"service.name": "vramancer"}
    extra = os.environ.get("VRM_TRACING_ATTRS")
    if extra:
        try:
            attrs.update(json.loads(extra))
        except Exception:
            pass
    provider = TracerProvider(resource=Resource.create(attrs))
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    # OTLP export si configuré
    if OTLPSpanExporter is not None:
        try:
            otlp = OTLPSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(otlp))
        except Exception:
            pass
    trace.set_tracer_provider(provider)
    _started = True


def get_tracer(name: str = "vramancer"):
    if trace:
        return trace.get_tracer(name)
    return _NoopTracer()


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, name: str):  # noqa: D401
        yield _NoopSpan()


class _NoopSpan:  # pragma: no cover - aucune logique
    def set_attribute(self, *_, **__):
        pass
    def add_event(self, *_ , **__):
        pass

__all__ = ["start_tracing", "get_tracer"]
