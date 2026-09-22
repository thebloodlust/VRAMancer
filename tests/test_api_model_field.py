#!/usr/bin/env python3
"""Le champ `model` d'un client OpenAI ne doit JAMAIS décharger le modèle servi.

Bug trouvé le 2026-09-22 en branchant un vrai client sur `vramancer serve` :
une requête avec `{"model": "local"}` (nom arbitraire, ce qu'envoient beaucoup
de clients) faisait partir le serveur en résolution HuggingFace. Or
`PipelineRegistry.load()` **arrête le pipeline courant AVANT** de charger le
nouveau : le chargement échouait, et le serveur se retrouvait sans aucun modèle.
Une seule requête d'un client mal configuré mettait donc le service à terre.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def app_client(monkeypatch):
    monkeypatch.setenv("VRM_TEST_MODE", "1")
    monkeypatch.setenv("VRM_MINIMAL_TEST", "1")
    monkeypatch.setenv("VRM_DISABLE_RATE_LIMIT", "1")
    monkeypatch.setenv("VRM_TEST_ALL_OPEN", "1")
    from core.production_api import create_app
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class _FakePipeline:
    """Pipeline minimal qui se déclare chargé et sait générer."""

    model_name = "mon-modele.gguf"
    tokenizer = None

    def __init__(self):
        self.loads = []

    def is_loaded(self):
        return True

    def generate(self, prompt, **kw):
        return "ok"

    def generate_stream(self, prompt, **kw):
        yield "ok"

    def shutdown(self):
        pass


@pytest.fixture
def loaded_registry(monkeypatch):
    """Registre avec un modèle chargé, et un load() qui explose s'il est appelé."""
    import core.production_api as api_mod
    fake = _FakePipeline()
    registry = api_mod._registry
    monkeypatch.setattr(registry, "_pipeline", fake, raising=False)
    monkeypatch.setattr(registry, "_model_name", "mon-modele.gguf", raising=False)

    def _explode(model_name, **kw):
        raise AssertionError(
            f"load({model_name!r}) appelé alors qu'un modèle est déjà servi — "
            "c'est exactement le bug : le pipeline courant serait arrêté."
        )

    monkeypatch.setattr(registry, "load", _explode, raising=False)
    return registry


def test_arbitrary_model_name_does_not_trigger_reload(app_client, loaded_registry):
    """« local », « gpt-4 »… : on sert le modèle chargé, comme llama.cpp/LM Studio."""
    for name in ("local", "gpt-4", "qwen3.6-coder", "default"):
        r = app_client.post("/v1/chat/completions",
                            json={"model": name,
                                  "messages": [{"role": "user", "content": "salut"}],
                                  "max_tokens": 8})
        assert r.status_code != 500, f"model={name!r} a fait tomber la requête: {r.data[:200]}"


def test_completions_endpoint_too(app_client, loaded_registry):
    r = app_client.post("/v1/completions",
                        json={"model": "local", "prompt": "salut", "max_tokens": 8})
    assert r.status_code != 500, r.data[:200]


def test_real_model_reference_is_still_honoured(app_client, monkeypatch):
    """Un vrai nom de modèle (avec '/' ou .gguf) doit toujours déclencher un chargement."""
    import core.production_api as api_mod
    registry = api_mod._registry
    calls = []
    monkeypatch.setattr(registry, "_pipeline", _FakePipeline(), raising=False)
    monkeypatch.setattr(registry, "_model_name", "mon-modele.gguf", raising=False)
    monkeypatch.setattr(registry, "load",
                        lambda model_name, **kw: calls.append(model_name), raising=False)
    app_client.post("/v1/chat/completions",
                    json={"model": "org/autre-modele",
                          "messages": [{"role": "user", "content": "salut"}],
                          "max_tokens": 8})
    assert calls == ["org/autre-modele"], f"chargement non déclenché: {calls}"
