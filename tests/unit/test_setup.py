from __future__ import annotations

from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from invoicer.app import preflight_env
from invoicer.setup import SETUP_FIELDS, SetupGate

_FULL_ENV = {
    "GOOGLE_API_KEY": "AIza-test",
    "GMAIL_SENDER_FILTER": "owner@example.com",
    "TWILIO_ACCOUNT_SID": "ACxxxxxxxx",
    "TWILIO_AUTH_TOKEN": "tok",
    "TWILIO_WHATSAPP_FROM": "whatsapp:+1",
    "APPROVER_WHATSAPP_TO": "whatsapp:+2",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name, *_ in SETUP_FIELDS:
        monkeypatch.delenv(name, raising=False)
    for name in ("GMAIL_TOKEN_B64", "INVOICER_SINK", "DATABASE_URL"):
        monkeypatch.delenv(name, raising=False)


class _FakeBuilder:
    """Stands in for create_app: a tiny app whose lifespan records that the scheduler would have started."""

    def __init__(self, fail: Exception | None = None) -> None:
        self.fail = fail
        self.built = 0
        self.started = 0
        self.stopped = 0

    def __call__(self) -> FastAPI:
        if self.fail:
            raise self.fail
        self.built += 1

        @asynccontextmanager
        async def lifespan(_):
            self.started += 1
            yield
            self.stopped += 1

        app = FastAPI(lifespan=lifespan)

        @app.get("/health")
        def health() -> dict:
            return {"status": "ok"}

        @app.get("/status")
        def status() -> dict:
            return {"pipeline": "running"}

        return app


def test_starts_in_setup_mode_without_secrets_instead_of_crashing():
    builder = _FakeBuilder()
    with TestClient(SetupGate(builder, preflight_env)) as client:
        assert client.get("/health").json()["status"] == "setup"
        r = client.get("/", follow_redirects=False)
        assert r.status_code == 302 and r.headers["location"] == "/setup"
        assert "GOOGLE_API_KEY" in client.get("/setup").text
        r = client.get("/status")
        assert r.status_code == 503 and r.json()["setup"] == "/setup"
    assert builder.built == 0


def test_saving_the_form_starts_the_pipeline_in_process():
    builder = _FakeBuilder()
    with TestClient(SetupGate(builder, preflight_env)) as client:
        r = client.post("/setup", data=_FULL_ENV)
        assert r.status_code == 200 and "running" in r.text
        assert builder.built == 1 and builder.started == 1
        assert client.get("/status").json() == {"pipeline": "running"}
        assert client.get("/health").json()["status"] == "ok"
    assert builder.stopped == 1


def test_an_incomplete_form_is_refused_and_nothing_is_kept(monkeypatch):
    builder = _FakeBuilder()
    with TestClient(SetupGate(builder, preflight_env)) as client:
        partial = {k: v for k, v in _FULL_ENV.items() if k != "TWILIO_AUTH_TOKEN"}
        r = client.post("/setup", data=partial)
        assert r.status_code == 400 and "TWILIO_AUTH_TOKEN" in r.text
    assert builder.built == 0
    import os

    assert "GOOGLE_API_KEY" not in os.environ


def test_a_build_failure_is_shown_and_the_environment_restored():
    builder = _FakeBuilder(fail=RuntimeError("Twilio rejected the account"))
    with TestClient(SetupGate(builder, preflight_env)) as client:
        r = client.post("/setup", data=_FULL_ENV)
        assert r.status_code == 400 and "Twilio rejected the account" in r.text
        assert client.get("/health").json()["status"] == "setup"
    import os

    assert "GOOGLE_API_KEY" not in os.environ


def test_with_full_environment_it_is_the_full_app_from_the_start(monkeypatch):
    for k, v in _FULL_ENV.items():
        monkeypatch.setenv(k, v)
    builder = _FakeBuilder()
    with TestClient(SetupGate(builder, preflight_env)) as client:
        assert builder.started == 1
        assert client.get("/health").json()["status"] == "ok"
