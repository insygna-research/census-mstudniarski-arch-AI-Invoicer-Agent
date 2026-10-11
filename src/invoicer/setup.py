"""Tryb konfiguracji: start bez sekretow, uzupelnienie ich na /setup i przelaczenie na pelna aplikacje.

Uvicorn startuje przez ``_factory`` -> ``create_entry_app``. Gdy srodowisko jest kompletne,
to jest dokladnie ``create_app()`` — zachowanie bez zmian. Gdy brakuje sekretow rdzennych
(``preflight_env`` rzuca), proces NIE pada: wstaje lekka bramka z /health i formularzem /setup.
Po zapisaniu formularza wartosci trafiaja do ``os.environ``, pelna aplikacja jest budowana
w tym samym procesie, jej lifespan (scheduler zaciagu z Gmaila) zostaje uruchomiony, a bramka
odtad przekazuje kazde zadanie do niej — bez restartu kontenera.

Wartosci wpisane na /setup zyja tylko w pamieci procesu: restart pyta ponownie. Zeby przetrwaly,
ustaw je w srodowisku (sekrety platformy). Kto siega do tego portu, ten moze je ustawic —
na platformie Insygna port jest za bramka dostepu organizacji.
"""

from __future__ import annotations

import html
import logging
import os
from collections.abc import Awaitable, Callable, MutableMapping
from contextlib import AbstractAsyncContextManager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

log = logging.getLogger("invoicer.setup")

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

# (zmienna, etykieta, podpowiedz, czy tajna)
SETUP_FIELDS: tuple[tuple[str, str, str, bool], ...] = (
    ("GOOGLE_API_KEY", "Google (Gemini) API key", "Extraction, reasoning and invoice detection.", True),
    ("GMAIL_SENDER_FILTER", "Gmail sender filter", "Address whose unread PDF invoices are taken in, e.g. invoices@supplier.com.", False),
    ("TWILIO_ACCOUNT_SID", "Twilio account SID", "WhatsApp approval channel.", False),
    ("TWILIO_AUTH_TOKEN", "Twilio auth token", "", True),
    ("TWILIO_WHATSAPP_FROM", "Twilio WhatsApp sender", "e.g. whatsapp:+14155238886", False),
    ("APPROVER_WHATSAPP_TO", "Approver WhatsApp number", "Who approves each booking, e.g. whatsapp:+48600000000", False),
)
OPTIONAL_FIELDS: tuple[tuple[str, str, str, bool], ...] = (
    ("GMAIL_TOKEN_B64", "Gmail OAuth token (base64 of token.json)", "Optional here, but intake reads no mail until a token exists.", True),
)


def _page(missing: list[str], message: tuple[str, str] | None = None) -> str:
    def field(name: str, label: str, hint: str, secret: bool, required: bool) -> str:
        present = bool(os.environ.get(name))
        kind = "password" if secret else "text"
        note = " (set — leave empty to keep)" if present else ""
        req = "required" if required and not present else ""
        return (
            f'<label for="{name}">{html.escape(label)}{note}</label>'
            f'<input id="{name}" name="{name}" type="{kind}" autocomplete="off" {req}>'
            + (f"<small>{html.escape(hint)}</small>" if hint else "")
        )

    fields = "".join(field(*f, required=True) for f in SETUP_FIELDS)
    fields += "".join(field(*f, required=False) for f in OPTIONAL_FIELDS)
    status = (
        "Missing: " + ", ".join(missing) + ". The invoice pipeline starts once these are saved."
        if missing
        else "All required settings are present."
    )
    banner = f'<p class="{message[0]}">{html.escape(message[1])}</p>' if message else ""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Invoicer setup</title>
<style>
 body{{font-family:system-ui,sans-serif;background:#f8fafc;color:#0f172a;margin:0;padding:32px 16px}}
 main{{max-width:560px;margin:0 auto;background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:24px}}
 h1{{font-size:20px;margin:0 0 8px}} p{{font-size:14px;line-height:1.5}} .status{{color:#475569}}
 .ok{{color:#166534;background:#f0fdf4;border:1px solid #bbf7d0;padding:8px 12px;border-radius:8px}}
 .error{{color:#991b1b;background:#fef2f2;border:1px solid #fecaca;padding:8px 12px;border-radius:8px}}
 label{{display:block;font-size:13px;font-weight:600;margin:16px 0 4px}}
 input{{width:100%;box-sizing:border-box;padding:8px 12px;border:1px solid #cbd5e1;border-radius:8px;font-size:14px}}
 small{{color:#64748b;font-size:12px}}
 button{{margin-top:20px;background:#0369a1;color:#fff;border:0;border-radius:8px;padding:10px 16px;font-size:14px;cursor:pointer}}
</style></head><body><main>
<h1>AI Invoicer setup</h1>
<p class="status" data-testid="setup-status">{html.escape(status)}</p>
{banner}
<form method="post" action="/setup">{fields}<button type="submit">Save and start</button></form>
<p class="status"><small>Values are kept in memory until the process restarts. Set them as environment secrets to keep them.</small></p>
</main></body></html>"""


class SetupGate:
    """ASGI: pelna aplikacja gdy skonfigurowana, inaczej formularz /setup.

    ``build_app`` buduje pelna aplikacje z aktualnego ``os.environ`` (w produkcji ``create_app``);
    ``preflight`` rzuca ``RuntimeError`` gdy czegos brakuje (w produkcji ``preflight_env``).
    """

    def __init__(
        self,
        build_app: Callable[[], FastAPI],
        preflight: Callable[[MutableMapping[str, str]], None],
    ) -> None:
        self._build_app = build_app
        self._preflight = preflight
        self.app: FastAPI | None = None
        self._lifespan: AbstractAsyncContextManager[Any] | None = None
        self._setup = self._setup_app()

    def missing(self) -> list[str]:
        return [name for name, *_ in SETUP_FIELDS if not os.environ.get(name)]

    async def activate(self) -> None:
        """Buduje pelna aplikacje i uruchamia jej lifespan (scheduler). Bledy budowy wracaja do wolajacego."""
        app = self._build_app()
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        self.app, self._lifespan = app, lifespan
        log.info("setup: configuration complete, invoice pipeline started")

    async def shutdown(self) -> None:
        if self._lifespan is not None:
            await self._lifespan.__aexit__(None, None, None)
            self._lifespan = None

    def _setup_app(self) -> FastAPI:
        api = FastAPI()

        @api.get("/health")
        def health() -> dict:
            return {"status": "setup", "missing": self.missing()}

        @api.get("/")
        def root() -> RedirectResponse:
            return RedirectResponse("/setup", status_code=302)

        @api.get("/setup", response_class=HTMLResponse)
        def setup_form() -> str:
            return _page(self.missing())

        @api.post("/setup", response_class=HTMLResponse)
        async def setup_save(request: Request) -> HTMLResponse:
            form = await request.form()
            entered = {
                name: str(form.get(name, "")).strip()
                for name, *_ in SETUP_FIELDS + OPTIONAL_FIELDS
            }
            still_missing = [n for n, *_ in SETUP_FIELDS if not (entered[n] or os.environ.get(n))]
            if still_missing:
                return HTMLResponse(
                    _page(still_missing, ("error", "Fill in: " + ", ".join(still_missing) + ". Nothing was saved.")),
                    status_code=400,
                )
            previous = {n: os.environ.get(n) for n in entered}
            for name, value in entered.items():
                if value:
                    os.environ[name] = value
            try:
                self._preflight(os.environ)
                await self.activate()
            except Exception as exc:  # przywroc srodowisko — polowiczna konfiguracja nie moze zostac
                for name, value in previous.items():
                    if value is None:
                        os.environ.pop(name, None)
                    else:
                        os.environ[name] = value
                log.warning("setup: could not start the pipeline: %s", exc)
                return HTMLResponse(
                    _page(self.missing(), ("error", f"Could not start with these settings: {exc}. Nothing was saved.")),
                    status_code=400,
                )
            return HTMLResponse(_page([], ("ok", "Saved. The invoice pipeline is running.")))

        @api.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
        def not_configured(path: str) -> JSONResponse:
            return JSONResponse(
                {"error": "not configured", "missing": self.missing(), "setup": "/setup"},
                status_code=503,
            )

        return api

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self._run_lifespan(receive, send)
            return
        target = self.app if self.app is not None else self._setup
        await target(scope, receive, send)

    async def _run_lifespan(self, receive: Receive, send: Send) -> None:
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                try:
                    if not self.missing():
                        await self.activate()
                except Exception as exc:
                    await send({"type": "lifespan.startup.failed", "message": str(exc)})
                    return
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await self.shutdown()
                await send({"type": "lifespan.shutdown.complete"})
                return
