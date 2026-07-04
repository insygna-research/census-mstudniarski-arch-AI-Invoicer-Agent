"""Rasteryzacja PDF -> PNG per strona (pypdfium2 + Pillow).

Dla providerow bez natywnego wejscia PDF (Ollama przyjmuje tylko obrazy).
Glosno odmawia przy zbyt wielu stronach — cichy trunc gubilby pozycje faktury.
"""

from __future__ import annotations

import io

_DEFAULT_DPI = 150
_DEFAULT_MAX_PAGES = 8


def pdf_to_png_pages(
    content: bytes, *, dpi: int = _DEFAULT_DPI, max_pages: int = _DEFAULT_MAX_PAGES
) -> list[bytes]:
    """Renderuje kazda strone PDF do PNG. ValueError: zly PDF albo > max_pages stron."""
    import pypdfium2 as pdfium

    try:
        document = pdfium.PdfDocument(content)
    except Exception as exc:
        raise ValueError(f"Nie mozna otworzyc PDF: {exc}") from exc
    n_pages = len(document)
    if n_pages > max_pages:
        raise ValueError(
            f"PDF ma {n_pages} stron (> max_pages={max_pages}) — podnies limit swiadomie."
        )
    scale = dpi / 72  # PDF ma 72 punkty na cal
    pngs: list[bytes] = []
    for page in document:
        image = page.render(scale=scale).to_pil()
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        pngs.append(buf.getvalue())
    return pngs
