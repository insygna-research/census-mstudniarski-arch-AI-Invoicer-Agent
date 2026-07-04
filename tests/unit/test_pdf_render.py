import io

import pypdfium2 as pdfium
import pytest

from invoicer.adapters.pdf_render import pdf_to_png_pages


def _pdf_bytes(pages: int = 1) -> bytes:
    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        doc.new_page(200, 100)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def test_renders_each_page_to_png():
    pngs = pdf_to_png_pages(_pdf_bytes(pages=2))
    assert len(pngs) == 2
    for png in pngs:
        assert png.startswith(b"\x89PNG")  # magic bajty PNG


def test_single_page_pdf():
    assert len(pdf_to_png_pages(_pdf_bytes(pages=1))) == 1


def test_too_many_pages_raises_loudly():
    # Cichy trunc gubilby pozycje faktury — lepiej glosno odmowic i dac podniesc limit.
    with pytest.raises(ValueError, match="stron"):
        pdf_to_png_pages(_pdf_bytes(pages=3), max_pages=2)


def test_invalid_pdf_raises_value_error():
    with pytest.raises(ValueError, match="PDF"):
        pdf_to_png_pages(b"to nie jest pdf")
