from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from invoicer.extraction import (
    InvoiceExtraction,
    LineItemExtraction,
    PartyExtraction,
    extraction_to_invoice,
)
from invoicer.models import Invoice


def _extraction() -> InvoiceExtraction:
    return InvoiceExtraction(
        seller=PartyExtraction(name="ACME", nip="5260001246", country="PL"),
        buyer=PartyExtraction(name="Klient", country="PL"),
        number="FV/2026/06/01",
        issue_date="2026-06-01",
        currency="PLN",
        lines=[
            LineItemExtraction(
                description="Usluga",
                quantity="1",
                unit_net="1000.00",
                vat_rate="0.23",
                net="1000.00",
                vat="230.00",
                gross="1230.00",
            )
        ],
        total_net="1000.00",
        total_vat="230.00",
        total_gross="1230.00",
        confidence=0.9,
    )


def test_mapper_produces_domain_invoice_with_decimals():
    inv = extraction_to_invoice(_extraction())
    assert isinstance(inv, Invoice)
    assert inv.number == "FV/2026/06/01"
    assert inv.issue_date == date(2026, 6, 1)
    assert inv.total_gross == Decimal("1230.00")
    assert inv.lines[0].vat == Decimal("230.00")
    assert inv.seller.nip == "5260001246"
    assert inv.extraction_confidence == 0.9


def test_mapper_handles_optional_dates():
    ex = _extraction()
    ex.sale_date = "2026-06-02"
    ex.due_date = None
    inv = extraction_to_invoice(ex)
    assert inv.sale_date == date(2026, 6, 2)
    assert inv.due_date is None


def test_party_country_is_required():
    # Bez defaultu: default="PL" pozwalal leniwym modelom pomijac pole, przez co
    # zagraniczny sprzedawca dostawal kraj PL i faktura szla sciezka krajowa.
    with pytest.raises(ValidationError):
        PartyExtraction(name="Nordwind Solutions GmbH")


def test_party_country_full_names_normalize_to_iso2():
    # Lokalny model uparcie zwraca pelne nazwy ('Germany') mimo opisow pol — pelna
    # nazwa nie sparuje z EU_COUNTRIES i unijna faktura poszlaby sciezka poza-UE.
    # Angielskie nazwy sa jednoznaczne -> normalizujemy przez pycountry.
    assert PartyExtraction(name="X", country="Germany").country == "DE"
    assert PartyExtraction(name="X", country="Poland").country == "PL"
    assert PartyExtraction(name="X", country="United Kingdom").country == "GB"


def test_party_country_unknown_name_raises():
    # Nieznana nazwa: glosny fail > cicha zla klasyfikacja.
    with pytest.raises(ValidationError):
        PartyExtraction(name="X", country="Xlandia")


def test_party_country_normalizes_case():
    assert PartyExtraction(name="X", country="de").country == "DE"
    assert PartyExtraction(name="X", country=" pl ").country == "PL"


def test_confidence_is_bounded_0_1():
    with pytest.raises(ValidationError):
        InvoiceExtraction(
            seller=PartyExtraction(name="A", country="PL"),
            buyer=PartyExtraction(name="B", country="PL"),
            number="X",
            issue_date="2026-01-01",
            lines=[],
            total_net="0",
            total_vat="0",
            total_gross="0",
            confidence=1.5,
        )


def test_malformed_amount_raises_with_field_context():
    ex = _extraction()
    ex.total_net = "N/A"
    with pytest.raises(ValueError, match="total_net"):
        extraction_to_invoice(ex)


def test_malformed_date_raises_with_field_context():
    ex = _extraction()
    ex.issue_date = "01/06/2026"
    with pytest.raises(ValueError, match="issue_date"):
        extraction_to_invoice(ex)


def test_vat_rate_accepts_percent_form():
    # Mniejsze modele (lokalny Qwen) potrafia zwrocic '0%'/'23%' zamiast ulamka —
    # forma procentowa jest semantycznie jednoznaczna, wiec ja normalizujemy.
    from decimal import Decimal

    ex = _extraction()
    ex.lines[0].vat_rate = "0%"
    assert extraction_to_invoice(ex).lines[0].vat_rate == Decimal("0")

    ex = _extraction()
    ex.lines[0].vat_rate = "23%"
    assert extraction_to_invoice(ex).lines[0].vat_rate == Decimal("0.23")

    ex = _extraction()
    ex.lines[0].vat_rate = " 8 % "
    assert extraction_to_invoice(ex).lines[0].vat_rate == Decimal("0.08")


def test_vat_rate_garbage_still_raises():
    ex = _extraction()
    ex.lines[0].vat_rate = "zw."
    with pytest.raises(ValueError, match="vat_rate"):
        extraction_to_invoice(ex)
