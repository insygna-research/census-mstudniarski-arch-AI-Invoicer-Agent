from pathlib import Path

from invoicer.models import Invoice, TaxTreatment
from invoicer.rag.eval import build_invoice_from_case, load_cases

_CASES = Path(__file__).resolve().parents[2] / "data" / "evals" / "legal_cases.jsonl"


def test_dataset_loads_and_has_expected_fields():
    cases = load_cases(_CASES)
    assert len(cases) >= 30
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))  # id unikalne
    for case in cases:
        assert case["expected_treatment"] in {t.value for t in TaxTreatment}
        assert isinstance(case["expected_article_refs"], list)
        assert case["expected_article_refs"]  # kazdy przypadek ma oczekiwana podstawe prawna
        assert case["seller_country"]
        assert case["seller_country"] != "PL"  # eval dotyczy wylacznie faktur zagranicznych


def test_build_invoice_from_case_produces_valid_invoice():
    case = load_cases(_CASES)[0]
    inv = build_invoice_from_case(case)
    assert isinstance(inv, Invoice)
    assert inv.seller.country == case["seller_country"]


def test_dataset_covers_key_treatments():
    cases = load_cases(_CASES)
    by_treatment: dict[str, int] = {}
    for c in cases:
        by_treatment[c["expected_treatment"]] = by_treatment.get(c["expected_treatment"], 0) + 1
    # rdzen kontrastu podatkowego: uslugi vs towary UE vs towary spoza UE
    assert by_treatment.get("import_uslug", 0) >= 15
    assert by_treatment.get("wnt", 0) >= 5
    assert by_treatment.get("import_towarow", 0) >= 5
