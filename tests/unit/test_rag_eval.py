from invoicer.models import Citation
from invoicer.rag.eval import faithfulness_rate, mean, recall_at_k, reciprocal_rank
from invoicer.rag.models import RetrievedChunk


def test_recall_at_k_full_and_partial():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5
    assert recall_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0  # 'a' poza top-2


def test_recall_at_k_empty_expected_is_one():
    assert recall_at_k(["a"], set(), k=1) == 1.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "y"], {"a"}) == 0.5  # 1/2
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_mean():
    assert mean([1.0, 0.0, 0.5]) == 0.5
    assert mean([]) == 0.0


_CHUNK = RetrievedChunk(
    source_id="s",
    article_ref="a",
    title="t",
    url="u",
    text="Miejscem swiadczenia uslug jest siedziba uslugobiorcy.",
)


def test_faithfulness_rate_counts_supported_citations():
    supported = Citation(source_id="s", article_ref="a", quoted_span="Miejscem swiadczenia uslug")
    fabricated = Citation(source_id="s", article_ref="a", quoted_span="zdanie spoza zrodla")
    rate = faithfulness_rate([supported, fabricated], [_CHUNK])
    assert rate == 0.5


def test_faithfulness_rate_no_citations_is_zero():
    assert faithfulness_rate([], [_CHUNK]) == 0.0


def test_render_report_includes_all_metrics_and_ablation_delta():
    from invoicer.rag.eval import render_report

    report = render_report(
        generated="2026-07-02",
        retrieval={"k": 5, "n": 30, "recall_at_k": 0.97, "mrr": 0.81},
        with_rag={"n": 30, "accuracy": 0.93, "faithfulness": 0.9},
        without_rag={"n": 30, "accuracy": 0.7},
        setup="pgvector · voyage-3-large · rerank-2.5 · gemini-2.5-flash (reasoner)",
    )
    assert "Recall@5: **0.97**" in report
    assert "MRR: **0.81**" in report
    assert "with RAG**: **0.93**" in report
    assert "without RAG**: **0.70**" in report
    assert "**+0.23**" in report  # delta ablacji
    assert "Faithfulness" in report and "**0.90**" in report
    assert "2026-07-02" in report
    assert "Cases: **30**" in report


def test_checkpoint_load_missing_file_is_empty(tmp_path):
    from invoicer.rag.eval import load_checkpoint

    assert load_checkpoint(tmp_path / "brak.jsonl") == {}


def test_checkpoint_append_and_load_roundtrip(tmp_path):
    # Dzienna kwota darmowego tieru (20 zapytan/model) wymusza dokanczanie evalu
    # przez wiele dni — checkpoint per przypadek robi postep trwalym.
    from invoicer.rag.eval import append_checkpoint, load_checkpoint

    path = tmp_path / "cp.jsonl"
    append_checkpoint(path, "with-rag:uk-saas", {"hit": 1.0, "faithfulness": 0.5})
    append_checkpoint(path, "no-rag:uk-saas", {"hit": 0.0})
    loaded = load_checkpoint(path)
    assert loaded["with-rag:uk-saas"] == {"hit": 1.0, "faithfulness": 0.5}
    assert loaded["no-rag:uk-saas"] == {"hit": 0.0}


def test_checkpoint_last_write_wins(tmp_path):
    from invoicer.rag.eval import append_checkpoint, load_checkpoint

    path = tmp_path / "cp.jsonl"
    append_checkpoint(path, "retrieval", {"recall_at_k": 0.5})
    append_checkpoint(path, "retrieval", {"recall_at_k": 0.93})
    assert load_checkpoint(path)["retrieval"] == {"recall_at_k": 0.93}
