import pytest

from src.evaluation import (
    EvalQuestion,
    evaluate_strategy,
    is_relevant,
    recall_at_k,
    reciprocal_rank,
    results_to_markdown,
)

# --- is_relevant ---


def test_is_relevant_substring_match():
    assert is_relevant("The sky is blue and bright today.", "sky is blue")


def test_is_relevant_case_insensitive():
    assert is_relevant("The SKY is BLUE.", "sky is blue")


def test_is_relevant_whitespace_tolerant():
    assert is_relevant("The   sky\nis  blue.", "sky is blue")


def test_is_relevant_no_match():
    assert not is_relevant("Dogs are loyal.", "sky is blue")


def test_is_relevant_heals_pdf_linebreak_hyphens():
    # Real-world pypdf artifact: word split across a line break.
    chunk = (
        "enforcing proportionality can cause a grow-\ning, yet sublinear, welfare loss"
    )
    assert is_relevant(chunk, "growing, yet sublinear, welfare loss")


def test_is_relevant_preserves_compound_hyphens():
    # "cross-encoder" should still match as-is — no whitespace after the hyphen.
    assert is_relevant("we use a cross-encoder reranker", "cross-encoder reranker")


# --- recall_at_k ---


def test_recall_at_k_hit_in_top_k():
    chunks = ["irrelevant", "sky is blue", "more text"]
    assert recall_at_k(chunks, "sky is blue", k=3) == 1.0


def test_recall_at_k_miss_outside_top_k():
    chunks = ["a", "b", "c", "sky is blue"]
    assert recall_at_k(chunks, "sky is blue", k=3) == 0.0


def test_recall_at_k_no_match_anywhere():
    chunks = ["a", "b", "c"]
    assert recall_at_k(chunks, "sky is blue", k=10) == 0.0


# --- reciprocal_rank ---


def test_reciprocal_rank_first_position():
    chunks = ["sky is blue", "other", "more"]
    assert reciprocal_rank(chunks, "sky is blue") == 1.0


def test_reciprocal_rank_third_position():
    chunks = ["a", "b", "sky is blue"]
    assert reciprocal_rank(chunks, "sky is blue") == pytest.approx(1 / 3)


def test_reciprocal_rank_no_match():
    assert reciprocal_rank(["a", "b"], "sky is blue") == 0.0


# --- evaluate_strategy ---


def test_evaluate_strategy_aggregates():
    questions = [
        EvalQuestion(question="What is X?", reference="X is one"),
        EvalQuestion(question="What is Y?", reference="Y is two"),
    ]
    # Synthetic retriever: returns chunks based on the question
    canned = {
        "What is X?": ["irrelevant", "X is one and important", "noise"],
        "What is Y?": ["noise", "noise", "Y is two here", "noise"],
    }

    def retrieve(question: str, n: int) -> list[str]:
        return canned[question][:n]

    chunks_all = ["irrelevant", "X is one and important", "noise", "Y is two here"]
    result = evaluate_strategy("test", chunks_all, questions, retrieve)

    assert result.strategy == "test"
    assert result.n_chunks == 4
    # Both questions hit within top-3
    assert result.recall_at_3 == 1.0
    # X hit at position 2 → 1/2 = 0.5; Y hit at position 3 → 1/3 → average ~0.417
    assert result.mrr == pytest.approx((0.5 + 1 / 3) / 2)


def test_results_to_markdown_renders_table():
    from src.evaluation import EvalResult

    result = EvalResult(
        strategy="token-based",
        n_chunks=42,
        recall_at_3=0.85,
        recall_at_5=0.90,
        recall_at_10=0.95,
        mrr=0.72,
    )
    md = results_to_markdown([result])
    assert "| token-based |" in md
    assert "| 42 |" in md
    assert "0.85" in md
