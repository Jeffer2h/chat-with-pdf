"""Reranker tests.

Real backends (LocalReranker / VoyageReranker) require either PyTorch or
network + API key. To keep unit tests fast and dependency-free we test
the factory and a mocked Voyage path. Integration with the real model is
covered by the eval script.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.reranker import VoyageReranker, get_reranker
from tests.conftest import read_llm_call_rows


def test_get_reranker_off_returns_none(monkeypatch):
    monkeypatch.setenv("RERANKER_BACKEND", "off")
    assert get_reranker() is None


@pytest.mark.parametrize("value", ["false", "0", "", "OFF", "False"])
def test_get_reranker_off_spellings_return_none(monkeypatch, value):
    """Any falsy spelling disables the reranker; matching is case-insensitive."""
    monkeypatch.setenv("RERANKER_BACKEND", value)
    assert get_reranker() is None


def test_get_reranker_invalid_raises(monkeypatch):
    monkeypatch.setenv("RERANKER_BACKEND", "nonsense")
    with pytest.raises(ValueError):
        get_reranker()


def test_voyage_reranker_requires_api_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        VoyageReranker()


def test_voyage_reranker_rerank_uses_api(monkeypatch, observability_db):
    """VoyageReranker.rerank should call the API and return documents in score order."""
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

    fake_response = MagicMock()
    fake_response.total_tokens = 42
    fake_response.results = [
        MagicMock(document="best match", relevance_score=0.9),
        MagicMock(document="ok match", relevance_score=0.5),
    ]

    fake_client = MagicMock()
    fake_client.rerank.return_value = fake_response

    with patch("voyageai.Client", return_value=fake_client):
        reranker = VoyageReranker()
        result = reranker.rerank("query", ["best match", "ok match", "worst"], top_k=2)

    assert result == ["best match", "ok match"]
    fake_client.rerank.assert_called_once()

    # Same contract as the embedder: every Voyage call must land in the DB.
    rows = read_llm_call_rows(observability_db)
    assert len(rows) == 1
    assert rows[0]["model"] == "rerank-2.5"
    assert rows[0]["input_tokens"] == 42
    assert rows[0]["output_tokens"] == 0
    assert rows[0]["success"] == 1


def test_voyage_reranker_clamps_top_k_to_chunk_count(monkeypatch):
    """If the caller passes top_k larger than len(chunks), it should be clamped
    before reaching the API (Voyage rejects top_k > len(documents))."""
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

    fake_response = MagicMock()
    fake_response.total_tokens = 5
    fake_response.results = [
        MagicMock(document="a", relevance_score=0.9),
        MagicMock(document="b", relevance_score=0.4),
    ]
    fake_client = MagicMock()
    fake_client.rerank.return_value = fake_response

    with patch("voyageai.Client", return_value=fake_client):
        reranker = VoyageReranker()
        reranker.rerank("query", ["a", "b"], top_k=10)

    kwargs = fake_client.rerank.call_args.kwargs
    assert kwargs["top_k"] == 2


def test_voyage_reranker_empty_input(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    with patch("voyageai.Client", return_value=MagicMock()):
        reranker = VoyageReranker()
        assert reranker.rerank("query", [], top_k=5) == []
