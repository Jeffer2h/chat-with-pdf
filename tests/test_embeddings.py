"""Tests for the Embedder abstraction.

Real backends require either PyTorch (LocalEmbedder) or network + API key
(VoyageEmbedder). We test the factory wiring and mock the Voyage path.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.embeddings import VoyageEmbedder, get_embedder
from tests.conftest import read_llm_call_rows


def test_get_embedder_invalid_raises(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "nonsense")
    with pytest.raises(ValueError):
        get_embedder()


def test_voyage_embedder_requires_api_key(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        VoyageEmbedder()


def test_voyage_embedder_rejects_unknown_model(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    with patch("voyageai.Client", return_value=MagicMock()):
        with pytest.raises(ValueError, match="Unknown Voyage model"):
            VoyageEmbedder(model_name="nonexistent-model")


def test_voyage_embedder_empty_input(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    with patch("voyageai.Client", return_value=MagicMock()):
        embedder = VoyageEmbedder()
        assert embedder.embed([]) == []


def test_voyage_embedder_calls_api(monkeypatch, observability_db):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

    fake_response = MagicMock()
    fake_response.embeddings = [[0.1, 0.2], [0.3, 0.4]]
    fake_response.total_tokens = 10

    fake_client = MagicMock()
    fake_client.embed.return_value = fake_response

    with patch("voyageai.Client", return_value=fake_client):
        embedder = VoyageEmbedder()
        result = embedder.embed(["hello", "world"], input_type="document")

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    fake_client.embed.assert_called_once()
    kwargs = fake_client.embed.call_args.kwargs
    assert kwargs["texts"] == ["hello", "world"]
    assert kwargs["model"] == "voyage-3-lite"
    assert kwargs["input_type"] == "document"

    # The whole point of wrapping the Voyage call in `log_llm_call` is that a
    # row lands in the observability DB. Assert it explicitly so a future
    # refactor that drops the wrapper cannot silently pass CI.
    rows = read_llm_call_rows(observability_db)
    assert len(rows) == 1
    assert rows[0]["model"] == "voyage-3-lite"
    assert rows[0]["input_tokens"] == 10
    assert rows[0]["output_tokens"] == 0
    assert rows[0]["success"] == 1


def test_voyage_embedder_batches_over_1000(monkeypatch, observability_db):
    """Inputs larger than the API cap should be split into multiple calls
    and reassembled transparently. Tokens are summed across batches."""
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")

    fake_client = MagicMock()

    # Each call returns embeddings of the same length as its input batch.
    def fake_embed(texts, model, input_type):
        resp = MagicMock()
        resp.embeddings = [[0.0] * 4 for _ in texts]
        resp.total_tokens = len(texts)  # 1 token per text, easy to assert
        return resp

    fake_client.embed.side_effect = fake_embed

    with patch("voyageai.Client", return_value=fake_client):
        embedder = VoyageEmbedder()
        result = embedder.embed(["x"] * 2500)

    # 2500 / 1000 = 3 batches (1000 + 1000 + 500)
    assert fake_client.embed.call_count == 3
    assert len(result) == 2500

    # Three API sub-batches should aggregate into ONE logical row with the
    # tokens summed (1 token per text in the fake -> 2500 total).
    rows = read_llm_call_rows(observability_db)
    assert len(rows) == 1
    assert rows[0]["input_tokens"] == 2500


def test_voyage_embedder_dimension(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    with patch("voyageai.Client", return_value=MagicMock()):
        embedder = VoyageEmbedder("voyage-3-lite")
        assert embedder.dimension == 512
        assert embedder.name == "voyage-3-lite"
