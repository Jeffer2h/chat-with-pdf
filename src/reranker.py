"""Optional two-stage retrieval reranker.

Bi-encoders (the embedder) are fast but coarse: they encode the query and
chunks independently. Cross-encoders (or rerank APIs like Voyage's) score
each (query, chunk) pair jointly — slower but more accurate.

The standard pattern: retrieve a wide top-N with the embedder, then rerank
that subset and keep the final top-k. Most of the precision benefit, none
of the cost of scoring every chunk in the corpus.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

from src.observability import log_llm_call


@runtime_checkable
class Reranker(Protocol):
    """Reorders chunks by relevance to a query."""

    def rerank(self, query: str, chunks: list[str], top_k: int = 5) -> list[str]: ...

    @property
    def name(self) -> str:
        """Human-readable model identifier (e.g. "rerank-2.5"). For logging/UI."""
        ...


# --------------------------------------------------------------------------- #
# Local backend (sentence-transformers CrossEncoder)
# --------------------------------------------------------------------------- #


class LocalReranker:
    """Wraps a sentence-transformers CrossEncoder. Free but pulls torch in."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        from sentence_transformers import CrossEncoder

        self._model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[str], top_k: int = 5) -> list[str]:
        if not chunks:
            return []
        pairs = [(query, c) for c in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [chunks[i] for i in ranked[:top_k]]

    @property
    def name(self) -> str:
        return self._model_name


# --------------------------------------------------------------------------- #
# Voyage backend (rerank-2.5 API)
# --------------------------------------------------------------------------- #


class VoyageReranker:
    """Reranking via Voyage AI API. Slim install, pay-per-token.

    Each call is wrapped in `log_llm_call` so cost and latency land in
    `logs/llm_calls.db` alongside Claude and embedder calls.
    """

    def __init__(self, model_name: str = "rerank-2.5") -> None:
        import voyageai

        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY is not set but RERANKER_BACKEND=voyage. "
                "Add it to your .env or switch to RERANKER_BACKEND=off."
            )
        # voyageai stubs don't re-export Client at the package level; runtime is fine.
        self._client = voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]
        self._model_name = model_name

    def rerank(self, query: str, chunks: list[str], top_k: int = 5) -> list[str]:
        if not chunks:
            return []

        # Voyage's API caps top_k at the document count; clamp to be safe.
        effective_top_k = min(top_k, len(chunks))

        with log_llm_call(self._model_name) as record:
            response = self._client.rerank(
                query=query,
                documents=chunks,
                model=self._model_name,
                top_k=effective_top_k,
            )
            record["input_tokens"] = response.total_tokens

        # Results are already sorted by descending relevance_score.
        return [r.document for r in response.results]

    @property
    def name(self) -> str:
        return self._model_name


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def get_reranker() -> Reranker | None:
    """Build the reranker declared by RERANKER_BACKEND.

    Returns:
        A Reranker, or None when RERANKER_BACKEND=off. Callers handle the
        None case by skipping the two-stage retrieval.

    Raises:
        ValueError: if the env var has an unrecognized value.
    """
    backend = os.environ.get("RERANKER_BACKEND", "voyage").lower()
    if backend in ("off", "false", "0", ""):
        return None
    if backend == "voyage":
        return VoyageReranker()
    if backend == "local":
        return LocalReranker()
    raise ValueError(
        f"RERANKER_BACKEND='{backend}' is not valid. "
        "Expected 'off', 'voyage' or 'local'."
    )
