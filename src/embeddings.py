"""Embedder abstraction with two interchangeable backends.

The rest of the codebase (VectorStore, semantic chunking) depends on the
`Embedder` Protocol, not on a concrete class. Backend selection happens
once at startup via `get_embedder()`, driven by the EMBEDDING_BACKEND env
var. This keeps the hot path agnostic to whether vectors come from a
local PyTorch model or a remote API.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from src.observability import log_llm_call

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns texts into fixed-size vectors.

    The Protocol is `runtime_checkable` so tests can `isinstance(x, Embedder)`
    without forcing concrete classes to inherit.
    """

    def embed(
        self, texts: list[str], *, input_type: str = "document"
    ) -> list[list[float]]:
        """Embed a batch of texts. `input_type` is "document" or "query"."""
        ...

    @property
    def dimension(self) -> int:
        """Output vector size. Used for sanity checks against ChromaDB collections."""
        ...

    @property
    def name(self) -> str:
        """Human-readable identifier (e.g. "voyage-3-lite"). For logging only."""
        ...


# --------------------------------------------------------------------------- #
# Local backend (sentence-transformers + PyTorch)
# --------------------------------------------------------------------------- #


class LocalEmbedder:
    """Wraps a sentence-transformers model. Free but heavy (~800MB of torch).

    Default is BGE-base-en-v1.5: it outperformed MiniLM on 2/3 eval corpora
    in our retriever experiments and pairs naturally with the BGE reranker.
    Pass model_name="all-MiniLM-L6-v2" if you want the lighter/faster option.
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5") -> None:
        # Imported lazily so the slim production image (without sentence-transformers
        # installed) doesn't crash at import time when only VoyageEmbedder is used.
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    def embed(
        self, texts: list[str], *, input_type: str = "document"
    ) -> list[list[float]]:
        # Local bi-encoders don't distinguish query vs document — input_type is ignored.
        if not texts:
            return []
        return cast("list[list[float]]", self._model.encode(texts).tolist())

    @property
    def dimension(self) -> int:
        # SentenceTransformer.get_sentence_embedding_dimension() is typed as
        # int | None; for all supported models it returns a real int.
        return cast(int, self._dimension)

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def raw_model(self) -> "SentenceTransformer":
        """Escape hatch for code that still needs the underlying SentenceTransformer
        (e.g. local-only semantic chunking experiments). Avoid in new code."""
        return cast("SentenceTransformer", self._model)


# --------------------------------------------------------------------------- #
# Voyage backend (API)
# --------------------------------------------------------------------------- #

# Voyage list prices, USD per 1M tokens. Output tokens are always 0 for embeddings.
_VOYAGE_DIMENSIONS = {
    "voyage-3-lite": 512,
    "voyage-3": 1024,
    "voyage-3-large": 1024,
}


class VoyageEmbedder:
    """Embeddings via Voyage AI API. Slim install, pay-per-token.

    Each call is wrapped in `log_llm_call` so token counts, latency, and cost
    land in `logs/llm_calls.db` alongside Claude calls — same observability
    convention across the portfolio.
    """

    def __init__(self, model_name: str = "voyage-3-lite") -> None:
        import voyageai  # lazy import: keeps tests that mock the backend fast

        if model_name not in _VOYAGE_DIMENSIONS:
            raise ValueError(
                f"Unknown Voyage model '{model_name}'. "
                f"Known: {sorted(_VOYAGE_DIMENSIONS)}"
            )

        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "VOYAGE_API_KEY is not set but EMBEDDING_BACKEND=voyage. "
                "Add it to your .env or switch to EMBEDDING_BACKEND=local."
            )

        # voyageai stubs don't re-export Client at the package level even though
        # the runtime does; the import path itself is stable.
        self._client = voyageai.Client(api_key=api_key)  # type: ignore[attr-defined]
        self._model_name = model_name
        self._dimension = _VOYAGE_DIMENSIONS[model_name]

    # Voyage caps each call at 1000 texts. Anything longer is batched internally
    # so the caller (VectorStore, semantic chunking) never has to care.
    _BATCH_SIZE = 1000

    def embed(
        self, texts: list[str], *, input_type: str = "document"
    ) -> list[list[float]]:
        """Embed a batch of texts via the Voyage API, batching to respect API limits."""
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total_tokens = 0

        # One observability record per logical embed() call, even if it spans
        # multiple API requests. Tokens and latency aggregate naturally.
        with log_llm_call(self._model_name) as record:
            for start in range(0, len(texts), self._BATCH_SIZE):
                batch = texts[start : start + self._BATCH_SIZE]
                response = self._client.embed(
                    texts=batch,
                    model=self._model_name,
                    input_type=input_type,
                )
                all_embeddings.extend(cast("list[list[float]]", response.embeddings))
                total_tokens += response.total_tokens

            record["input_tokens"] = total_tokens

        return all_embeddings

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return self._model_name


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def get_embedder() -> Embedder:
    """Build the embedder declared by the EMBEDDING_BACKEND env var.

    Returns:
        A ready-to-use Embedder. Caller is expected to cache the instance
        (e.g. via @st.cache_resource) — both backends are expensive to create.

    Raises:
        ValueError: if EMBEDDING_BACKEND is set to something other than
            "voyage" or "local".
    """
    backend = os.environ.get("EMBEDDING_BACKEND", "voyage").lower()
    if backend == "voyage":
        return VoyageEmbedder()
    if backend == "local":
        return LocalEmbedder()
    raise ValueError(
        f"EMBEDDING_BACKEND='{backend}' is not valid. Expected 'voyage' or 'local'."
    )
