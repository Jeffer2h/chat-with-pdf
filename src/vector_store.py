import uuid

import chromadb

from src.embeddings import Embedder


class VectorStore:
    """In-process vector store backed by ChromaDB.

    Embeddings come from any `Embedder` (local sentence-transformers or
    Voyage API). Each instance is ephemeral: data lives only for the
    lifetime of the object — that's why every Streamlit upload constructs
    a fresh VectorStore.
    """

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder
        self._client = chromadb.EphemeralClient()
        # Unique collection name so concurrent Streamlit sessions don't collide
        self._collection = self._client.create_collection(
            name=f"docs_{uuid.uuid4().hex}"
        )
        self._doc_count = 0

    @property
    def embedder(self) -> Embedder:
        """The underlying embedder, exposed for reuse (e.g. semantic chunking)."""
        return self._embedder

    def add_documents(self, chunks: list[str]) -> None:
        """Embed and store a list of text chunks."""
        if not chunks:
            return
        embeddings = self._embedder.embed(chunks, input_type="document")
        ids = [str(self._doc_count + i) for i in range(len(chunks))]
        # Chroma's stubs expect Sequence[float] per row but list[list[float]]
        # is accepted at runtime — the stubs are overly narrow.
        self._collection.add(
            documents=chunks,
            embeddings=embeddings,  # type: ignore[arg-type]
            ids=ids,
        )
        self._doc_count += len(chunks)

    def query(self, question: str, n_results: int = 3) -> list[str]:
        """Find the most relevant chunks for a question."""
        n_results = min(n_results, self._collection.count())
        if n_results == 0:
            return []
        # input_type="query" is asymmetric-encoding hint for Voyage; ignored locally.
        embedding = self._embedder.embed([question], input_type="query")
        results = self._collection.query(
            query_embeddings=embedding,  # type: ignore[arg-type]
            n_results=n_results,
        )
        # With the default include=["documents"], Chroma always returns the
        # documents key — but the stubs declare it Optional, so narrow it.
        # Explicit guard (not assert) so it survives `python -O`.
        documents = results["documents"]
        if documents is None:
            return []
        return documents[0]
