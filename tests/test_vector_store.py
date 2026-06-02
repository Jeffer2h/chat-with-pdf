from src.vector_store import VectorStore


def test_add_and_query(fake_embedder):
    vs = VectorStore(embedder=fake_embedder)
    chunks = [
        "The sky is blue and full of clouds.",
        "Python is a popular programming language.",
        "Dogs are friendly and loyal animals.",
    ]
    vs.add_documents(chunks)
    results = vs.query("The sky is blue and full of clouds.", n_results=1)
    # The FakeEmbedder is deterministic: querying with the exact chunk text
    # returns that chunk as the nearest neighbor.
    assert len(results) == 1
    assert results[0] == chunks[0]


def test_query_returns_n_results(fake_embedder):
    vs = VectorStore(embedder=fake_embedder)
    chunks = [f"Document chunk number {i}." for i in range(10)]
    vs.add_documents(chunks)
    results = vs.query("document chunk", n_results=3)
    assert len(results) == 3


def test_query_clamps_n_results_to_available_docs(fake_embedder):
    """Asking for more results than exist returns all of them, not an error.

    ChromaDB raises if n_results exceeds the collection size, so VectorStore
    clamps it. Guards the path where the reranker requests top-20 from a PDF
    that only produced a handful of chunks.
    """
    vs = VectorStore(embedder=fake_embedder)
    chunks = [f"Document chunk number {i}." for i in range(3)]
    vs.add_documents(chunks)
    results = vs.query("document chunk", n_results=20)
    assert len(results) == 3


def test_empty_query_returns_empty(fake_embedder):
    vs = VectorStore(embedder=fake_embedder)
    assert vs.query("anything", n_results=3) == []


def test_embedder_property_exposed(fake_embedder):
    vs = VectorStore(embedder=fake_embedder)
    assert vs.embedder is fake_embedder
