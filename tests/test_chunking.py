import pytest

from src.chunking import (
    split_into_chunks,
    split_into_chunks_recursive,
    split_into_chunks_semantic,
)

# --- Token-based chunking ---


def test_split_into_chunks_produces_multiple_chunks():
    text = "This is a test sentence. " * 60
    chunks = split_into_chunks(text, chunk_size=150, overlap=15)
    assert len(chunks) > 1


def test_split_into_chunks_respects_token_limit():
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    text = "Hello world. " * 200
    chunks = split_into_chunks(text, chunk_size=150, overlap=15)
    for chunk in chunks:
        assert len(enc.encode(chunk)) <= 150


def test_split_into_chunks_short_text():
    text = "Hello world"
    chunks = split_into_chunks(text, chunk_size=150, overlap=15)
    assert len(chunks) == 1
    assert chunks[0] == "Hello world"


def test_split_into_chunks_rejects_overlap_ge_chunk_size():
    # Without the guard, step = chunk_size - overlap is <= 0 and the loop never
    # advances: an infinite loop. Fail fast at the boundary instead.
    with pytest.raises(ValueError, match="overlap"):
        split_into_chunks("any text", chunk_size=100, overlap=100)
    with pytest.raises(ValueError, match="overlap"):
        split_into_chunks("any text", chunk_size=100, overlap=200)


def test_split_into_chunks_empty():
    chunks = split_into_chunks("", chunk_size=150, overlap=15)
    assert chunks == []


def test_chunks_share_overlap_content_across_all_pairs():
    # Property test: for every consecutive pair, the textual tail of chunk[k]
    # must appear at the start of chunk[k+1]. We compare on decoded text — not
    # re-encoded token IDs — because the implementation applies `.strip()` to
    # each chunk, which can shift the BPE tokenization at the boundary (a
    # leading-space token like " word" becomes "word"). The content overlap is
    # what the algorithm actually guarantees; this catches any regression that
    # silently zeros out the overlap step.
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    text = "word " * 500
    chunks = split_into_chunks(text, chunk_size=50, overlap=10)

    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        tail_text = enc.decode(enc.encode(prev)[-10:]).strip()
        assert tail_text and nxt.startswith(tail_text)


def test_first_chunk_tail_starts_second_chunk():
    # Companion to the property test: same invariant, but on readable text so a
    # failure surfaces as a legible string diff instead of a list of token IDs.
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    text = "alpha beta gamma delta epsilon " * 30
    chunks = split_into_chunks(text, chunk_size=20, overlap=5)

    assert len(chunks) >= 2
    tail = enc.decode(enc.encode(chunks[0])[-5:]).strip()
    assert chunks[1].startswith(tail)


# --- Recursive chunking ---


def test_split_into_chunks_recursive_basic():
    text = "First paragraph.\n\nSecond paragraph here.\n\n" * 30
    chunks = split_into_chunks_recursive(text, chunk_size=50, overlap=5)
    assert len(chunks) > 1
    assert all(isinstance(c, str) and c for c in chunks)


def test_split_into_chunks_recursive_empty():
    assert split_into_chunks_recursive("") == []


def test_split_into_chunks_recursive_short_text():
    chunks = split_into_chunks_recursive("Hello world", chunk_size=150, overlap=15)
    assert chunks == ["Hello world"]


# --- Semantic chunking (uses the Embedder protocol via FakeEmbedder) ---


def test_split_into_chunks_semantic_returns_list(fake_embedder):
    text = (
        "The sky is blue. Clouds float above. The sun shines brightly. "
        "Python is a programming language. It is used for data science. "
        "Dogs are loyal animals. They make great companions."
    )
    chunks = split_into_chunks_semantic(text, embedder=fake_embedder)
    assert isinstance(chunks, list)
    assert len(chunks) >= 1


def test_split_into_chunks_semantic_empty(fake_embedder):
    chunks = split_into_chunks_semantic("", embedder=fake_embedder)
    assert chunks == []


def test_split_into_chunks_semantic_respects_max_tokens(fake_embedder):
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    text = "This is a sentence about topic A. " * 30
    chunks = split_into_chunks_semantic(
        text, embedder=fake_embedder, max_chunk_tokens=50
    )
    for chunk in chunks:
        # Allow a small margin for single sentences that exceed the limit on their own
        assert len(enc.encode(chunk)) <= 100
