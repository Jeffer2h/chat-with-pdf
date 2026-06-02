import re

import numpy as np
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.embeddings import Embedder

_ENCODING = tiktoken.get_encoding("cl100k_base")


def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks measured in tokens (not characters).

    Args:
        text: The full document text.
        chunk_size: Target size of each chunk in tokens.
        overlap: Number of tokens shared between consecutive chunks.

    Returns:
        List of non-empty text chunks.

    Raises:
        ValueError: If overlap is not strictly smaller than chunk_size.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    tokens = _ENCODING.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = _ENCODING.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def split_into_chunks_recursive(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text using LangChain's RecursiveCharacterTextSplitter, measured in tokens.

    Splits hierarchically: paragraphs → sentences → words → characters. This
    keeps chunks aligned with natural text boundaries instead of cutting mid-word.

    Args:
        text: The full document text.
        chunk_size: Target chunk size in tokens.
        overlap: Token overlap between consecutive chunks.

    Returns:
        List of non-empty text chunks.
    """
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]


def split_into_chunks_semantic(
    text: str,
    embedder: Embedder,
    max_chunk_tokens: int = 500,
    breakpoint_threshold: float = 0.5,
) -> list[str]:
    """Split text into chunks based on semantic similarity between sentences.

    Cuts at sentence boundaries where the topic shifts (similarity drops below
    breakpoint_threshold) or when the chunk would exceed max_chunk_tokens.

    Note:
        Sentences whose own token count exceeds max_chunk_tokens are kept whole
        (they form a single oversized chunk). The splitter never breaks within
        a sentence — that is a deliberate trade-off favoring semantic coherence
        over hard size limits.

    Args:
        text: The full document text.
        embedder: Embedder used to vectorize each sentence (local or Voyage).
        max_chunk_tokens: Hard limit on chunk size in tokens.
        breakpoint_threshold: Cosine similarity below this value triggers a new chunk.

    Returns:
        List of non-empty semantic chunks.
    """
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Split into sentences on common punctuation
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    embeddings = np.array(embedder.embed(sentences, input_type="document"))
    sentence_token_counts = [len(_ENCODING.encode(s)) for s in sentences]

    chunks: list[str] = []
    current: list[str] = [sentences[0]]
    current_tokens = sentence_token_counts[0]
    current_indices: list[int] = [0]

    for i in range(1, len(sentences)):
        sentence_tokens = sentence_token_counts[i]

        # Compare the new sentence against the mean embedding of the current chunk,
        # not just the previous sentence. This catches gradual topic drift correctly.
        # Recomputing the mean each iteration is O(n) per sentence (O(n^2) per
        # chunk); an incremental running mean would be O(1). Left simple on
        # purpose — semantic is the documented underperformer and PDFs are small.
        chunk_embedding = np.mean(embeddings[current_indices], axis=0)
        sim = float(
            np.dot(chunk_embedding, embeddings[i])
            / (np.linalg.norm(chunk_embedding) * np.linalg.norm(embeddings[i]) + 1e-10)
        )

        topic_shift = sim < breakpoint_threshold
        would_exceed = current_tokens + sentence_tokens > max_chunk_tokens

        if topic_shift or would_exceed:
            chunks.append(" ".join(current))
            current = [sentences[i]]
            current_tokens = sentence_tokens
            current_indices = [i]
        else:
            current.append(sentences[i])
            current_tokens += sentence_tokens
            current_indices.append(i)

    if current:
        chunks.append(" ".join(current))

    return chunks
