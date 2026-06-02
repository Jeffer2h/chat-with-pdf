"""Shared test fixtures.

We use a deterministic FakeEmbedder so unit tests don't pull in PyTorch
or hit any external API. Integration tests for real backends live
separately and require the corresponding extras.
"""

import hashlib
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def observability_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Send every log_llm_call write to a temp SQLite file per test.

    The real logs/ directory is owned by root (created inside the docker
    container in past runs); local pytest runs can't write to it. Per-test
    tmp_path also keeps assertions on the DB hermetic.

    Returns:
        The temp DB path so tests can query rows that were logged.
    """
    db_path = tmp_path / "llm_calls.db"
    monkeypatch.setattr("src.observability.DEFAULT_DB_PATH", db_path)
    return db_path


def read_llm_call_rows(db_path: Path) -> list[dict]:
    """Read every row written to the observability DB. Empty list if DB missing."""
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM llm_calls")]
    finally:
        conn.close()


class FakeEmbedder:
    """Cheap, deterministic embedder for unit tests.

    Produces a fixed-dim vector by hashing the input text. Two identical
    strings get identical vectors; otherwise vectors are pseudo-random but
    stable across runs. Good enough for testing plumbing (shapes, flow,
    counts) — not for testing retrieval quality.
    """

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def embed(self, texts, *, input_type="document"):
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Cycle the digest bytes to fill the requested dimension
        return [(digest[i % len(digest)] / 255.0) for i in range(self._dim)]

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "fake-embedder"


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()
