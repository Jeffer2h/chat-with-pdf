"""SQLite-backed observability helper for LLM calls.

Each call is logged with model, tokens, latency, estimated cost and
success/error info. Mirrors the portfolio-wide observability convention.
"""

import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

_logger = logging.getLogger(__name__)

PROJECT_NAME = "chat-with-pdf"
DEFAULT_DB_PATH = Path(__file__).parent.parent / "logs" / "llm_calls.db"

# List prices, USD per 1M tokens: (input, output).
# Voyage embeddings/rerankers only charge input tokens, so output price is 0.
# Cached/batch discounts ignored — this is a portfolio demo, not finance-grade.
_PRICING_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-sonnet-4-7": (3.0, 15.0),
    "claude-opus-4-7": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    # Voyage embeddings
    "voyage-3-lite": (0.02, 0.0),
    "voyage-3": (0.06, 0.0),
    "voyage-3-large": (0.18, 0.0),
    # Voyage rerankers
    "rerank-2.5-lite": (0.02, 0.0),
    "rerank-2.5": (0.05, 0.0),
}


def _ensure_table(db_path: Path) -> sqlite3.Connection:
    """Create the logs directory and llm_calls table if missing."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            project TEXT NOT NULL,
            model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            latency_ms REAL NOT NULL,
            cost_usd REAL NOT NULL,
            success INTEGER NOT NULL,
            error_msg TEXT
        )
        """
    )
    conn.commit()
    return conn


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the USD cost of a single call based on list pricing.

    Returns 0.0 for unknown models rather than raising — the log entry is
    still useful even if the price isn't in the table.
    """
    if model not in _PRICING_PER_M_TOKENS:
        return 0.0
    in_price, out_price = _PRICING_PER_M_TOKENS[model]
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


def read_session_stats(db_path: Path | None = None) -> dict[str, float | int]:
    """Return aggregate stats over every row in the observability DB.

    For a portfolio demo the database lives only for the lifetime of the
    container (it's gitignored and not persisted across rebuilds), so "all
    rows" effectively means "this session". A future production version
    would scope by a session id.

    Returns:
        Dict with total_calls, total_cost_usd, total_input_tokens,
        total_output_tokens, avg_latency_ms. All zeros if the DB is empty.
    """
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if not db_path.exists():
        return {
            "total_calls": 0,
            "total_cost_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "avg_latency_ms": 0.0,
        }
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_calls,
                COALESCE(SUM(cost_usd), 0) AS total_cost_usd,
                COALESCE(SUM(input_tokens), 0) AS total_input_tokens,
                COALESCE(SUM(output_tokens), 0) AS total_output_tokens,
                COALESCE(AVG(latency_ms), 0) AS avg_latency_ms
            FROM llm_calls
            """
        ).fetchone()
    finally:
        conn.close()
    return {
        "total_calls": int(row[0]),
        "total_cost_usd": float(row[1]),
        "total_input_tokens": int(row[2]),
        "total_output_tokens": int(row[3]),
        "avg_latency_ms": float(row[4]),
    }


@contextmanager
def log_llm_call(
    model: str,
    project: str = PROJECT_NAME,
    db_path: Path | None = None,
) -> Iterator[dict[str, int]]:
    """Context manager that times and logs an LLM call.

    Usage:
        with log_llm_call("claude-sonnet-4-6") as record:
            response = client.messages.create(...)
            record["input_tokens"] = response.usage.input_tokens
            record["output_tokens"] = response.usage.output_tokens

    The caller MUST set input_tokens and output_tokens on success.
    If the wrapped block raises, the error is recorded and re-raised.
    """
    # Resolve at call time so test monkeypatches of DEFAULT_DB_PATH take effect.
    if db_path is None:
        db_path = DEFAULT_DB_PATH

    record: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
    }
    success = True
    error_msg: str | None = None
    start = time.perf_counter()

    try:
        yield record
    except Exception as exc:
        success = False
        error_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        latency_ms = (time.perf_counter() - start) * 1000
        cost_usd = estimate_cost_usd(
            model, record["input_tokens"], record["output_tokens"]
        )
        # Wrap the DB write so a SQLite failure never replaces the original
        # exception. If the logger itself fails, we lose the log entry but the
        # caller's exception (e.g. an API error) is still propagated correctly.
        try:
            conn = _ensure_table(db_path)
            try:
                conn.execute(
                    """
                    INSERT INTO llm_calls (
                        timestamp, project, model,
                        input_tokens, output_tokens,
                        latency_ms, cost_usd, success, error_msg
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        project,
                        model,
                        record["input_tokens"],
                        record["output_tokens"],
                        latency_ms,
                        cost_usd,
                        1 if success else 0,
                        error_msg,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
        except Exception as log_exc:
            # Surface the failure during development without masking the
            # caller's original exception (which is already in flight if any).
            _logger.warning("Failed to write llm_calls row: %s", log_exc)
