import sqlite3
from pathlib import Path

import pytest

from src.observability import estimate_cost_usd, log_llm_call, read_session_stats


def _rows(db_path: Path) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT model, input_tokens, output_tokens, success, error_msg "
            "FROM llm_calls"
        ).fetchall()
    finally:
        conn.close()


def test_estimate_cost_known_model():
    # 1M input tokens at $3/M + 1M output tokens at $15/M = $18
    cost = estimate_cost_usd("claude-sonnet-4-6", 1_000_000, 1_000_000)
    assert cost == pytest.approx(18.0)


def test_estimate_cost_unknown_model_returns_zero():
    assert estimate_cost_usd("nonexistent-model", 1000, 1000) == 0.0


def test_log_llm_call_success(tmp_path):
    db = tmp_path / "logs" / "calls.db"
    with log_llm_call("claude-sonnet-4-6", db_path=db) as record:
        record["input_tokens"] = 100
        record["output_tokens"] = 50

    rows = _rows(db)
    assert len(rows) == 1
    model, in_tok, out_tok, success, error = rows[0]
    assert model == "claude-sonnet-4-6"
    assert in_tok == 100
    assert out_tok == 50
    assert success == 1
    assert error is None


def test_read_session_stats_aggregates_rows(tmp_path):
    db = tmp_path / "logs" / "calls.db"
    with log_llm_call("claude-sonnet-4-6", db_path=db) as r:
        r["input_tokens"] = 100
        r["output_tokens"] = 50
    with log_llm_call("voyage-3-lite", db_path=db) as r:
        r["input_tokens"] = 200

    stats = read_session_stats(db_path=db)
    assert stats["total_calls"] == 2
    assert stats["total_input_tokens"] == 300
    assert stats["total_output_tokens"] == 50
    # Claude: 100 * 3 + 50 * 15 = 1050 per 1M -> 0.00105
    # Voyage: 200 * 0.02 per 1M             -> 0.000004
    assert stats["total_cost_usd"] == pytest.approx(0.001054)


def test_read_session_stats_empty_db(tmp_path):
    stats = read_session_stats(db_path=tmp_path / "nonexistent.db")
    assert stats == {
        "total_calls": 0,
        "total_cost_usd": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "avg_latency_ms": 0.0,
    }


def test_log_llm_call_failure_records_and_reraises(tmp_path):
    db = tmp_path / "logs" / "calls.db"

    with pytest.raises(ValueError):
        with log_llm_call("claude-sonnet-4-6", db_path=db):
            raise ValueError("boom")

    rows = _rows(db)
    assert len(rows) == 1
    _, _, _, success, error = rows[0]
    assert success == 0
    assert "ValueError" in error and "boom" in error
