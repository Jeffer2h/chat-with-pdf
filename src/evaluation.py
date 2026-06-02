"""Evaluation metrics for chunking strategies.

A retrieved chunk is "relevant" if it contains the reference passage that
holds the answer (case-insensitive substring). Two metrics are reported:

- recall@k: fraction of questions whose relevant passage appears in the
  top-k retrieved chunks. Aggregated across questions.
- MRR (mean reciprocal rank): 1/rank of the first relevant chunk per
  question, averaged across questions. Captures how *high* the right
  answer is ranked, not just whether it appeared.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class EvalQuestion:
    question: str
    reference: str  # exact passage from the document that contains the answer


def load_eval_questions(path: Path) -> list[EvalQuestion]:
    """Load eval questions from a JSON file.

    Expected schema:
        {"questions": [{"question": "...", "reference": "..."}]}
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalQuestion(**q) for q in data["questions"]]


def _normalize(text: str) -> str:
    """Lower-case, heal line-break hyphens, and collapse whitespace.

    PDF text extraction often preserves soft hyphens introduced by justified
    line wrapping (e.g. "represen-\ntation"). Without healing them the
    substring matcher would miss otherwise-relevant chunks, undercounting
    the true retrieval quality. We collapse `word_char + "-" + whitespace +
    word_char` into a single word; standalone or compound hyphens like
    "cross-encoder" are left intact.
    """
    text = text.lower()
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_relevant(chunk: str, reference: str) -> bool:
    """A chunk is relevant if it (mostly) contains the reference passage.

    Uses substring match on normalized text. Tolerates whitespace and case
    differences but not paraphrasing — that's by design: the reference is
    the *exact* answer text, so substring match is the strict-but-fair rule.
    """
    return _normalize(reference) in _normalize(chunk)


def recall_at_k(retrieved_chunks: list[str], reference: str, k: int) -> float:
    """1.0 if any of the top-k retrieved chunks contains the reference, else 0.0."""
    top_k = retrieved_chunks[:k]
    return 1.0 if any(is_relevant(c, reference) for c in top_k) else 0.0


def reciprocal_rank(retrieved_chunks: list[str], reference: str) -> float:
    """1/rank of the first relevant chunk (1-indexed), or 0.0 if none match."""
    for i, chunk in enumerate(retrieved_chunks, start=1):
        if is_relevant(chunk, reference):
            return 1.0 / i
    return 0.0


@dataclass
class EvalResult:
    strategy: str
    n_chunks: int
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    mrr: float


def evaluate_strategy(
    strategy_name: str,
    chunks: list[str],
    questions: list[EvalQuestion],
    retrieve_fn: Callable[[str, int], list[str]],
) -> EvalResult:
    """Run all eval questions through a retriever and aggregate metrics.

    Args:
        strategy_name: Label for the result row (e.g. "token-based").
        chunks: The chunks produced by this strategy (for reporting count).
        questions: Eval set.
        retrieve_fn: Callable(question, n_results) -> top chunks. Usually
            `VectorStore.query` bound to a store already populated with `chunks`.

    Returns:
        Aggregated EvalResult across all questions.
    """
    if not questions:
        return EvalResult(
            strategy=strategy_name,
            n_chunks=len(chunks),
            recall_at_3=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            mrr=0.0,
        )

    r3 = r5 = r10 = mrr_sum = 0.0
    for q in questions:
        retrieved = retrieve_fn(q.question, 10)
        r3 += recall_at_k(retrieved, q.reference, 3)
        r5 += recall_at_k(retrieved, q.reference, 5)
        r10 += recall_at_k(retrieved, q.reference, 10)
        mrr_sum += reciprocal_rank(retrieved, q.reference)

    n = len(questions)
    return EvalResult(
        strategy=strategy_name,
        n_chunks=len(chunks),
        recall_at_3=r3 / n,
        recall_at_5=r5 / n,
        recall_at_10=r10 / n,
        mrr=mrr_sum / n,
    )


def results_to_markdown(results: list[EvalResult]) -> str:
    """Render results as a markdown table suitable for the README."""
    header = "| Strategy | # Chunks | recall@3 | recall@5 | recall@10 | MRR |\n"
    sep = "|---|---|---|---|---|---|\n"
    rows = "".join(
        f"| {r.strategy} | {r.n_chunks} | {r.recall_at_3:.2f} | "
        f"{r.recall_at_5:.2f} | {r.recall_at_10:.2f} | {r.mrr:.3f} |\n"
        for r in results
    )
    return header + sep + rows
