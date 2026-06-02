"""Compare chunking strategies on the eval set.

Runs each strategy (token-based, recursive, semantic) over the eval PDF,
indexes the resulting chunks, retrieves top-10 for each eval question, and
reports recall@3 / recall@5 / recall@10 / MRR as a markdown table.

Usage (inside container or local venv):
    python -m scripts.run_evaluation
    python -m scripts.run_evaluation --pdf path/to/other.pdf
    python -m scripts.run_evaluation --eval data/other_questions.json --out results.md
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

# Allow running as `python scripts/run_evaluation.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import Embedder, get_embedder  # noqa: E402
from src.evaluation import (  # noqa: E402
    EvalQuestion,
    evaluate_strategy,
    is_relevant,
    load_eval_questions,
    reciprocal_rank,
    results_to_markdown,
)
from src.chunking import (  # noqa: E402
    split_into_chunks,
    split_into_chunks_recursive,
    split_into_chunks_semantic,
)
from src.pdf_loader import extract_text  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402


def _build_strategies(text: str, embedder: Embedder) -> dict[str, list[str]]:
    """Run all chunking strategies on the same text."""
    return {
        "token-based": split_into_chunks(text),
        "recursive": split_into_chunks_recursive(text),
        "semantic": split_into_chunks_semantic(text, embedder=embedder),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf",
        type=Path,
        default=None,
        help="Path to the PDF. Defaults to the one referenced in the eval JSON.",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("data/eval_questions.json"),
        help="Eval questions JSON. Default: data/eval_questions.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write the markdown table. Defaults to stdout only.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-question hit/miss diagnostics for each strategy.",
    )
    return parser.parse_args()


def _print_verbose_breakdown(
    strategy_name: str,
    questions: list[EvalQuestion],
    retrieve_fn: Callable[[str, int], list[str]],
) -> None:
    """Print per-question retrieval diagnostics: rank of first hit and a preview."""
    print(f"\n--- Per-question breakdown: {strategy_name} ---")
    for i, q in enumerate(questions, 1):
        retrieved = retrieve_fn(q.question, 10)
        # Find the rank of the first relevant chunk, if any
        rank = next(
            (j for j, c in enumerate(retrieved, 1) if is_relevant(c, q.reference)),
            None,
        )
        rr = reciprocal_rank(retrieved, q.reference)
        status = f"rank={rank} (RR={rr:.2f})" if rank else "MISS"
        q_preview = q.question[:70] + ("..." if len(q.question) > 70 else "")
        ref_preview = q.reference[:60] + ("..." if len(q.reference) > 60 else "")
        print(f"  [{i:>2}] {status:<18} Q: {q_preview}")
        print(f"        ref: {ref_preview!r}")
        if rank is None:
            # Show top-1 chunk so we can see what the retriever thought was best
            top = retrieved[0][:200].replace("\n", " ") if retrieved else "<empty>"
            print(f"        top-1 chunk: {top!r}")


def main() -> None:
    args = _parse_args()

    questions = load_eval_questions(args.eval)
    print(f"Loaded {len(questions)} eval questions from {args.eval}")

    eval_payload = json.loads(args.eval.read_text(encoding="utf-8"))
    pdf_path = args.pdf or Path(eval_payload["pdf"])
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    print(f"Reading PDF: {pdf_path}")

    with open(pdf_path, "rb") as f:
        text = extract_text(f)

    print("Loading embedder (one-time)...")
    embedder = get_embedder()
    print(f"  Using {embedder.name} (dim={embedder.dimension})")

    print("Chunking with all strategies...")
    strategies = _build_strategies(text, embedder)
    for name, chunks in strategies.items():
        print(f"  - {name}: {len(chunks)} chunks")

    results = []
    for name, chunks in strategies.items():
        print(f"\nEvaluating {name}...")
        start = time.perf_counter()

        store = VectorStore(embedder=embedder)
        store.add_documents(chunks)

        result = evaluate_strategy(
            strategy_name=name,
            chunks=chunks,
            questions=questions,
            retrieve_fn=store.query,
        )
        elapsed = time.perf_counter() - start
        print(
            f"  recall@3={result.recall_at_3:.2f}  "
            f"recall@5={result.recall_at_5:.2f}  "
            f"recall@10={result.recall_at_10:.2f}  "
            f"MRR={result.mrr:.3f}  ({elapsed:.1f}s)"
        )
        results.append(result)

        if args.verbose:
            _print_verbose_breakdown(name, questions, store.query)

    table = results_to_markdown(results)
    print("\n=== Final results ===\n")
    print(table)

    if args.out is not None:
        args.out.write_text(table, encoding="utf-8")
        print(f"\nTable written to {args.out}")


if __name__ == "__main__":
    main()
