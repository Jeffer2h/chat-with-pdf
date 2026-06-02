"""Compare retriever configurations on a fixed chunking strategy.

Holds chunking constant (token-based, which won the chunking comparison)
and varies two axes:

- Embedding model: all-MiniLM-L6-v2 (current default, 80 MB) vs.
  BAAI/bge-base-en-v1.5 (440 MB, retrieval-tuned).
- Reranker: off vs. BAAI/bge-reranker-base on top-20 candidates.

This isolates "which piece of the retrieval stack is the bottleneck?".

Usage:
    python -m scripts.run_retriever_experiments
    python -m scripts.run_retriever_experiments --eval data/other.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# This experiment specifically compares local sentence-transformers models.
# Requires `uv sync --extra local` (PyTorch is needed).
from src.embeddings import LocalEmbedder  # noqa: E402
from src.evaluation import (  # noqa: E402
    evaluate_strategy,
    load_eval_questions,
    results_to_markdown,
)
from src.chunking import split_into_chunks  # noqa: E402
from src.pdf_loader import extract_text  # noqa: E402
from src.reranker import LocalReranker, Reranker  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402

CONFIGS = [
    ("MiniLM-L6 (baseline)", "sentence-transformers/all-MiniLM-L6-v2", False),
    ("MiniLM-L6 + reranker", "sentence-transformers/all-MiniLM-L6-v2", True),
    ("BGE-base-en-v1.5", "BAAI/bge-base-en-v1.5", False),
    ("BGE-base-en-v1.5 + reranker", "BAAI/bge-base-en-v1.5", True),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=Path("data/eval_questions.json"))
    parser.add_argument(
        "--out", type=Path, default=Path("data/retriever_experiments.md")
    )
    return parser.parse_args()


def _build_retrieve_fn(store: VectorStore, reranker: Reranker | None):
    """Return a retrieve_fn(question, n_results) suitable for evaluate_strategy."""
    if reranker is None:
        return store.query

    def retrieve(question: str, n_results: int) -> list[str]:
        # Two-stage: cheap top-20 from bi-encoder, then rerank to top-n.
        candidates = store.query(question, n_results=20)
        return reranker.rerank(question, candidates, top_k=n_results)

    return retrieve


def main() -> None:
    args = _parse_args()

    questions = load_eval_questions(args.eval)
    eval_payload = json.loads(args.eval.read_text(encoding="utf-8"))
    pdf_path = Path(eval_payload["pdf"])
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"Loaded {len(questions)} eval questions from {args.eval}")
    print(f"Reading PDF: {pdf_path}")
    with open(pdf_path, "rb") as f:
        text = extract_text(f)

    # Chunking is fixed for this experiment
    chunks = split_into_chunks(text)
    print(f"Token-based chunks: {len(chunks)}\n")

    # Load the reranker once and reuse across reranker-enabled configs
    print("Loading reranker (once)...")
    reranker = LocalReranker()  # BAAI/bge-reranker-base

    results = []
    embedder_cache: dict[str, LocalEmbedder] = {}

    for label, model_name, use_reranker in CONFIGS:
        print(f"\n=== {label} ===")
        if model_name not in embedder_cache:
            print(f"  Loading embedding model: {model_name}")
            embedder_cache[model_name] = LocalEmbedder(model_name)
        embedder = embedder_cache[model_name]

        start = time.perf_counter()
        store = VectorStore(embedder=embedder)
        store.add_documents(chunks)

        retrieve_fn = _build_retrieve_fn(store, reranker if use_reranker else None)

        result = evaluate_strategy(
            strategy_name=label,
            chunks=chunks,
            questions=questions,
            retrieve_fn=retrieve_fn,
        )
        elapsed = time.perf_counter() - start
        print(
            f"  recall@3={result.recall_at_3:.2f}  "
            f"recall@5={result.recall_at_5:.2f}  "
            f"recall@10={result.recall_at_10:.2f}  "
            f"MRR={result.mrr:.3f}  ({elapsed:.1f}s)"
        )
        results.append(result)

    table = results_to_markdown(results)
    print("\n=== Final comparison ===\n")
    print(table)

    args.out.write_text(table, encoding="utf-8")
    print(f"\nTable written to {args.out}")


if __name__ == "__main__":
    main()
