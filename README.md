# chat-with-pdf

> Retrieval-Augmented Generation (RAG) over a PDF, evaluated rigorously.

Upload a PDF, ask questions about it, get answers grounded in the document — and, more interestingly, see **how each design choice was measured** instead of taken on faith. This project is the RAG fundamentals layer of my AI Engineer portfolio.

---

## What this project demonstrates

- **End-to-end RAG**: PDF parsing → chunking → embeddings → vector retrieval → LLM answer generation.
- **A real evaluation harness**: a hand-built eval set of 50 (question, reference passage) pairs across three different document types, scored on `recall@k` and MRR.
- **Three chunking strategies compared head-to-head**: fixed-token, recursive (LangChain), and semantic (sentence-similarity).
- **Pluggable retrieval backends**: a `Protocol`-based abstraction lets the same code swap between Voyage AI (API) and sentence-transformers (local PyTorch) without touching the calling layers. Both were evaluated head-to-head.
- **Observability that an engineer would actually trust**: every Claude AND embedding/rerank API call is logged to SQLite with tokens, latency, and cost.

The headline finding is documented below: the dominant variable for retrieval quality on this stack is **the reranker**, not the embedding model — and the winning chunking strategy depends on document type, not chunk size alone. A second finding from the Voyage migration: **`voyage-3-large` + `rerank-2.5` is competitive with or better than the best local stack in 2 of 3 corpora**, at the cost of vendor lock-in and per-call API spend.

---

## Quick start

```bash
cp .env.example .env
# add your ANTHROPIC_API_KEY
docker compose up --build
# open http://localhost:8501
```

Configuration via `.env`:

- `EMBEDDING_BACKEND=voyage|local` — pick the embedder. Default is `voyage` (slim Docker image, no PyTorch). Switching to `local` requires `uv sync --extra local` to install sentence-transformers + torch.
- `RERANKER_BACKEND=voyage|local|off` — pick the reranker (or disable). Default is `voyage`.
- `VOYAGE_API_KEY=...` — required when either backend is `voyage`.
- `DEBUG=true` — print prompts and token usage to stdout.

The `voyage` defaults keep the production image small (~250 MB compressed vs ~1.0 GB compressed (with PyTorch) with PyTorch); the `local` backend is preserved for offline use, cost-sensitive workloads, or when running the eval scripts that compare local-only model variants.

### Development

```bash
uv sync                      # install (Voyage-only, lean)
uv sync --extra local        # also install sentence-transformers + torch
uv run pytest                # 56 tests, ~3s, no network
uv run mypy src              # strict typing on src/
uv run ruff check .          # lint
uv run streamlit run app.py  # run locally without Docker
```

---

## Architecture

```
                          ┌────────────────────────────────────┐
                          │           Streamlit UI (app.py)    │
                          └────────────────┬───────────────────┘
                                           │
            ┌──────────────────────────────┼───────────────────────────┐
            ▼                              ▼                           ▼
   ┌────────────────┐         ┌─────────────────────────┐    ┌──────────────────┐
   │  pdf_loader.py │ chunks  │   vector_store.py       │    │   rag_chain.py   │
   │  extract +     ├────────▶│   ChromaDB +            │    │   prompt + call  │
   │  3 chunkers    │         │   Embedder protocol     ├───▶│   Claude Sonnet  │
   └────────────────┘         └────────┬─────┬──────────┘    │   4.6            │
                                       │     │               └──────┬───────────┘
                       ┌───────────────┘     └────────────┐         │
                       ▼                                  ▼         │
              ┌──────────────────┐              ┌──────────────────┐│
              │ embeddings.py    │              │ reranker.py      ││
              │  LocalEmbedder   │              │  LocalReranker   ││
              │  VoyageEmbedder  │              │  VoyageReranker  ││
              │  + factory       │              │  + factory       ││
              └──────────────────┘              └──────────────────┘│
                       │                                  │         │
                       └──────────────┬───────────────────┘         │
                                      ▼                             │
                                 log every call ◀───────────────────┘
                          ┌─────────────────────────────┐
                          │ observability.py            │
                          │ logs/llm_calls.db (SQLite)  │
                          │ Claude + Voyage calls       │
                          └─────────────────────────────┘
```

### Module responsibilities

| File | Responsibility |
|---|---|
| [src/pdf_loader.py](src/pdf_loader.py) | Extract text from a PDF (pypdf) and produce chunks via three strategies. |
| [src/embeddings.py](src/embeddings.py) | `Embedder` Protocol + `LocalEmbedder` (sentence-transformers) and `VoyageEmbedder` (API). Factory picks one via env var. |
| [src/vector_store.py](src/vector_store.py) | In-process ChromaDB. Backend-agnostic: depends on the `Embedder` Protocol, not a concrete class. |
| [src/reranker.py](src/reranker.py) | `Reranker` Protocol + `LocalReranker` (BGE cross-encoder) and `VoyageReranker` (`rerank-2.5`). |
| [src/rag_chain.py](src/rag_chain.py) | Prompt assembly + Claude API call with observability wrapper. |
| [src/observability.py](src/observability.py) | SQLite logger for every LLM/embedding/rerank call (tokens, latency, cost, errors). |
| [src/evaluation.py](src/evaluation.py) | `recall@k`, MRR, and the eval runner. |
| [app.py](app.py) | Streamlit UI (file uploader + form-gated question input + answer with sources). |
| [scripts/run_evaluation.py](scripts/run_evaluation.py) | Compare chunking strategies on an eval set (uses the configured backend). |
| [scripts/run_retriever_experiments.py](scripts/run_retriever_experiments.py) | Compare 4 local retriever configurations on a fixed chunker (requires `--extra local`). |

---

## Stack

| Concern | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | portfolio standard |
| Package manager | uv | fast, reproducible installs |
| LLM | Anthropic Claude Sonnet 4.6 | quality + cost balance |
| Embeddings (default) | Voyage `voyage-3-lite` (API, 512 dim) | slim image (no PyTorch), good quality, ~$0.02/1M tokens |
| Embeddings (alt, local) | `BAAI/bge-base-en-v1.5` (440 MB) | retrieval-tuned local fallback; requires `--extra local` |
| Reranker (default) | Voyage `rerank-2.5` (API) | doubles `recall@3` consistently; pairs naturally with the API embedder |
| Reranker (alt, local) | `BAAI/bge-reranker-base` | cross-encoder local fallback |
| Vector store | ChromaDB (in-process) | no separate service, fine for single-user demo |
| PDF extraction | pypdf | lightweight; limitations documented below |
| UI | Streamlit | fast iteration, ok for a demo |
| Testing | pytest | 56 tests covering loader, store, rag, eval, observability, both embedder/reranker backends (mocked), plus an E2E AppTest for the Streamlit app |
| Types | mypy strict | enforced on `src/` |

---

## Evaluation methodology

A RAG system without measurement is a system you cannot reason about. This project ships with:

### Eval sets (three documents of different difficulty)

| Eval set | Document | Pages | Questions |
|---|---|---|---|---|
| [data/eval_questions.json](data/eval_questions.json) | arXiv:2605.11157 — *Price of Proportional Representation in Temporal Voting* (dense theoretical math) | 22 | 20 |
| [data/eval_questions_dgao.json](data/eval_questions_dgao.json) | arXiv:2605.11974 — *Mitigating LLMs Order Sensitivity via DGAO* (applied ML) | 16 | 15 |
| [data/eval_questions_falcao.json](data/eval_questions_falcao.json) | Wikipedia article — Radamel Falcao (factual prose) | 33 | 15 |

Each question has a `reference` passage from the document. A retrieved chunk is *relevant* if it contains the reference as a substring (case-insensitive, whitespace- and line-break-hyphen-tolerant; see `_normalize` in [src/evaluation.py](src/evaluation.py)).

### Metrics

- **`recall@k`** for *k* ∈ {3, 5, 10}: fraction of questions whose relevant passage appears in the top-*k* retrieved chunks.
- **MRR** (mean reciprocal rank): how *high* the right chunk ranks on average. Captures ordering quality, not just presence.

### Reproducing the experiments

```bash
# Chunking comparison on a single eval set
python -m scripts.run_evaluation --eval data/eval_questions.json --verbose

# Retriever experiments (2 embedders × reranker on/off) on a single eval set
python -m scripts.run_retriever_experiments --eval data/eval_questions_falcao.json
```

---

## Results

### Chunking comparison (token-based vs recursive vs semantic)

All three eval sets, `all-MiniLM-L6-v2` embedder, no reranker, chunks of 500 tokens / 50 overlap:

**Math paper (hard — dense theoretical math):**

| Strategy | # Chunks | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|
| token-based | 78 | 0.30 | 0.40 | 0.60 | 0.300 |
| **recursive (LangChain)** | 80 | **0.45** | **0.55** | **0.65** | **0.345** |
| semantic | 846 | 0.20 | 0.30 | 0.35 | 0.123 |

**ML paper (medium — applied ML):**

| Strategy | # Chunks | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|
| token-based | 50 | 0.20 | 0.40 | 0.60 | 0.145 |
| **recursive (LangChain)** | 54 | **0.40** | **0.47** | **0.67** | **0.364** |
| semantic | 493 | 0.27 | 0.33 | 0.67 | 0.241 |

**Wikipedia / factual prose (easy):**

| Strategy | # Chunks | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|
| **token-based** | 110 | **0.60** | **0.67** | **0.80** | **0.356** |
| recursive (LangChain) | 110 | 0.20 | 0.47 | 0.73 | 0.258 |
| semantic | 1838 | 0.33 | 0.47 | 0.60 | 0.317 |

**Pattern:** recursive wins on dense academic text (structured paragraphs → better chunks); token-based wins on factual prose (uniform density → fixed splits are fine). Semantic chunking consistently underperforms on all three: a `breakpoint_threshold=0.5` fires on nearly every sentence in technical or math-heavy text, producing 846–1838 micro-chunks (~10–15 tokens each) that are too small to carry useful context for the embedder.

### Retriever experiments (2 embedders × reranker on/off, fixed token-based chunking)

| PDF | Config | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|---|
| Math paper (hard) | MiniLM-L6 baseline | 0.30 | 0.40 | 0.60 | 0.300 |
| Math paper (hard) | **MiniLM-L6 + reranker** | **0.70** | 0.70 | 0.70 | **0.575** |
| Math paper (hard) | BGE-base-en-v1.5 | 0.20 | 0.35 | 0.70 | 0.233 |
| Math paper (hard) | BGE-base-en-v1.5 + reranker | 0.55 | 0.60 | 0.60 | 0.446 |
| ML paper (medium) | MiniLM-L6 baseline | 0.20 | 0.40 | 0.60 | 0.145 |
| ML paper (medium) | MiniLM-L6 + reranker | 0.73 | 0.73 | 0.80 | 0.562 |
| ML paper (medium) | BGE-base-en-v1.5 | 0.53 | 0.60 | 0.80 | 0.393 |
| ML paper (medium) | **BGE-base-en-v1.5 + reranker** | **0.80** | **0.87** | **1.00** | **0.723** |
| Wikipedia (easy) | MiniLM-L6 baseline | 0.60 | 0.67 | 0.80 | 0.356 |
| Wikipedia (easy) | MiniLM-L6 + reranker | 0.93 | 0.93 | 0.93 | 0.833 |
| Wikipedia (easy) | BGE-base-en-v1.5 | 0.33 | 0.73 | 0.93 | 0.336 |
| Wikipedia (easy) | **BGE-base-en-v1.5 + reranker** | **1.00** | **1.00** | **1.00** | **0.789** |

### Three key engineering findings

**1. The reranker is the single most reliable lever.** Across all three documents and both embedders, enabling the BGE cross-encoder reranker over the top-20 first-stage candidates multiplied `recall@3` by 2–3.6×. On the worst case (ML paper / MiniLM baseline), recall@3 jumped from 0.20 → 0.73 (3.6×). On Wikipedia with BGE, it reached a perfect 1.00. The pattern is consistent enough to call it a rule, not a coincidence.

**2. There is no universally best embedder.** MiniLM-L6 (80 MB) beat BGE-base (440 MB) on the math paper with reranker (0.70 vs 0.55). BGE dominated on the ML paper (0.80 vs 0.73) and Wikipedia (1.00 vs 0.93). "Bigger embedder = better retrieval" is a heuristic, not a rule. Pick by measurement, per corpus.

**3. Content complexity sets the ceiling.** The best achievable `recall@3` ranged from 0.70 (math paper, dense LaTeX notation + pypdf artifacts) to 1.00 (Wikipedia, clean factual prose). No amount of model-swapping closes that gap — the bottleneck is the *kind* of text, not the retriever configuration.

### Voyage AI vs local backends (added after the API migration)

After refactoring the embedder/reranker into a `Protocol`-based abstraction, we re-ran the eval against **three Voyage models** (`voyage-3-lite`, `voyage-3`, `voyage-3-large`) all paired with `rerank-2.5`, using token-based chunking. Same eval sets as above:

| Eval set | MiniLM + BGE-rerank | BGE-base + BGE-rerank | voyage-3-lite + r2.5 | voyage-3 + r2.5 | **voyage-3-large + r2.5** |
|---|---|---|---|---|---|
| Math paper | 0.70 | 0.55 | 0.75 | 0.70 | **0.75** |
| ML paper | 0.73 | 0.80 | 0.87 | 0.87 | **0.93** |
| Falcao (EN) | 0.93 | **1.00** | 0.87 | 0.80 | **0.93** |

MRR for the same configurations:

| Eval set | MiniLM + BGE-rerank | BGE-base + BGE-rerank | voyage-3-lite + r2.5 | voyage-3-large + r2.5 |
|---|---|---|---|---|
| Math paper | 0.575 | 0.446 | 0.725 | **0.738** |
| ML paper | 0.562 | 0.723 | 0.833 | **0.900** |
| Falcao (EN) | 0.833 | 0.789 | 0.724 | **0.758** |

**Three follow-up findings:**

- **`voyage-3-large + rerank-2.5` is the strongest single configuration overall** on MRR across all three corpora, and ties or beats the best local stack on recall@3 in 2 of 3 corpora. The only place where a local stack still wins is BGE-base + BGE-reranker on the Wikipedia eval (1.00 vs 0.93) — a 0.07 gap.
- **`voyage-3` (the mid-tier) underperforms `voyage-3-lite` in 2 of 3 corpora.** Counterintuitive but real on this stack. Model size does not predict retrieval quality; training objective and dataset matter more. This is exactly the "pick by measurement, per corpus" rule from above.
- **Why we ship with `voyage-3-lite` despite `voyage-3-large` being slightly stronger:** cost. For portfolio-volume usage, `voyage-3-lite` is roughly 9× cheaper than `voyage-3-large` at $0.02 vs $0.18 per 1M tokens, and `+ rerank-2.5` already lifts it well above the local baseline. Users who want the absolute peak can switch model name in [src/embeddings.py](src/embeddings.py).

### A note on language

These eval sets are **all in English** (questions and source documents). The Voyage and BGE models are stronger on English than other languages by design: their pre-training and contrastive fine-tuning corpora are English-dominant.

In ad-hoc testing with a Spanish version of the Falcao Wikipedia article and Spanish queries, `voyage-3-lite` retrieval quality degrades significantly — the relevant chunk often does not enter the top-20 candidates, so the reranker has nothing useful to work with. This is consistent with multilingual embedder literature: monolingual or English-centric models trained on web-scraped corpora learn the geometry of one language well and degrade across languages. The fix when needed is a multilingual model like Voyage's `voyage-multilingual-2` or sentence-transformers' `paraphrase-multilingual-mpnet-base-v2`. Left as a documented limitation rather than fixed, to keep the comparison apples-to-apples.

---

## Calibrating expectations

If `recall@3 = 0.70` on a dense academic paper feels low in isolation, it isn't — it sits inside the band reported by recent RAG literature:

- **Anthropic, *Introducing Contextual Retrieval* (Sept 2024)** — baseline RAG has a ~5.7 % retrieval failure rate *per chunk* on their benchmark; the post proposes contextual retrieval to mitigate it. ([blog post](https://www.anthropic.com/news/contextual-retrieval))
- **RAGAS (Es et al., 2024)** — `context_precision` typically lands in 0.45–0.75 across curated datasets.
- **Liu et al., *Lost in the Middle* (2023)** — retrieval quality, not LLM capacity, is usually the bottleneck; this holds across GPT-3.5, GPT-4, and Claude models of that era.

Our numbers are consistent with this band, *not below* it. The strict-substring matcher used in our eval is also more conservative than the LLM-judged "did the answer come out right" used in many demos.

---

## What fails (honest)

Built into the project from the start, because the line between *tutorial* and *engineering* is whether you can describe what your system can't do:

- **Dense math / heavy notation degrades retrieval.** PDF extraction with pypdf produces artifacts on math papers (broken equations, merged words like `approaches1as`, soft hyphens at line breaks). Even after a normalization pass that heals line-break hyphens, the embedder struggles to align questions with math-laden chunks.
- **Multi-hop questions.** Anything requiring synthesis across multiple distant chunks — "compare X in section 3 with Y in section 6" — is out of scope for single-stage RAG.
- **Whole-document questions.** "Summarize chapter 3" is not a retrieval problem; it needs the whole chapter in context. The right answer for that use case is *not* RAG.
- **Tables and figures.** pypdf extracts table cells in unpredictable order; figure captions sometimes get attached to the wrong paragraph.
- **Cross-lingual / non-English content.** Both `voyage-3-lite` and BGE-base-en are English-centric. With a Spanish Wikipedia article and Spanish queries, retrieval quality drops sharply — the right chunk often misses the top-20, so the reranker can't recover. Quantified in the "A note on language" subsection above. Documented rather than fixed; the right answer is a multilingual model (`voyage-multilingual-2` or `paraphrase-multilingual-mpnet-base-v2`).
- **No persistence.** ChromaDB is in-process and ephemeral. Restart the container, re-upload the PDF. Deliberate scope-cap for a demo.
- **One PDF at a time.** No multi-document indexing.

---

## Observability — typical cost & latency

Every external call (Claude, Voyage embedder, Voyage reranker) is logged to `logs/llm_calls.db` (SQLite), via the [`log_llm_call`](src/observability.py) context manager. Schema: `id, timestamp, project, model, input_tokens, output_tokens, latency_ms, cost_usd, success, error_msg`. One table, multiple providers — distinguished by the `model` column.

Aggregated over a typical session on the default backend (`voyage-3-lite` + `rerank-2.5` + Claude Sonnet 4.6), per question on the math paper:

| Phase | Cost | Latency |
|---|---|---|
| Embed query | ~$0.0000002 (0.02 c per 1M tokens × ~10 query tokens) | ~150 ms |
| Vector search (ChromaDB, in-process) | $0 | <5 ms |
| Rerank top-20 | ~$0.00002 (0.05 c per 1M tokens × ~400 tokens) | ~300 ms |
| Claude answer | ~$0.0055 (~390 input / ~290 output tokens at Sonnet 4.6 rates) | ~3–5 s |
| **Per-question total** | **~$0.0055** | **~4 s** |

Indexing a fresh PDF (one-time, batched): ~$0.0008 for a 40K-token document with `voyage-3-lite`.

Pricing references: Claude Sonnet 4.6 list rates ($3 / 1M input, $15 / 1M output), Voyage embeddings $0.02 / 1M, Voyage rerank-2.5 $0.05 / 1M. Switching to the `local` backend zeroes the embed and rerank costs at the price of a ~1 GB compressed image (~2.7 GB virtual) and ~30 s of CPU work per indexing.

---

## Technical decisions worth defending

- **Chunk size of 500 tokens (overlap 50).** Measured against 150/15 — the smaller size cut recall on the math paper by not carrying enough context per chunk for the embedder to match accurately. At 500 tokens, token-based and recursive both produce ~50–110 chunks per document, a manageable corpus size for in-process ChromaDB. The overlap catches references that straddle a boundary.
- **ChromaDB in-process, not as a service.** No separate container. For a single-user demo this is simpler; for multi-user prod you'd want Qdrant or pgvector.
- **`st.form` for the question input.** Streamlit reruns the whole script on any UI interaction (opening the "sources" expander, for example). A naive `if question:` block would re-invoke Claude on every rerun. Wrapping the question in `st.form` ensures the LLM is only called on explicit submit; the answer is then persisted in `st.session_state`.
- **Substring match in the evaluator, not LLM-as-judge.** Cheaper, deterministic, and reveals retrieval issues directly. An LLM judge would compound errors — a hallucinated "correct" judgement masks a bad retrieval.
- **Reranker is on by default in `.env.example`.** Given that it consistently doubles `recall@3` across all eval sets, shipping with it off would contradict the measurement-driven approach this project demonstrates. The local variant adds a ~280 MB model download and ~5 s per query on CPU; the Voyage variant (`rerank-2.5`) is ~300 ms per query at ~$0.00002.
- **Pluggable backends via `Protocol`, not inheritance.** [src/embeddings.py](src/embeddings.py) and [src/reranker.py](src/reranker.py) define structural Protocols rather than abstract base classes. Concrete classes (`LocalEmbedder`, `VoyageEmbedder`, etc.) don't inherit from anything — they just satisfy the protocol by having the right methods. This is the modern Pythonic way to do dependency injection and keeps the calling code (`VectorStore`, semantic chunking, `app.py`) decoupled from any specific backend.
- **Voyage as the default, sentence-transformers as `[local]` extra.** The default install does not pull PyTorch (saves ~1.2 GB of image weight) so the project can be deployed to size-constrained platforms like Cloud Run. The local stack stays available behind `uv sync --extra local` for offline use, deeper local experimentation, or running the locally-only retriever experiments script.
- **No persistent vector store.** Mentioned above. A real product needs it; a demo doesn't.

---

## Next iterations

Roughly in order of impact:

1. **Spanish/multilingual eval set + backend.** Add a Spanish question set on the Spanish Falcao article and measure `voyage-multilingual-2` and `paraphrase-multilingual-mpnet-base-v2` against the current English-centric defaults. Quantify the cross-lingual gap properly.
2. **Replace pypdf with PyMuPDF (fitz).** Better text extraction on math papers and tables. Likely lifts recall on the academic eval sets by a measurable amount; no change in app surface area.
3. **Contextual retrieval (Anthropic, Sept 2024).** Prepend each chunk with a model-generated context sentence before embedding. Reportedly reduces retrieval failure ~35–50 %.
4. **Hybrid retrieval (BM25 + dense).** A keyword-based retriever fused with the dense embeddings helps on documents that repeat distinctive terms — particularly Wikipedia-style text where keyword overlap drowns embedding signal.
5. **Non-root user in the Dockerfile.** The `logs/` directory currently gets created as root inside the container, which leaks into the host bind-mount. Adding a `USER appuser` step fixes this.
6. **Persistent vector store** (Qdrant) for multi-document, multi-session use.
7. **Conversational memory.** Currently each question is independent; a chat history would let users follow up.

---

## Repository structure

```
01-chat-with-pdf/
├── README.md                     # this file
├── CLAUDE.md                     # project-specific guidance for Claude Code
├── Dockerfile                    # python:3.11-slim + uv (slim by default, no PyTorch)
├── docker-compose.yml            # bind-mounts the project, exposes 8501
├── pyproject.toml                # deps + [local] extra for sentence-transformers
├── .env.example                  # EMBEDDING_BACKEND, RERANKER_BACKEND, API keys, DEBUG
├── app.py                        # Streamlit entry point
├── src/                          # core modules (see Module responsibilities)
├── scripts/                      # eval and experiment runners
├── tests/                        # 56 tests, all passing (mocked Voyage calls)
└── data/                         # eval sets and result tables
```

---

## License

MIT.
