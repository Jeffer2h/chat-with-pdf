import os

import anthropic
import streamlit as st

from src.embeddings import Embedder, get_embedder
from src.observability import read_session_stats
from src.chunking import (
    split_into_chunks,
    split_into_chunks_recursive,
    split_into_chunks_semantic,
)
from src.pdf_loader import extract_text
from src.rag_chain import retrieve_and_answer
from src.reranker import Reranker, get_reranker
from src.vector_store import VectorStore

st.set_page_config(page_title="Chat with PDF", page_icon="📄")
st.title("📄 Chat with PDF")
st.caption("Upload a PDF and ask questions about its content.")

# --- Cached resources (loaded once per server process) ---


@st.cache_resource
def get_client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("ANTHROPIC_API_KEY environment variable is not set.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


@st.cache_resource
def cached_embedder() -> Embedder:
    return get_embedder()


@st.cache_resource
def cached_reranker() -> Reranker | None:
    return get_reranker()


client = get_client()
embedder = cached_embedder()
reranker = cached_reranker()

# --- Session state initialization ---
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "chunk_method" not in st.session_state:
    st.session_state.chunk_method = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_chunks" not in st.session_state:
    st.session_state.last_chunks = []
if "last_question" not in st.session_state:
    st.session_state.last_question = None

# --- Sidebar: chunking method and backend info ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown(f"**Embedder:** `{embedder.name}`")
    st.markdown(
        f"**Reranker:** {'`' + reranker.name + '`' if reranker else 'off'} *(env var)*"
    )
    chunk_method = st.radio(
        "Chunking method",
        options=["token-based", "recursive", "semantic"],
        help=(
            "**Token-based:** splits every N tokens with overlap. "
            "Fast and predictable.\n\n"
            "**Recursive:** splits hierarchically on paragraphs → sentences → chars. "
            "Respects natural text structure.\n\n"
            "**Semantic:** cuts at topic boundaries using embedding similarity. "
            "Better context preservation, slower."
        ),
    )

    # Observability panel — surfaces the SQLite log so the cost/latency
    # convention is visible at a glance instead of buried in logs/llm_calls.db.
    st.divider()
    st.subheader("📊 Session stats")
    stats = read_session_stats()
    col1, col2 = st.columns(2)
    col1.metric("API calls", stats["total_calls"])
    col2.metric("Cost (USD)", f"${stats['total_cost_usd']:.4f}")
    col1.metric("Input tokens", f"{stats['total_input_tokens']:,}")
    col2.metric("Avg latency", f"{stats['avg_latency_ms']:.0f} ms")
    st.caption("Aggregated from `logs/llm_calls.db`.")

# --- PDF upload ---
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

pdf_changed = uploaded_file and uploaded_file.name != st.session_state.pdf_name
method_changed = chunk_method != st.session_state.chunk_method

if uploaded_file and (pdf_changed or method_changed):
    # Clear stale results so the displayed answer always matches the active index.
    st.session_state.last_answer = None
    st.session_state.last_question = None
    st.session_state.last_chunks = []

    with st.spinner(f"Processing PDF with {chunk_method} chunking..."):
        try:
            text = extract_text(uploaded_file)
        except Exception as e:
            st.error(f"Could not read the PDF: {e}")
            st.stop()

        # A scanned/image-only PDF yields no extractable text. Stop here with a
        # clear message instead of indexing zero chunks and silently accepting
        # questions that can never be answered.
        if not text.strip():
            st.error(
                "This PDF has no extractable text — it may be scanned or "
                "image-only. OCR is out of scope for this demo."
            )
            st.stop()

        if chunk_method == "semantic":
            chunks = split_into_chunks_semantic(text, embedder=embedder)
        elif chunk_method == "recursive":
            chunks = split_into_chunks_recursive(text)
        else:
            chunks = split_into_chunks(text)

        vs = VectorStore(embedder=embedder)
        vs.add_documents(chunks)
        st.session_state.vector_store = vs
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.chunk_method = chunk_method

    st.success(
        f"'{uploaded_file.name}' processed — "
        f"{len(chunks)} chunks indexed ({chunk_method})."
    )

# --- Question input ---
st.divider()

vector_store_ready = st.session_state.vector_store is not None

with st.form("question_form", clear_on_submit=False):
    question = st.text_input(
        "Ask a question about the document",
        placeholder="e.g. What is the main topic of this document?",
        disabled=not vector_store_ready,
    )
    submitted = st.form_submit_button("Ask", disabled=not vector_store_ready)

# Only call the API when the form is submitted with a non-empty question.
# Streamlit reruns the whole script on any UI interaction (e.g. opening an
# expander), so we cannot trigger the call from a plain `if question:` block —
# that would re-invoke Claude on every rerun.
if submitted and question and vector_store_ready:
    with st.spinner("Searching and generating answer..."):
        answer, relevant_chunks = retrieve_and_answer(
            question, st.session_state.vector_store, reranker, client
        )
        st.session_state.last_question = question
        st.session_state.last_answer = answer
        st.session_state.last_chunks = relevant_chunks

# Render the latest answer from session_state so it persists across reruns.
if st.session_state.last_answer:
    st.markdown("### Answer")
    st.caption(f"Q: {st.session_state.last_question}")
    st.write(st.session_state.last_answer)

    with st.expander("View source excerpts used"):
        for i, chunk in enumerate(st.session_state.last_chunks, 1):
            st.markdown(f"**Excerpt {i}:**")
            st.text(chunk)
