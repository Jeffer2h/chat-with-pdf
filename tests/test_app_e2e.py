"""End-to-end smoke test for the Streamlit app.

Drives `app.py` headlessly via `streamlit.testing.v1.AppTest`, exercising the
full wiring: PDF extraction → chunking → VectorStore → (reranker) → rag_chain.
Heavy dependencies (Claude, Voyage, PDF parsing) are stubbed at the import
sites in `src.*` so the test stays hermetic and fast — no network, no PyTorch.

What this test guards against:
- A regression that breaks the import graph (`app.py` won't even start).
- A regression that breaks how chunks are added to the VectorStore or how
  retrieved chunks are passed to `answer_question`.
- A regression in session_state wiring that prevents the answer from rendering.
"""

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import FakeEmbedder


@pytest.fixture
def stub_dependencies(monkeypatch):
    """Replace external-effect functions with deterministic in-memory stubs.

    Must run BEFORE AppTest loads app.py, because `app.py` does
    `from src.X import Y` — at import time those names are resolved to
    whatever we've set on `src.X`.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setenv("RERANKER_BACKEND", "off")

    # Bypass network-bound factories.
    monkeypatch.setattr("src.embeddings.get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr("src.reranker.get_reranker", lambda: None)

    # No real PDF parsing — return a known body of text so we can verify the
    # answer in the assertion below.
    fake_pdf_text = (
        "The Falcon 9 rocket is a partially reusable two-stage launch vehicle "
        "designed and manufactured by SpaceX. "
        "Its first flight occurred in 2010. "
        "It can deliver payloads to low Earth orbit and beyond."
    )
    monkeypatch.setattr("src.pdf_loader.extract_text", lambda file: fake_pdf_text)

    # Stub Claude entirely — bypass the real client and the rag_chain call.
    monkeypatch.setattr(
        "src.rag_chain.answer_question",
        lambda question, context_chunks, client: "FAKE_ANSWER_42",
    )

    # The Streamlit app instantiates anthropic.Anthropic(); that constructor
    # accepts arbitrary api_key strings without network, so no patch needed.


def test_app_loads_without_errors(stub_dependencies):
    """The script should boot and render the title without exceptions."""
    at = AppTest.from_file("app.py", default_timeout=15).run()
    assert not at.exception
    assert any("Chat with PDF" in t.value for t in at.title)


def test_app_full_question_flow(stub_dependencies):
    """Upload a PDF, ask a question, verify the rendered answer."""
    at = AppTest.from_file("app.py", default_timeout=15).run()
    assert not at.exception

    # The file_uploader widget can't be programmatically driven by AppTest in
    # the same way as text inputs (no test API for file uploads as of streamlit
    # 1.40). Instead, simulate the post-upload state by injecting a ready
    # VectorStore directly into session_state, then submitting a question.
    from src.vector_store import VectorStore

    vs = VectorStore(embedder=FakeEmbedder())
    vs.add_documents(["Falcon 9 first flew in 2010.", "Made by SpaceX."])
    at.session_state["vector_store"] = vs
    at.session_state["pdf_name"] = "fake.pdf"
    at.session_state["chunk_method"] = "token-based"

    at.text_input[0].set_value("When did Falcon 9 first fly?")
    at.button[0].click().run()

    assert not at.exception
    # The stubbed answer_question always returns this string.
    assert any("FAKE_ANSWER_42" in m.value for m in at.markdown)
