from typing import BinaryIO

from pypdf import PdfReader


def extract_text(file: BinaryIO) -> str:
    """Extract all text from a PDF file.

    Args:
        file: File-like object (e.g. from Streamlit's file_uploader).

    Returns:
        Concatenated text from all pages.
    """
    reader = PdfReader(file)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text)
