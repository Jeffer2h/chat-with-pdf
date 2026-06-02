FROM python:3.11-slim

# uv binary (~10MB) installed from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Venv outside /app so the bind mount doesn't clobber it
ENV UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Deps layer: cached until pyproject.toml or uv.lock changes.
# We deliberately omit `--extra local` so sentence-transformers / torch are not
# installed. The image compresses to ~250 MB on push (vs ~1 GB with PyTorch),
# which is what Cloud Run actually pulls on cold start. Local backend can
# still be used during development with `uv sync --extra local` outside Docker.
COPY pyproject.toml uv.lock* ./
# Install deps and strip dead weight IN THE SAME LAYER. Docker layers are
# immutable: deleting files in a later RUN only marks a whiteout, the bytes
# remain in the previous layer. To actually shrink the image we must remove
# the cruft inline.
#   - __pycache__: bytecode regenerates on first import (slightly slower cold start).
#   - per-package tests/ dirs: numpy/pyarrow/jsonschema/etc. ship their own test
#     suites; never imported at runtime.
RUN uv sync --no-dev --no-install-project \
 && find /opt/venv -type d -name __pycache__ -exec rm -rf {} + \
 && find /opt/venv/lib/python*/site-packages -maxdepth 2 -type d -name tests -exec rm -rf {} +

# Project code
COPY . .

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]
