# syntax=docker/dockerfile:1
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-default-groups --group cpu


FROM python:3.13-slim AS runtime

RUN useradd --create-home --uid 1000 user

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user

WORKDIR /app

COPY --from=builder --chown=user /app/.venv /app/.venv
COPY --chown=user src/ src/

USER user

EXPOSE 8501
# `python -m` puts /app on sys.path so the `src.*` imports resolve.
CMD ["python", "-m", "streamlit", "run", "src/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
