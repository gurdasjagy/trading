FROM python:3.12-slim-bookworm

# System deps (libzmq3-dev needed for pyzmq wheel build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure Python can resolve top-level packages (core, ai, monitoring, config, …)
ENV PYTHONPATH=/app

# NOTE: The Rust trading_engine binary runs in its own container (rust-engine).
# There is no need to build the PyO3 extension here — the previous maturin
# command referenced a path (/app/../rust_engine) that doesn't exist inside
# this image's build context and was silently failing via || true.

CMD ["python", "main.py", "run"]

