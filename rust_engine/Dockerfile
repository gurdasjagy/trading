# Stage 1: Build
FROM rust:1.85-slim-bookworm AS builder

RUN apt-get update && apt-get install -y \
    pkg-config libssl-dev cmake python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/rust_engine
COPY rust_engine/Cargo.toml rust_engine/Cargo.lock ./
COPY rust_engine/src/ src/

# Build with maximum optimization
ENV RUSTFLAGS="-C target-cpu=x86-64-v3"
RUN cargo build --release --bin trading_engine

# Stage 2: Runtime (minimal)
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/rust_engine/target/release/trading_engine /usr/local/bin/

# Copy templates and static files for the Rust dashboard server
COPY crypto_trading_bot/templates/ /app/templates/
COPY crypto_trading_bot/static/ /app/static/

# Expose dashboard + ZeroMQ ports
EXPOSE 8080 5555 5556

# Health check via heartbeat file
HEALTHCHECK --interval=5s --timeout=3s --retries=3 \
    CMD test -f /tmp/engine_heartbeat || exit 1

ENTRYPOINT ["trading_engine"]
CMD ["--config", "/etc/trading/engine_config.toml"]

