#!/usr/bin/env bash
# setup_env.sh — Bootstrap the crypto trading bot environment.
# Usage: bash scripts/setup_env.sh

set -euo pipefail

BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
RED="\033[31m"
RESET="\033[0m"

info()    { echo -e "${GREEN}[INFO]${RESET}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }

# ── 1. Check Python version ───────────────────────────────────────────────
info "Checking Python version…"
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Please install Python 3.11 or later."
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 11 ]; }; then
    error "Python 3.11+ required but found Python ${PYTHON_VERSION}."
fi
info "Python ${PYTHON_VERSION} ✓"

# ── 2. Create virtual environment ─────────────────────────────────────────
VENV_DIR="${VENV_DIR:-.venv}"
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment in ./${VENV_DIR}…"
    python3 -m venv "$VENV_DIR"
else
    info "Virtual environment already exists at ./${VENV_DIR} — skipping creation."
fi

# Activate the venv
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
info "Virtual environment activated."

# ── 3. Install requirements ───────────────────────────────────────────────
if [ -f "requirements.txt" ]; then
    info "Installing dependencies from requirements.txt…"
    pip install --quiet --upgrade pip
    pip install --quiet -r requirements.txt
    info "Dependencies installed ✓"
else
    warn "requirements.txt not found — skipping pip install."
fi

# ── 4. Copy .env.example → .env ──────────────────────────────────────────
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        info ".env created from .env.example ✓"
        warn "Please edit .env and fill in your API keys before running the bot."
    else
        warn ".env.example not found — skipping .env creation."
    fi
else
    info ".env already exists — skipping copy."
fi

# ── 5. Run Alembic migrations ─────────────────────────────────────────────
if command -v alembic &>/dev/null && [ -f "alembic.ini" ]; then
    info "Running Alembic database migrations…"
    alembic upgrade head
    info "Migrations applied ✓"
else
    warn "alembic not found or alembic.ini missing — skipping migrations."
fi

# ── Success ───────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}✅  Setup complete!${RESET}"
echo ""
echo "Next steps:"
echo "  1. Edit .env with your exchange API keys"
echo "  2. Run: python scripts/generate_keys.py"
echo "  3. Optionally: python scripts/download_models.py"
echo "  4. Start the bot: python main.py"
echo ""
