.PHONY: install install-dev run paper backtest dashboard test docker-up docker-down \
        docker-logs docker-logs-rl docker-logs-execution docker-logs-profit docker-shell \
        docker-status docker-rebuild docker-clean generate-keys setup lint format clean help

# ============================================================
# Default target
# ============================================================
help:
	@echo "Crypto Trading Bot — Available Targets"
	@echo "======================================="
	@echo "  install        Install production Python dependencies"
	@echo "  install-dev    Install production + development/testing dependencies"
	@echo "  run            Start the bot (live/paper based on .env)"
	@echo "  paper          Start the bot in paper trading mode"
	@echo "  backtest       Run backtesting engine"
	@echo "  dashboard      Launch the web dashboard only"
	@echo "  test           Run test suite with pytest"
	@echo "  docker-up      Start all services with docker-compose"
	@echo "  docker-down    Stop all docker-compose services"
	@echo "  docker-logs    Tail logs from the bot container"
	@echo "  docker-logs-rl Tail logs filtered for RL-related messages"
	@echo "  docker-logs-execution Tail logs filtered for execution engine messages"
	@echo "  docker-logs-profit Tail logs filtered for profit maximization messages"
	@echo "  docker-shell   Open bash shell in running bot container"
	@echo "  docker-status  Show container status and resource usage"
	@echo "  generate-keys  Generate encryption/JWT keys"
	@echo "  setup          Run interactive first-time setup"
	@echo "  lint           Check code style (ruff + black)"
	@echo "  format         Auto-format code (black + ruff --fix)"
	@echo "  clean          Remove build artifacts and caches"

# ============================================================
# Development
# ============================================================
install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	pip install --upgrade pip
	pip install -r requirements-dev.txt

run:
	python main.py run

paper:
	TRADING_MODE=paper python main.py run

backtest:
	python main.py backtest --strategy momentum --symbol BTC/USDT --from 2025-01-01 --to 2025-12-31

dashboard:
	python main.py dashboard

# ============================================================
# Testing
# ============================================================
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term-missing

# ============================================================
# Docker
# ============================================================
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f bot

docker-logs-rl:
	docker-compose logs -f bot | grep -iE "(rl|reinforcement|parameter.*tun|bayesian|optimiz)"

docker-logs-execution:
	docker-compose logs -f bot | grep -iE "(execution|twap|vwap|iceberg|sniper|latency|slippage)"

docker-logs-profit:
	docker-compose logs -f bot | grep -iE "(profit|compound|leverage.*scal|trade.*quality|drawdown)"

docker-shell:
	docker exec -it crypto_trading_bot bash

docker-status:
	@echo "=== Container Status ==="
	@docker ps -a --filter "name=crypto_" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
	@echo ""
	@echo "=== Resource Usage ==="
	@docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" crypto_trading_bot crypto_redis crypto_timescaledb

docker-rebuild:
	docker-compose up -d --build

docker-clean:
	docker-compose down -v --remove-orphans

# ============================================================
# Setup & Key Management
# ============================================================
generate-keys:
	python main.py generate-keys

setup:
	python main.py setup

# ============================================================
# Code Quality
# ============================================================
lint:
	ruff check . && black --check .

format:
	black . && ruff check --fix .

# ============================================================
# Cleanup
# ============================================================
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "coverage.xml" -delete 2>/dev/null || true
	@echo "Clean complete."
