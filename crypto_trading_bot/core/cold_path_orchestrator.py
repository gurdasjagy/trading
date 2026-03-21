"""Cold Path Orchestrator — Unified Python Services for 4vCPU VPS.

Runs all Python cold-path services in a single process to minimize
resource usage on constrained VPS deployments:

  1. Regime Detection (ML-based market regime classification)
  2. Sentiment Analysis (social + news + LLM)
  3. Graceful Degradation Manager (resource monitoring)
  4. PnL Attribution Engine (trade analytics)
  5. Dashboard / Health Endpoints (aiohttp)
  6. Forex / TradFi Session Manager (if enabled)
  7. Alert Dispatcher (Telegram, Discord)

Each service runs as an async task within a single event loop.
The orchestrator manages lifecycle, health checks, and clean shutdown.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from typing import Any, Dict, Optional

from loguru import logger

# Configure logging early
logger.remove()
logger.add(
    sys.stderr,
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
)


class ColdPathOrchestrator:
    """Manages all Python cold-path services in a unified process."""

    def __init__(self) -> None:
        self._tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        self._start_time = time.time()

        # Service instances (lazily initialized)
        self._degradation_manager = None
        self._pnl_engine = None
        self._health_app = None
        self._alpha_oracle = None
        self._ai_brain = None

    async def start(self) -> None:
        """Start all cold-path services."""
        logger.info("=" * 60)
        logger.info("Cold Path Orchestrator starting (4vCPU VPS mode)")
        logger.info("=" * 60)

        self._running = True

        # Install signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(self.shutdown()))

        # 1. Start Graceful Degradation Manager (must be first)
        await self._start_degradation_manager()

        # 2. Start Health & Status endpoints
        await self._start_health_server()

        # 3. Start PnL Attribution Engine
        await self._start_pnl_engine()

        # 4. Start Regime Detection Loop
        self._tasks["regime"] = asyncio.create_task(
            self._regime_detection_loop(), name="regime"
        )

        # 5. Start Sentiment Analysis Loop (if not degraded)
        self._tasks["sentiment"] = asyncio.create_task(
            self._sentiment_analysis_loop(), name="sentiment"
        )

        # 6. Start Forex Session Manager (if enabled)
        if os.getenv("ENABLE_FOREX_TRADING", "").lower() in ("true", "1", "yes"):
            self._tasks["forex_session"] = asyncio.create_task(
                self._forex_session_loop(), name="forex_session"
            )

        # 7. Start Alert Dispatcher
        self._tasks["alerts"] = asyncio.create_task(
            self._alert_dispatcher_loop(), name="alerts"
        )

        # 8. Start Trade Journal Sync
        self._tasks["journal_sync"] = asyncio.create_task(
            self._journal_sync_loop(), name="journal_sync"
        )

        # 9. Initialize AI Brain (LLM-powered analysis for Alpha Oracle)
        await self._init_ai_brain()

        # 10. Start Alpha Oracle (Confluence Engine + SHM Signal Queue)
        await self._start_alpha_oracle()
        self._tasks["alpha_oracle"] = asyncio.create_task(
            self._alpha_oracle_loop(), name="alpha_oracle"
        )

        logger.info("All cold-path services started ({} tasks)", len(self._tasks))

        # Wait for all tasks (or until shutdown)
        try:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        except asyncio.CancelledError:
            pass

    async def shutdown(self) -> None:
        """Gracefully shut down all services."""
        if not self._running:
            return
        self._running = False
        logger.warning("Shutting down Cold Path Orchestrator...")

        # Stop degradation manager
        if self._degradation_manager:
            await self._degradation_manager.stop()

        # Cancel all tasks
        for name, task in self._tasks.items():
            if not task.done():
                task.cancel()
                logger.info("Cancelled task: {}", name)

        # Wait for tasks to complete
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)

        logger.info("Cold Path Orchestrator shutdown complete")

    async def _start_degradation_manager(self) -> None:
        """Initialize and start the graceful degradation manager."""
        try:
            from core.graceful_degradation import GracefulDegradationManager
            self._degradation_manager = GracefulDegradationManager(
                check_interval_seconds=5.0
            )
            self._degradation_manager.on_level_change(self._on_degradation_change)
            await self._degradation_manager.start()
            logger.info("Graceful Degradation Manager started")
        except Exception as e:
            logger.error("Failed to start Degradation Manager: {}", e)

    async def _start_health_server(self) -> None:
        """Start minimal health/status endpoints only.
        
        Dashboard is served by Rust engine on :8080. Python only provides
        lightweight /health, /status, and /metrics endpoints on :8081.
        """
        await self._start_fallback_health_server()
    
    async def _start_fallback_health_server(self) -> None:
        """Start minimal health endpoints if dashboard fails."""
        try:
            from aiohttp import web

            app = web.Application()
            app.router.add_get("/health", self._health_handler)
            app.router.add_get("/status", self._status_handler)
            app.router.add_get("/metrics", self._metrics_handler)

            runner = web.AppRunner(app)
            await runner.setup()
            fallback_port = int(os.getenv("DASHBOARD_PORT", "8081"))
            site = web.TCPSite(runner, "0.0.0.0", fallback_port)
            await site.start()
            logger.info("Fallback health server listening on :{}", fallback_port)
        except Exception as e:
            logger.error("Failed to start fallback health server: {}", e)

    async def _start_pnl_engine(self) -> None:
        """Initialize the PnL Attribution Engine."""
        try:
            from analytics.pnl_attribution import PnLAttributionEngine
            self._pnl_engine = PnLAttributionEngine()
            logger.info("PnL Attribution Engine initialized")
        except Exception as e:
            logger.error("Failed to initialize PnL engine: {}", e)

    async def _health_handler(self, request) -> Any:
        """Handle /health requests."""
        from aiohttp import web
        status = {
            "status": "healthy",
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "tasks": {
                name: "running" if not task.done() else "stopped"
                for name, task in self._tasks.items()
            },
        }
        if self._degradation_manager:
            status["degradation_level"] = self._degradation_manager.level.name
        return web.json_response(status)

    async def _status_handler(self, request) -> Any:
        """Handle /status requests (detailed)."""
        from aiohttp import web
        status = {
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "tasks": {},
        }
        for name, task in self._tasks.items():
            status["tasks"][name] = {
                "running": not task.done(),
                "cancelled": task.cancelled() if task.done() else False,
            }
        if self._degradation_manager:
            status["degradation"] = self._degradation_manager.get_status()
        if self._pnl_engine:
            status["pnl_summary"] = await self._pnl_engine.get_full_report()
        return web.json_response(status)

    async def _metrics_handler(self, request) -> Any:
        """Handle /metrics requests (Prometheus format)."""
        from aiohttp import web
        lines = []
        lines.append(f"# HELP cold_path_uptime_seconds Uptime in seconds")
        lines.append(f"# TYPE cold_path_uptime_seconds gauge")
        lines.append(f'cold_path_uptime_seconds {time.time() - self._start_time:.1f}')

        if self._degradation_manager:
            level = self._degradation_manager.level
            lines.append(f"# HELP degradation_level Current degradation level 0-4")
            lines.append(f"# TYPE degradation_level gauge")
            lines.append(f"degradation_level {int(level)}")

            snap = self._degradation_manager.last_snapshot
            if snap:
                lines.append(f"# HELP system_cpu_percent CPU usage percent")
                lines.append(f"# TYPE system_cpu_percent gauge")
                lines.append(f"system_cpu_percent {snap.cpu_percent:.1f}")
                lines.append(f"# HELP system_memory_percent Memory usage percent")
                lines.append(f"# TYPE system_memory_percent gauge")
                lines.append(f"system_memory_percent {snap.memory_percent:.1f}")

        for name, task in self._tasks.items():
            running = 0 if task.done() else 1
            lines.append(f'cold_path_task_running{{name="{name}"}} {running}')

        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")

    async def _regime_detection_loop(self) -> None:
        """Periodic regime detection and shared memory update."""
        interval = int(os.getenv("REGIME_INTERVAL_SECONDS", "300"))
        logger.info("Regime detection loop starting (interval={}s)", interval)

        while self._running:
            try:
                # Check degradation level
                if self._degradation_manager:
                    actions = self._degradation_manager.get_recommended_actions()
                    interval = actions.get("regime_interval_seconds", interval)

                # Import regime detector lazily
                try:
                    from ai.regime_detector import RegimeDetector
                    detector = RegimeDetector()
                    regime = await detector.detect_current_regime()
                    # Write to shared memory for Rust engine
                    shm_path = os.getenv("REGIME_SHM_PATH", "/dev/shm/regime_weights")
                    self._write_regime_to_shm(regime, shm_path)
                    logger.info("Regime updated: {}", regime)
                except ImportError:
                    logger.debug("RegimeDetector not available — skipping")
                except Exception as e:
                    logger.error("Regime detection error: {}", e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Regime loop error: {}", e)

            await asyncio.sleep(interval)

    async def _sentiment_analysis_loop(self) -> None:
        """Periodic sentiment analysis."""
        interval = int(os.getenv("SENTIMENT_INTERVAL_SECONDS", "900"))
        logger.info("Sentiment analysis loop starting (interval={}s)", interval)

        while self._running:
            try:
                # Check if AI is enabled under degradation
                if self._degradation_manager:
                    actions = self._degradation_manager.get_recommended_actions()
                    if not actions.get("sentiment_polling", True):
                        logger.debug("Sentiment polling disabled by degradation manager")
                        await asyncio.sleep(60)
                        continue

                try:
                    from ai.sentiment_aggregator import SentimentAggregator
                    aggregator = SentimentAggregator()
                    sentiment = await aggregator.aggregate_sentiment()
                    logger.info("Sentiment updated: {}", sentiment)
                except ImportError:
                    logger.debug("SentimentAggregator not available — skipping")
                except Exception as e:
                    logger.error("Sentiment error: {}", e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Sentiment loop error: {}", e)

            await asyncio.sleep(interval)

    async def _forex_session_loop(self) -> None:
        """Manage forex trading sessions (Asian, London, NY)."""
        logger.info("Forex session manager starting")

        while self._running:
            try:
                from datetime import datetime, timezone
                now_utc = datetime.now(timezone.utc)
                hour = now_utc.hour

                # Determine active session
                if 22 <= hour or hour < 8:
                    session = "asian"
                elif 8 <= hour < 13:
                    session = "london"
                elif 13 <= hour < 17:
                    session = "newyork"
                else:
                    session = "overlap" if 13 <= hour < 17 else "late"

                # Write session info to shared memory
                session_data = {
                    "active_session": session,
                    "utc_hour": hour,
                    "london_open": 8 <= hour < 17,
                    "ny_open": 13 <= hour < 22,
                    "tokyo_open": 0 <= hour < 8,
                    "timestamp": now_utc.isoformat(),
                }
                try:
                    shm_path = "/dev/shm/forex_session"
                    with open(shm_path, "w") as f:
                        json.dump(session_data, f)
                except Exception:
                    pass

                logger.debug("Forex session: {} (UTC hour {})", session, hour)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Forex session loop error: {}", e)

            await asyncio.sleep(60)  # Check every minute

    async def _alert_dispatcher_loop(self) -> None:
        """Dispatch alerts to Telegram/Discord."""
        logger.info("Alert dispatcher starting")

        while self._running:
            try:
                # Check for pending alerts in Redis
                try:
                    import redis.asyncio as aioredis
                    redis_url = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")
                    r = aioredis.from_url(redis_url)
                    alert = await r.lpop("trading:alerts")
                    if alert:
                        await self._send_alert(json.loads(alert))
                    await r.aclose()
                except ImportError:
                    pass
                except Exception:
                    pass  # Redis not available, skip silently

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Alert dispatcher error: {}", e)

            await asyncio.sleep(5)

    async def _journal_sync_loop(self) -> None:
        """Sync trade journal from Rust binary segment_*.dat files to PnL engine.

        The Rust engine writes binary journal entries using memory-mapped,
        fixed-size ``#[repr(C, packed)]`` structs.  This loop reads them
        directly using ``struct.unpack``, matching the Rust layout exactly.

        Journal entry header (8 bytes):
            entry_type: u16, payload_size: u16, sequence: u32

        We only extract ENTRY_ORDER_RESULT (4) and ENTRY_TRADE (8) entries
        for PnL attribution.  All other entry types are skipped.
        """
        import struct as _struct
        import mmap as _mmap

        HEADER_SIZE = 8
        HEADER_FMT = "<HHI"  # entry_type(u16), payload_size(u16), sequence(u32)
        # JournalOrderResult payload (after 8-byte header): 84 bytes
        # timestamp_ns(Q) symbol_id(H) side(B) status(B) filled_size(q) avg_fill_price_fp(q) fee_fp(q) exchange_latency_us(Q) order_id(32s)
        ORDER_RESULT_FMT = "<Q HBB qqq Q 32s"
        ORDER_RESULT_TYPE = 4
        # JournalTrade payload (after 8-byte header): 52 bytes
        # timestamp_ns(Q) symbol_id(H) side(B) _pad(B) size(q) price_fp(q) fee_fp(q) is_maker(B) _reserved(15s)
        TRADE_ENTRY_FMT = "<Q HBB qqq B 15s"
        TRADE_ENTRY_TYPE = 8

        FP_PRECISION = 1e8
        FQ_PRECISION = 1e4

        logger.info("Trade journal sync starting (reading binary segment_*.dat files)")
        journal_dir = os.getenv("JOURNAL_DIR", "/dev/shm/trading_journal")

        # Track the last processed sequence to avoid re-reading
        last_processed_seq: int = 0

        while self._running:
            try:
                if not os.path.isdir(journal_dir):
                    await asyncio.sleep(10)
                    continue

                # Find all segment files sorted by index
                segment_files = sorted(
                    f for f in os.listdir(journal_dir)
                    if f.startswith("segment_") and f.endswith(".dat")
                )

                trades_ingested = 0
                for seg_file in segment_files:
                    filepath = os.path.join(journal_dir, seg_file)
                    try:
                        file_size = os.path.getsize(filepath)
                        if file_size < HEADER_SIZE:
                            continue

                        with open(filepath, "rb") as f:
                            with _mmap.mmap(f.fileno(), 0, access=_mmap.ACCESS_READ) as mm:
                                pos = 0
                                while pos + HEADER_SIZE <= file_size:
                                    hdr = _struct.unpack_from(HEADER_FMT, mm, pos)
                                    entry_type, payload_size, sequence = hdr

                                    if entry_type == 0:
                                        break  # Reached unused frontier

                                    entry_total = HEADER_SIZE + payload_size
                                    if pos + entry_total > file_size:
                                        break  # Partial entry

                                    # Skip already-processed entries
                                    if sequence <= last_processed_seq:
                                        pos += entry_total
                                        continue

                                    payload_offset = pos + HEADER_SIZE

                                    if entry_type == TRADE_ENTRY_TYPE and self._pnl_engine:
                                        try:
                                            vals = _struct.unpack_from(TRADE_ENTRY_FMT, mm, payload_offset)
                                            ts_ns, sym_id, side, _pad, size_fp, price_fp, fee_fp, is_maker, _res = vals
                                            from analytics.pnl_attribution import TradeRecord
                                            from datetime import datetime, timezone
                                            ts = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
                                            record = TradeRecord(
                                                trade_id=f"j-{sequence}",
                                                symbol=f"SYM_{sym_id}",
                                                strategy="rust_engine",
                                                direction="long" if side == 0 else "short",
                                                entry_price=price_fp / FP_PRECISION,
                                                exit_price=price_fp / FP_PRECISION,
                                                quantity=abs(size_fp) / FQ_PRECISION,
                                                entry_time=ts,
                                                exit_time=ts,
                                                realized_pnl=0.0,
                                                fees_paid=fee_fp / FP_PRECISION,
                                                slippage_cost=0.0,
                                                signal_confidence=0.5,
                                                regime_at_entry="unknown",
                                                leverage=1,
                                            )
                                            await self._pnl_engine.record_trade(record)
                                            trades_ingested += 1
                                        except Exception as e:
                                            logger.debug("Journal trade parse error seq={}: {}", sequence, e)

                                    elif entry_type == ORDER_RESULT_TYPE and self._pnl_engine:
                                        try:
                                            vals = _struct.unpack_from(ORDER_RESULT_FMT, mm, payload_offset)
                                            ts_ns, sym_id, side, status, filled_size, avg_price_fp, fee_fp, lat_us, oid_bytes = vals
                                            # status: 1=filled — only record fills
                                            if status == 1:
                                                from analytics.pnl_attribution import TradeRecord
                                                from datetime import datetime, timezone
                                                ts = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc)
                                                record = TradeRecord(
                                                    trade_id=f"o-{sequence}",
                                                    symbol=f"SYM_{sym_id}",
                                                    strategy="rust_engine",
                                                    direction="long" if side == 0 else "short",
                                                    entry_price=avg_price_fp / FP_PRECISION,
                                                    exit_price=avg_price_fp / FP_PRECISION,
                                                    quantity=abs(filled_size) / FQ_PRECISION,
                                                    entry_time=ts,
                                                    exit_time=ts,
                                                    realized_pnl=0.0,
                                                    fees_paid=fee_fp / FP_PRECISION,
                                                    slippage_cost=0.0,
                                                    signal_confidence=0.5,
                                                    regime_at_entry="unknown",
                                                    leverage=1,
                                                )
                                                await self._pnl_engine.record_trade(record)
                                                trades_ingested += 1
                                        except Exception as e:
                                            logger.debug("Journal order result parse error seq={}: {}", sequence, e)

                                    last_processed_seq = max(last_processed_seq, sequence)
                                    pos += entry_total

                    except Exception as e:
                        logger.debug("Journal segment read error {}: {}", seg_file, e)

                if trades_ingested > 0:
                    logger.info("Journal sync: ingested {} trade entries (last_seq={})", trades_ingested, last_processed_seq)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Journal sync error: {}", e)

            await asyncio.sleep(10)

    async def _start_alpha_oracle(self) -> None:
        """Initialize the Alpha Oracle confluence engine and SHM signal queue."""
        try:
            from core.alpha_oracle import AlphaOracle
            self._alpha_oracle = AlphaOracle(
                min_confluence_pct=0.75,
                min_risk_reward=2.0,
                min_confidence=0.6,
                cooldown_seconds=300.0,
            )
            self._alpha_oracle.start()
            logger.info("Alpha Oracle initialized — ready to emit signals to SHM queue")
        except Exception as e:
            logger.error("Failed to start Alpha Oracle: {}", e)
            self._alpha_oracle = None

    async def _init_ai_brain(self) -> None:
        """Initialize the AI Brain for LLM-powered trade analysis.

        When at least one LLM API key is configured and AI is enabled,
        the brain is used by the Alpha Oracle to validate strategy signals
        against news, risk/reward, and market regime before emission.
        """
        try:
            use_ai = os.getenv("USE_AI", "on").lower()
            if use_ai in ("off", "false", "no", "0"):
                logger.info("AI Brain disabled (USE_AI=off)")
                return

            # Collect API keys
            gaterouter_key = os.getenv("GATEROUTER_API_KEY", "")
            openai_key = os.getenv("OPENAI_API_KEY", "")
            anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
            gemini_key = os.getenv("GEMINI_API_KEY", "")
            grok_key = os.getenv("GROK_API_KEY", "")
            openrouter_key = os.getenv("OPENROUTER_API_KEY", "")

            any_key = any([gaterouter_key, openai_key, anthropic_key, gemini_key, grok_key, openrouter_key])
            if not any_key:
                logger.info("AI Brain: no LLM API keys configured — running strategy-only mode")
                return

            from ai.llm_client import LLMClient
            from ai.brain import AIBrain

            llm_client = LLMClient(
                gaterouter_api_key=gaterouter_key,
                openai_api_key=openai_key,
                anthropic_api_key=anthropic_key,
                gemini_api_key=gemini_key,
                grok_api_key=grok_key,
                openrouter_api_key=openrouter_key,
                gaterouter_model=os.getenv("GATEROUTER_MODEL", "deepseek/deepseek-chat"),
                openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                gemini_flash_model=os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash"),
                grok_model=os.getenv("GROK_MODEL", "grok-3-mini"),
                openrouter_model=os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free"),
                use_local_first=False,
            )
            self._ai_brain = AIBrain(llm_client=llm_client)

            active = [n for n, k in [
                ("GateRouter", gaterouter_key), ("Gemini", gemini_key),
                ("Grok", grok_key), ("OpenRouter", openrouter_key),
                ("OpenAI", openai_key), ("Anthropic", anthropic_key),
            ] if k]
            logger.info(
                "AI Brain initialized for Alpha Oracle (providers: {})",
                " → ".join(active),
            )
        except Exception as e:
            logger.warning("Failed to init AI Brain: {} — continuing without AI", e)
            self._ai_brain = None

    def _get_trading_pairs(self) -> list:
        """Read trading pairs from environment or use defaults."""
        raw = os.getenv("TRADING_PAIRS", "BTC/USDT,ETH/USDT,SOL/USDT")
        pairs = []
        for p in raw.split(","):
            s = p.strip()
            if not s:
                continue
            # Normalize: BTC/USDT:USDT → BTC_USDT
            base = s.split(":")[0]
            pairs.append(base.replace("/", "_").upper())
        return pairs or ["BTC_USDT", "ETH_USDT", "SOL_USDT"]

    async def _ai_validate_signal(
        self, symbol: str, signals: list, ohlcv
    ) -> Optional[Dict[str, Any]]:
        """Consult the AI brain to validate/enrich strategy signals.

        Returns the AI TradeDecision dict if AI approves (should_enter=True),
        or None if AI vetoes or is unavailable.  The decision's confidence
        and reasoning are used to enrich the final trade intent.
        """
        if self._ai_brain is None:
            return None  # No AI → pass through to confluence engine only

        import pandas as pd
        from core.alpha_oracle import SignalSide

        try:
            current_price = float(ohlcv["close"].iloc[-1]) if ohlcv is not None and not ohlcv.empty else 0.0

            # Build strategy summary for AI context
            long_sigs = [s for s in signals if s.side == SignalSide.LONG]
            short_sigs = [s for s in signals if s.side == SignalSide.SHORT]
            strategy_summary = (
                f"{len(signals)} strategies evaluated: "
                f"{len(long_sigs)} LONG, {len(short_sigs)} SHORT. "
            )
            # Add top-3 strategy names by confidence
            top_signals = sorted(signals, key=lambda s: s.confidence, reverse=True)[:3]
            for sig in top_signals:
                strategy_summary += f"\n  - {sig.strategy_name}: {sig.side.name} conf={sig.confidence:.2f}"

            # Build indicators from OHLCV
            indicators = {}
            if ohlcv is not None and len(ohlcv) >= 14:
                closes = ohlcv["close"].tolist()
                # Simple RSI
                from strategy.base_strategy import BaseStrategy
                indicators["rsi"] = BaseStrategy._calculate_rsi(closes)
                macd = BaseStrategy._calculate_macd(closes)
                indicators.update(macd)

            market_data = {
                "price": current_price,
                "indicators": indicators,
                "ohlcv": ohlcv,
            }
            context = {
                "news": [],  # TODO: wire in live news feed
                "balance": float(os.getenv("PAPER_TRADING_BALANCE", "10000")),
                "open_positions": [],
                "strategy_summary": strategy_summary,
            }

            decision = await self._ai_brain.analyze(symbol, market_data, context)

            if decision.should_enter:
                logger.info(
                    "AI Brain APPROVES {} {}: conf={:.0%}, {}",
                    symbol, decision.direction.value,
                    decision.confidence, decision.reasoning[:80],
                )
                return {
                    "direction": decision.direction.value,
                    "confidence": decision.confidence,
                    "leverage": decision.suggested_leverage,
                    "sl_pct": decision.suggested_stop_loss_pct,
                    "tp_pct": decision.suggested_take_profit_pct,
                    "reasoning": decision.reasoning,
                }
            else:
                logger.debug(
                    "AI Brain VETOES {} signal: {}",
                    symbol, decision.reasoning[:80],
                )
                return None

        except Exception as e:
            logger.warning("AI Brain validation error for {}: {}", symbol, e)
            return None  # On error, let confluence engine decide alone

    async def _alpha_oracle_loop(self) -> None:
        """Main Alpha Oracle evaluation loop.

        Every 60 seconds (aligned with 1m candle close), evaluate all configured
        trading pairs through the strategy ensemble and confluence engine.
        When a golden setup is found, consult the AI brain for news/risk analysis,
        then emit a TradeIntent to the SHM signal queue for Rust to execute.
        """
        from core.alpha_oracle import StrategySignal, SignalSide

        logger.info("Alpha Oracle loop starting (60s evaluation cycle)")

        trading_pairs = self._get_trading_pairs()
        logger.info("Alpha Oracle trading pairs: {}", trading_pairs)

        while self._running:
            try:
                if self._alpha_oracle is None:
                    await asyncio.sleep(60)
                    continue

                for symbol in trading_pairs:
                    try:
                        # 1. Evaluate 60+ strategies against live OHLCV data
                        signals = await self._evaluate_strategies_for_symbol(symbol)

                        if not signals:
                            continue

                        # 2. Consult AI brain (news + risk/reward + strategy review)
                        ohlcv = await self._fetch_ohlcv_gateio(symbol, interval="1m", limit=200)
                        ai_decision = await self._ai_validate_signal(symbol, signals, ohlcv)

                        # 3. If AI is available and vetoes, skip this symbol
                        if self._ai_brain is not None and ai_decision is None:
                            logger.debug(
                                "Alpha Oracle: AI vetoed {} — skipping emission",
                                symbol,
                            )
                            continue

                        # 4. If AI approved, use its leverage suggestion
                        leverage = ai_decision["leverage"] if ai_decision else 10

                        # 5. Run confluence engine and emit to SHM queue
                        intent = self._alpha_oracle.evaluate_and_emit(
                            signals,
                            leverage=leverage,
                            max_contracts=5,
                        )
                        if intent:
                            logger.info(
                                "Alpha Oracle: signal emitted for {} — "
                                "AI={}, queue depth: {}",
                                symbol,
                                "approved" if ai_decision else "N/A",
                                self._alpha_oracle.queue_depth,
                            )
                    except Exception as e:
                        logger.debug("Alpha Oracle: eval error for {}: {}", symbol, e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Alpha Oracle loop error: {}", e)

            # Sleep until next 1m candle close (approximately)
            await asyncio.sleep(60)

        # Cleanup
        if self._alpha_oracle:
            self._alpha_oracle.stop()

    # ── Strategy ensemble (lazily initialized) ───────────────────────────

    _strategy_ensemble: list = []
    _strategies_initialized: bool = False

    def _init_strategy_ensemble(self) -> list:
        """Lazily create instances of all available strategy classes.

        Each strategy's ``analyze(ohlcv, symbol)`` method is used by the
        Alpha Oracle to collect directional opinions.  Strategies that fail
        to import are silently skipped so one broken module cannot block the
        entire ensemble.
        """
        if self._strategies_initialized:
            return self._strategy_ensemble

        # (class_path, kwargs_override) — symbols=[] means "all symbols"
        _STRATEGY_CLASSES = [
            ("strategy.strategies.rsi_divergence", "RSIDivergenceStrategy", {}),
            ("strategy.strategies.macd_crossover", "MACDCrossoverStrategy", {}),
            ("strategy.strategies.bollinger_squeeze", "BollingerSqueezeStrategy", {}),
            ("strategy.strategies.ema_ribbon", "EMARibbonStrategy", {}),
            ("strategy.strategies.donchian_breakout", "DonchianBreakoutStrategy", {}),
            ("strategy.strategies.ichimoku_cloud", "IchimokuCloudStrategy", {}),
            ("strategy.strategies.fibonacci_retracement", "FibonacciRetracementStrategy", {}),
            ("strategy.strategies.adx_trend", "ADXTrendStrategy", {}),
            ("strategy.strategies.harmonic_pattern", "HarmonicPatternStrategy", {}),
            ("strategy.strategies.elliott_wave", "ElliottWaveStrategy", {}),
            ("strategy.strategies.fair_value_gap", "FairValueGapStrategy", {}),
            ("strategy.strategies.accumulation_distribution", "AccumulationDistributionStrategy", {}),
            ("strategy.strategies.correlation_divergence", "CorrelationDivergenceStrategy", {}),
            ("strategy.strategies.momentum_strategy", "MomentumStrategy", {}),
            ("strategy.strategies.mean_reversion_strategy", "MeanReversionStrategy", {}),
            ("strategy.strategies.trend_following_strategy", "TrendFollowingStrategy", {}),
            ("strategy.strategies.funding_rate_arb", "FundingRateArbStrategy", {}),
            ("strategy.strategies.grid_trading", "GridTradingStrategy", {}),
            ("strategy.strategies.dca_strategy", "DCAStrategy", {}),
            ("strategy.strategies.heikin_ashi", "HeikinAshiStrategy", {}),
            ("strategy.strategies.entropy_strategy", "EntropyStrategy", {}),
            ("strategy.strategies.hurst_exponent", "HurstExponentStrategy", {}),
            ("strategy.strategies.bayesian_strategy", "BayesianStrategy", {}),
            ("strategy.strategies.delta_divergence", "DeltaDivergenceStrategy", {}),
            ("strategy.strategies.dollar_bars", "DollarBarsStrategy", {}),
            ("strategy.strategies.auction_theory", "AuctionTheoryStrategy", {}),
            ("strategy.strategies.amihud_illiquidity", "AmihudIlliquidityStrategy", {}),
            ("strategy.strategies.cross_sectional_momentum", "CrossSectionalMomentumStrategy", {}),
            ("strategy.strategies.dbscan_breakout", "DBSCANBreakoutStrategy", {}),
            ("strategy.strategies.fear_greed_contrarian", "FearGreedContrarianStrategy", {}),
            ("strategy.strategies.funding_momentum", "FundingMomentumStrategy", {}),
            ("strategy.strategies.gamma_scalping", "GammaScalpingStrategy", {}),
            ("strategy.strategies.genetic_optimizer", "GeneticOptimizerStrategy", {}),
            ("strategy.strategies.footprint_chart", "FootprintChartStrategy", {}),
            ("strategy.strategies.cointegration_pairs", "CointegrationPairsStrategy", {}),
            ("strategy.strategies.copula_dependence", "CopulaDependenceStrategy", {}),
            ("strategy.strategies.dispersion_trading", "DispersionTradingStrategy", {}),
            ("strategy.strategies.defi_tvl_flow", "DefiTvlFlowStrategy", {}),
            ("strategy.strategies.cross_exchange_arb", "CrossExchangeArbStrategy", {}),
        ]

        ensemble = []
        for module_path, class_name, kwargs in _STRATEGY_CLASSES:
            try:
                import importlib
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
                instance = cls(symbols=[], **kwargs)
                ensemble.append(instance)
            except Exception as e:
                logger.debug("Strategy {} skipped: {}", class_name, e)

        self._strategy_ensemble = ensemble
        self._strategies_initialized = True
        logger.info(
            "Strategy ensemble initialized: {}/{} strategies loaded",
            len(ensemble), len(_STRATEGY_CLASSES),
        )
        return ensemble

    # ── OHLCV fetcher (Gate.io REST API) ──────────────────────────────

    _ohlcv_cache: Dict[str, Any] = {}
    _OHLCV_CACHE_TTL: float = 55.0  # seconds (just under the 60s eval cycle)

    async def _fetch_ohlcv_gateio(
        self, symbol: str, interval: str = "1m", limit: int = 200
    ) -> Optional["pd.DataFrame"]:
        """Fetch OHLCV candlestick data from Gate.io public REST API.

        Returns a pandas DataFrame with columns [open, high, low, close, volume]
        or None if the fetch fails.  Results are cached for ~55 seconds to avoid
        redundant API calls within the same evaluation cycle.
        """
        import pandas as pd

        cache_key = f"{symbol}:{interval}:{limit}"
        now = time.time()
        cached = self._ohlcv_cache.get(cache_key)
        if cached and (now - cached[0]) < self._OHLCV_CACHE_TTL:
            return cached[1]

        # Normalize symbol: BTC_USDT (Gate.io futures format)
        contract = symbol.replace("/", "_").replace(":", "").upper()

        url = (
            f"https://api.gateio.ws/api/v4/futures/usdt/candlesticks"
            f"?contract={contract}&interval={interval}&limit={limit}"
        )
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning("Gate.io OHLCV fetch HTTP {}: {}", resp.status, symbol)
                        return None
                    data = await resp.json()

            if not data or not isinstance(data, list):
                return None

            rows = []
            for c in data:
                rows.append({
                    "open": float(c.get("o", 0)),
                    "high": float(c.get("h", 0)),
                    "low": float(c.get("l", 0)),
                    "close": float(c.get("c", 0)),
                    "volume": float(c.get("v", 0) or c.get("sum", 0)),
                })

            df = pd.DataFrame(rows)
            if df.empty:
                return None

            self._ohlcv_cache[cache_key] = (now, df)
            return df

        except Exception as e:
            logger.warning("Gate.io OHLCV fetch error for {}: {}", symbol, e)
            return None

    # ── Strategy evaluation (production implementation) ────────────────

    async def _evaluate_strategies_for_symbol(
        self, symbol: str
    ) -> list:
        """Evaluate the full strategy ensemble for a single symbol.

        Fetches OHLCV data from Gate.io, runs every strategy's ``analyze()``
        method, and converts actionable results into ``StrategySignal`` objects
        for the confluence engine.
        """
        from core.alpha_oracle import StrategySignal, SignalSide
        import pandas as pd

        signals = []

        # Fetch OHLCV data (1m candles, 200 bars ≈ 3.3 hours)
        ohlcv = await self._fetch_ohlcv_gateio(symbol, interval="1m", limit=200)
        if ohlcv is None or ohlcv.empty or len(ohlcv) < 50:
            return signals

        current_price = float(ohlcv["close"].iloc[-1])
        if current_price <= 0:
            return signals

        # Initialize strategies if needed
        ensemble = self._init_strategy_ensemble()

        # Run each strategy's analyze() method
        for strategy in ensemble:
            try:
                result = strategy.analyze(ohlcv, symbol)
                if result is None:
                    continue

                direction = result.get("direction", "neutral")
                if direction == "neutral":
                    continue

                confidence = float(result.get("confidence", 0.0))
                if confidence < 0.3:
                    continue

                entry_price = float(result.get("entry_price", current_price))
                atr = float(result.get("atr", 0.0))

                # Compute SL/TP from strategy result or derive from ATR
                if result.get("stop_loss") and result.get("take_profit"):
                    sl = float(result["stop_loss"])
                    tp = float(result["take_profit"])
                else:
                    # Default ATR-based SL/TP
                    atr_val = atr if atr > 0 else entry_price * 0.01
                    if direction == "long":
                        sl = entry_price - atr_val * 2.0
                        tp = entry_price + atr_val * 3.0
                    else:
                        sl = entry_price + atr_val * 2.0
                        tp = entry_price - atr_val * 3.0

                side = SignalSide.LONG if direction == "long" else SignalSide.SHORT

                signals.append(StrategySignal(
                    strategy_name=result.get("strategy", strategy.name if hasattr(strategy, 'name') else "unknown"),
                    symbol=symbol,
                    side=side,
                    confidence=confidence,
                    entry_price=entry_price,
                    stop_loss=sl,
                    take_profit=tp,
                    timeframe=result.get("timeframe", "1m"),
                ))

            except Exception as e:
                logger.debug(
                    "Strategy {} error for {}: {}",
                    getattr(strategy, 'name', strategy.__class__.__name__),
                    symbol, e,
                )

        if signals:
            long_count = sum(1 for s in signals if s.side == SignalSide.LONG)
            short_count = len(signals) - long_count
            logger.info(
                "Alpha Oracle: {} strategies fired for {} (long={}, short={})",
                len(signals), symbol, long_count, short_count,
            )

        return signals

    def _write_regime_to_shm(self, regime: Any, shm_path: str) -> None:
        """Write regime weights to shared memory for Rust engine."""
        try:
            import struct
            if hasattr(regime, "weights"):
                weights = regime.weights
            elif isinstance(regime, dict):
                weights = regime
            else:
                weights = {"trending": 0.25, "ranging": 0.25, "volatile": 0.25, "crash": 0.25}

            # Pack as [trending, ranging, volatile, crash] f64s
            data = struct.pack(
                "dddd",
                float(weights.get("trending", 0.25)),
                float(weights.get("ranging", 0.25)),
                float(weights.get("volatile", 0.25)),
                float(weights.get("crash", 0.25)),
            )
            with open(shm_path, "wb") as f:
                f.write(data)
        except Exception as e:
            logger.error("Failed to write regime to SHM: {}", e)

    async def _send_alert(self, alert_data: Dict[str, Any]) -> None:
        """Send an alert via configured channels."""
        message = alert_data.get("message", "Unknown alert")
        severity = alert_data.get("severity", "info")
        logger.info("Alert [{}]: {}", severity, message)

        # Telegram
        bot_token = os.getenv("ALERT_TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("ALERT_TELEGRAM_CHAT_ID")
        if bot_token and chat_id:
            try:
                import aiohttp
                url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                async with aiohttp.ClientSession() as session:
                    await session.post(url, json={
                        "chat_id": chat_id,
                        "text": f"🤖 [{severity.upper()}] {message}",
                        "parse_mode": "HTML",
                    })
            except Exception as e:
                logger.error("Telegram alert failed: {}", e)

    def _on_degradation_change(self, old_level, new_level, snapshot) -> None:
        """Callback when degradation level changes."""
        logger.warning(
            "Degradation: {} → {} (CPU={:.1f}% MEM={:.1f}%)",
            old_level.name, new_level.name,
            snapshot.cpu_percent, snapshot.memory_percent,
        )


async def main():
    """Entry point for the cold path orchestrator."""
    orchestrator = ColdPathOrchestrator()
    await orchestrator.start()


if __name__ == "__main__":
    asyncio.run(main())
