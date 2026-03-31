"""Master configuration using pydantic-settings."""

from __future__ import annotations

import functools
import os
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ExchangeConfig(BaseModel):
    """Exchange-level trading configuration."""

    primary_exchange: str = "gateio"
    trading_pairs: List[str] = Field(
        default=[
            "BTC/USDT",
            "ETH/USDT",
            "SOL/USDT",
            "BNB/USDT",
            "XRP/USDT",
        ]
    )
    default_leverage: int = 5
    max_leverage: int = 20
    order_type: str = "limit"
    use_testnet: bool = False
    fee_rate: float = 0.0002

    @field_validator("order_type")
    @classmethod
    def validate_order_type(cls, v: str) -> str:
        allowed = {"limit", "market"}
        if v not in allowed:
            raise ValueError(f"order_type must be one of {allowed}")
        return v

    @field_validator("default_leverage", "max_leverage")
    @classmethod
    def validate_leverage(cls, v: int) -> int:
        if v < 1 or v > 125:
            raise ValueError("Leverage must be between 1 and 125")
        return v


class AIConfig(BaseModel):
    """AI / LLM configuration."""

    enabled: bool = True  # Master toggle for AI features (analysis, news, sentiment)
    primary_llm: str = "auto"  # "auto" = free-first fallback chain
    openai_model: str = "gpt-4o"
    anthropic_model: str = "claude-3-5-sonnet-20241022"
    gemini_flash_model: str = "gemini-2.5-flash"
    gemini_flash_lite_model: str = "gemini-2.5-flash-lite"
    grok_model: str = "grok-3-mini"
    openrouter_model: str = "mistralai/mistral-7b-instruct:free"
    gaterouter_model: str = "deepseek/deepseek-chat"
    ollama_model: str = "llama3:8b"
    temperature: float = 0.3
    max_tokens: int = 4096
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    use_local_models: bool = True
    fallback_to_cloud: bool = True

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v


class RiskConfig(BaseModel):
    """Risk management configuration."""

    max_position_size_pct: float = 10.0
    max_open_positions: int = 10
    max_daily_loss_pct: float = 2.0
    daily_profit_target_pct: float = 1.0
    max_drawdown_pct: float = 10.0
    default_stop_loss_pct: float = 2.0
    default_take_profit_pct: float = 3.0
    trailing_stop_pct: float = 1.5
    risk_reward_min: float = 1.06
    use_kelly_criterion: bool = True
    max_correlation: float = 0.7
    circuit_breaker_loss_pct: float = 5.0
    cooldown_after_loss_minutes: int = 30

    # ── Advanced position management ────────────────────────────────────
    enable_trailing_tp: bool = True
    trailing_tp_distance_pct: float = 0.5  # 0.5% trailing distance
    enable_break_even_sl: bool = True
    break_even_buffer_pct: float = 0.1  # 0.1% buffer above entry
    position_sizing_method: str = "volatility_adjusted"  # fixed | kelly | volatility_adjusted
    max_correlation_risk: float = 0.7  # reduce new positions if correlation > 0.7
    max_funding_rate_tolerance: float = -0.05  # close if funding rate worse than -0.05%
    auto_close_orphaned_positions: bool = False  # auto-close positions with no DB record

    # ── Liquidation proximity thresholds ────────────────────────────────
    liquidation_warning_pct: float = 5.0   # Alert when within 5% of liquidation
    liquidation_emergency_pct: float = 2.0  # Auto-close when within 2% of liquidation

    # ── Per-symbol cooldown after close ─────────────────────────────────
    symbol_cooldown_seconds: int = 60  # Seconds before re-opening on same symbol

    # ── Profit maximization settings ────────────────────────────────────
    enable_dynamic_leverage: bool = True  # Scale leverage based on performance
    leverage_scale_on_wins: bool = True  # Increase leverage after consecutive wins
    min_dynamic_leverage: int = 3  # Minimum leverage when scaling
    max_dynamic_leverage: int = 15  # Maximum leverage when scaling
    wins_for_leverage_increase: int = 3  # Consecutive wins to increase leverage
    compound_profits: bool = True  # Reinvest profits into position sizing

    # ── Margin mode ──────────────────────────────────────────────────────
    margin_mode: str = "isolated"  # "isolated" or "cross"

    # ── Maker entry orders (Task 4) ───────────────────────────────────────
    use_maker_entries: bool = False  # Enable post-only maker entries to save fees
    maker_entry_max_wait_seconds: float = 15.0

    # ── Slippage protection (Task 5) ─────────────────────────────────────
    max_entry_slippage_pct: float = 0.01  # 1% max slippage on entry orders
    max_exit_slippage_pct: float = 0.02   # 2% max slippage on exit orders (more lenient)

    # ── Economic calendar filter (Task 7) ─────────────────────────────────
    enable_economic_filter: bool = True
    economic_event_buffer_minutes: int = 15

    # ── Liquidation margin top-up (Task 10) ───────────────────────────────
    liquidation_margin_topup_pct: float = 3.0   # Auto top-up margin when within 3% of liquidation
    max_margin_topup_usdt: float = 50.0          # Maximum USDT to add per top-up

    # ── Contract size floor (Gate.io / contract markets) ─────────────────
    min_contract_override: int = 1  # Minimum contracts per trade (Gate.io minimum is 1)

    # ── Maximum daily trade count ──────────────────────────────────────────
    max_daily_trades: int = 20  # Maximum number of trades per day (prevents overtrading)

    @field_validator("margin_mode")
    @classmethod
    def validate_margin_mode(cls, v: str) -> str:
        if v not in {"isolated", "cross"}:
            raise ValueError("margin_mode must be 'isolated' or 'cross'")
        return v

    @field_validator("max_position_size_pct", "max_daily_loss_pct", "max_drawdown_pct")
    @classmethod
    def validate_percentage(cls, v: float) -> float:
        if not 0.0 < v <= 100.0:
            raise ValueError("Percentage must be between 0 and 100")
        return v


class DataSourceConfig(BaseModel):
    """Social & news data source configuration."""

    twitter_accounts: List[str] = Field(
        default=[
            "elonmusk",
            "cz_binance",
            "VitalikButerin",
            "Brian_Armstrong",
            "aantonop",
            "CryptoKaleo",
            "PlanB",
            "APompliano",
            "RaoulGMI",
            "woonomic",
            "100trillionUSD",
            "CryptoCobain",
            "Rager",
            "CryptoWendyO",
            "scottmelker",
        ]
    )
    telegram_channels: List[str] = Field(
        default=[
            "CryptoSignals",
            "WhaleAlert",
            "CryptoCompass",
            "BitcoinNews",
            "DeFiAlpha",
            "CryptoInsiders",
            "AltcoinAlerts",
            "BlockchainNews",
        ]
    )
    reddit_subreddits: List[str] = Field(
        default=[
            "CryptoCurrency",
            "Bitcoin",
            "ethereum",
            "CryptoMarkets",
            "altcoin",
            "defi",
            "SatoshiStreetBets",
        ]
    )
    rss_feeds: List[str] = Field(
        default=[
            "https://cointelegraph.com/rss",
            "https://coindesk.com/arc/outboundfeeds/rss/",
            "https://decrypt.co/feed",
            "https://theblock.co/rss.xml",
            "https://cryptoslate.com/feed/",
            "https://bitcoinmagazine.com/feed",
        ]
    )
    polling_interval_seconds: int = 30
    max_news_age_minutes: int = 60


class MonitoringConfig(BaseModel):
    """Monitoring and alerting configuration."""

    dashboard_port: int = int(os.environ.get("DASHBOARD_PORT", "8081"))
    enable_telegram_alerts: bool = True
    enable_discord_alerts: bool = True
    enable_email_alerts: bool = False
    alert_on_trade: bool = True
    alert_on_error: bool = True
    alert_on_daily_summary: bool = True
    dashboard_ws_interval: float = 2.0  # WebSocket push interval in seconds


class TradingLoopConfig(BaseModel):
    """Trading loop timing configuration."""

    fast_loop_interval: int = 5   # seconds — position monitoring, trailing stops, SL/TP
    slow_loop_interval: int = 30  # seconds — signal generation, strategy evaluation


class ForexConfig(BaseModel):
    """Forex trading configuration (Gate.io TradFi — XAU/USD, XAG/USD)."""

    primary_exchange: str = "gateio"
    trading_pairs: List[str] = Field(default=["XAU/USD", "XAG/USD"])
    default_leverage: int = 20
    max_leverage: int = 100
    risk_per_trade_pct: float = 1.0   # Percentage of equity to risk per trade
    max_open_trades: int = 5
    daily_loss_limit_pct: float = 3.0

    @field_validator("default_leverage", "max_leverage")
    @classmethod
    def validate_leverage(cls, v: int) -> int:
        if v < 1 or v > 500:
            raise ValueError("Leverage must be between 1 and 500")
        return v


class ExnessForexConfig(BaseModel):
    """Exness forex trading configuration (12 pairs + session features)."""

    trading_pairs: List[str] = Field(
        default=[
            "XAUUSD",
            "XAGUSD",
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "AUDUSD",
            "USDCAD",
            "USDCHF",
            "GBPJPY",
            "EURJPY",
            "NZDUSD",
            "EURGBP",
        ]
    )
    default_leverage: int = 100  # Exness supports up to 1:2000, default 1:100
    max_leverage: int = 2000
    risk_per_trade_pct: float = 1.0
    max_open_trades: int = 8
    daily_loss_limit_pct: float = 3.0
    session_filter_enabled: bool = True  # Enable session-aware strategy routing
    swap_free_account: bool = False      # Islamic account with no swap charges
    news_event_pause_minutes: int = 15   # Pause trading N minutes before high-impact news
    max_spread_multiplier: float = 2.0   # Don't trade when spread > typical * multiplier

    @field_validator("default_leverage", "max_leverage")
    @classmethod
    def validate_leverage(cls, v: int) -> int:
        if v < 1 or v > 2000:
            raise ValueError("Leverage must be between 1 and 2000")
        return v


class Settings(BaseSettings):
    """Master application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_nested_delimiter="__",
    )

    # ── Exchange API credentials ──────────────────────────────────────────────
    mexc_api_key: Optional[str] = Field(default=None, alias="MEXC_API_KEY")
    mexc_secret_key: Optional[str] = Field(default=None, alias="MEXC_SECRET_KEY")

    gateio_api_key: Optional[str] = Field(default=None, alias="GATEIO_API_KEY")
    gateio_secret_key: Optional[str] = Field(default=None, alias="GATEIO_SECRET_KEY")

    bingx_api_key: Optional[str] = Field(default=None, alias="BINGX_API_KEY")
    bingx_secret_key: Optional[str] = Field(default=None, alias="BINGX_SECRET_KEY")

    bitget_api_key: Optional[str] = Field(default=None, alias="BITGET_API_KEY")
    bitget_secret_key: Optional[str] = Field(default=None, alias="BITGET_SECRET_KEY")
    bitget_passphrase: Optional[str] = Field(default=None, alias="BITGET_PASSPHRASE")


    # ── Exness credentials (for forex trading via MT5) ────────────────────────
    exness_login: Optional[str] = Field(default=None, alias="EXNESS_LOGIN")
    exness_password: Optional[str] = Field(default=None, alias="EXNESS_PASSWORD")
    exness_server: str = Field(default="Exness-MT5Real", alias="EXNESS_SERVER")
    exness_account_type: str = Field(default="Raw Spread", alias="EXNESS_ACCOUNT_TYPE")

    exness_demo_login: Optional[str] = Field(default=None, alias="EXNESS_DEMO_LOGIN")
    exness_demo_password: Optional[str] = Field(default=None, alias="EXNESS_DEMO_PASSWORD")
    exness_demo_server: str = Field(default="Exness-MT5Demo", alias="EXNESS_DEMO_SERVER")
    exness_demo_account_type: str = Field(default="Raw Spread", alias="EXNESS_DEMO_ACCOUNT_TYPE")

    # ── Gate.io TradFi MT5 credentials (for forex/precious metals trading) ────
    gateio_tradfi_login: Optional[str] = Field(default=None, alias="GATEIO_TRADFI_LOGIN")
    gateio_tradfi_password: Optional[str] = Field(default=None, alias="GATEIO_TRADFI_PASSWORD")
    gateio_tradfi_server: str = Field(default="GateIO-TradFi", alias="GATEIO_TRADFI_SERVER")
    gateio_tradfi_account_type: str = Field(default="Standard", alias="GATEIO_TRADFI_ACCOUNT_TYPE")

    gateio_tradfi_demo_login: Optional[str] = Field(default=None, alias="GATEIO_TRADFI_DEMO_LOGIN")
    gateio_tradfi_demo_password: Optional[str] = Field(default=None, alias="GATEIO_TRADFI_DEMO_PASSWORD")
    gateio_tradfi_demo_server: str = Field(default="GateIO-TradFi-Demo", alias="GATEIO_TRADFI_DEMO_SERVER")
    gateio_tradfi_demo_account_type: str = Field(default="Standard", alias="GATEIO_TRADFI_DEMO_ACCOUNT_TYPE")

    # ── AI / LLM credentials ─────────────────────────────────────────────────
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    grok_api_key: Optional[str] = Field(default=None, alias="GROK_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    gaterouter_api_key: Optional[str] = Field(default=None, alias="GATEROUTER_API_KEY")

    # ── Social / data credentials ─────────────────────────────────────────────
    twitter_bearer_token: Optional[str] = Field(default=None, alias="TWITTER_BEARER_TOKEN")
    twitter_api_key: Optional[str] = Field(default=None, alias="TWITTER_API_KEY")
    twitter_api_secret: Optional[str] = Field(default=None, alias="TWITTER_API_SECRET")
    twitter_access_token: Optional[str] = Field(default=None, alias="TWITTER_ACCESS_TOKEN")
    twitter_access_secret: Optional[str] = Field(default=None, alias="TWITTER_ACCESS_SECRET")

    telegram_bot_token: Optional[str] = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: Optional[str] = Field(default=None, alias="TELEGRAM_CHAT_ID")

    reddit_client_id: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_ID")
    reddit_client_secret: Optional[str] = Field(default=None, alias="REDDIT_CLIENT_SECRET")
    reddit_user_agent: str = Field(default="crypto-trading-bot/1.0", alias="REDDIT_USER_AGENT")

    discord_webhook_url: Optional[str] = Field(default=None, alias="DISCORD_WEBHOOK_URL")
    discord_bot_token: Optional[str] = Field(default=None, alias="DISCORD_BOT_TOKEN")

    cryptopanic_api_key: Optional[str] = Field(default=None, alias="CRYPTOPANIC_API_KEY")

    # ── Alert destinations ────────────────────────────────────────────────────
    # These ALERT_TELEGRAM_* fields are the intended credentials for the alerting
    # bot (as documented in .env.example).  They take precedence over the
    # TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID fields, which are reserved for the
    # data-source Telegram reader bot.
    alert_telegram_bot_token: Optional[str] = Field(default=None, alias="ALERT_TELEGRAM_BOT_TOKEN")
    alert_telegram_chat_id: Optional[str] = Field(default=None, alias="ALERT_TELEGRAM_CHAT_ID")
    alert_email_from: Optional[str] = Field(default=None, alias="ALERT_EMAIL_FROM")
    alert_email_to: Optional[str] = Field(default=None, alias="ALERT_EMAIL_TO")
    alert_email_smtp_host: Optional[str] = Field(default=None, alias="ALERT_EMAIL_SMTP_HOST")
    alert_email_smtp_port: int = Field(default=587, alias="ALERT_EMAIL_SMTP_PORT")
    alert_email_smtp_user: Optional[str] = Field(default=None, alias="ALERT_EMAIL_SMTP_USER")
    alert_email_smtp_password: Optional[str] = Field(
        default=None, alias="ALERT_EMAIL_SMTP_PASSWORD"
    )

    # ── Infrastructure ────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/trading_bot",
        alias="DATABASE_URL",
    )

    # ── Security ──────────────────────────────────────────────────────────────
    encryption_key: Optional[str] = Field(default=None, alias="ENCRYPTION_KEY")
    secret_key: str = Field(default="change-me-in-production", alias="SECRET_KEY")

    # ── Dashboard authentication ───────────────────────────────────────────────
    dashboard_username: Optional[str] = Field(default=None, alias="DASHBOARD_USERNAME")
    dashboard_password: Optional[str] = Field(default=None, alias="DASHBOARD_PASSWORD")

    # ── Testnet / sandbox credentials ───────────────────────────────────────
    gateio_testnet_api_key: Optional[str] = Field(default=None, alias="GATEIO_TESTNET_API_KEY")
    gateio_testnet_secret_key: Optional[str] = Field(default=None, alias="GATEIO_TESTNET_SECRET_KEY")

    # Generic testnet credentials (override exchange-specific ones when mode=testnet)
    exchange_testnet_api_key: Optional[str] = Field(default=None, alias="EXCHANGE_TESTNET_API_KEY")
    exchange_testnet_api_secret: Optional[str] = Field(
        default=None, alias="EXCHANGE_TESTNET_API_SECRET"
    )
    exchange_testnet_password: Optional[str] = Field(
        default=None, alias="EXCHANGE_TESTNET_PASSWORD"
    )

    # ── Generic exchange credentials (take precedence over exchange-specific ones) ──
    primary_exchange: str = Field(default="gateio", alias="PRIMARY_EXCHANGE")
    exchange_api_key: Optional[str] = Field(default=None, alias="EXCHANGE_API_KEY")
    exchange_secret: Optional[str] = Field(default=None, alias="EXCHANGE_SECRET")
    exchange_passphrase: Optional[str] = Field(default=None, alias="EXCHANGE_PASSPHRASE")

    # ── Paper trading ─────────────────────────────────────────────────────────
    paper_trading_balance: float = Field(default=10_000.0, alias="PAPER_TRADING_BALANCE")

    # ── App behaviour ─────────────────────────────────────────────────────────
    trading_mode: str = Field(default="paper", alias="TRADING_MODE")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    enable_forex_trading: bool = Field(default=False, alias="ENABLE_FOREX_TRADING")

    # ── AI master toggle (convenience alias for AI__ENABLED nested field) ────────
    # Accepts: "on", "off", "true", "false", "1", "0" (case-insensitive).
    # When set, this takes precedence over the AI__ENABLED nested setting.
    use_ai: Optional[str] = Field(default=None, alias="USE_AI")

    # ── Trading pairs override (comma-separated) ───────────────────────────────
    trading_pairs_env: Optional[str] = Field(default=None, alias="TRADING_PAIRS")

    # ── Nested configuration objects ──────────────────────────────────────────
    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    data_sources: DataSourceConfig = Field(default_factory=DataSourceConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    trading_loop: TradingLoopConfig = Field(default_factory=TradingLoopConfig)
    forex: ForexConfig = Field(default_factory=ForexConfig)
    exness_forex: ExnessForexConfig = Field(default_factory=ExnessForexConfig)

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("trading_mode")
    @classmethod
    def validate_trading_mode(cls, v: str) -> str:
        allowed = {
            "paper",
            "live",
            "backtest",
            "testnet",
            "forex_live",
            "forex_demo",
            "forex_paper",

            "forex_exness_live",
            "forex_exness_demo",
            "forex_exness_paper",
        }
        v = v.lower()
        if v not in allowed:
            raise ValueError(f"trading_mode must be one of {allowed}")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v

    @model_validator(mode="after")
    def apply_use_ai_toggle(self) -> "Settings":
        """Propagate USE_AI env var to ai.enabled.

        Accepts ``on`` / ``off`` / ``true`` / ``false`` / ``1`` / ``0``
        (case-insensitive).  When USE_AI is set it takes precedence over the
        ``AI__ENABLED`` nested-model setting.
        """
        raw = self.use_ai.strip().lower() if self.use_ai is not None else ""
        if raw in ("off", "false", "0", "no"):
            self.ai.enabled = False
        elif raw in ("on", "true", "1", "yes"):
            self.ai.enabled = True
        # If USE_AI is not set (raw == ""), leave ai.enabled as-is (respects AI__ENABLED).
        return self

    @model_validator(mode="after")
    def enforce_live_safety(self) -> "Settings":
        """Enforce conservative safety limits when trading_mode is 'live'.

        Caps position size, leverage, open positions, and daily loss to
        sane values so that accidental misconfiguration cannot result in
        catastrophic losses in live trading.  Also forces isolated margin
        mode and ensures Telegram alerts are enabled.
        """
        if self.trading_mode == "live":
            if self.risk.max_position_size_pct > 10:
                self.risk.max_position_size_pct = 10.0
            if self.exchange.max_leverage > 20:
                self.exchange = ExchangeConfig(
                    **{**self.exchange.model_dump(), "max_leverage": 20}
                )
            if self.risk.max_open_positions > 5:
                self.risk.max_open_positions = 5
            if self.risk.max_daily_loss_pct > 3:
                self.risk.max_daily_loss_pct = 3.0
            self.risk.margin_mode = "isolated"
            self.monitoring.enable_telegram_alerts = True
        return self

    @model_validator(mode="after")
    def apply_trading_pairs_env(self) -> "Settings":
        """Parse the TRADING_PAIRS env var and override exchange.trading_pairs."""
        pairs_str = self.trading_pairs_env
        if pairs_str:
            pairs = [p.strip() for p in pairs_str.split(",") if p.strip()]
            if pairs:
                self.exchange = ExchangeConfig(
                    **{**self.exchange.model_dump(), "trading_pairs": pairs}
                )
        return self

    # ── Convenience properties ────────────────────────────────────────────────
    @property
    def is_paper_trading(self) -> bool:
        """Return True when running in paper-trading mode."""
        return self.trading_mode == "paper"

    @property
    def is_live_trading(self) -> bool:
        """Return True when running in live-trading mode."""
        return self.trading_mode == "live"

    @property
    def is_testnet_trading(self) -> bool:
        """Return True when running in testnet/sandbox mode."""
        return self.trading_mode == "testnet"

    # ── Cached factory ────────────────────────────────────────────────────────
    @classmethod
    @functools.lru_cache(maxsize=1)
    def get_settings(cls) -> "Settings":
        """Return a cached singleton Settings instance."""
        return cls()
