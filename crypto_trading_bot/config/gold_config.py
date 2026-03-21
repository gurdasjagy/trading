"""Gold-specific trading configuration and knowledge base."""

# Gate.io Standard Futures (Crypto Mode) - XAUT/USDT perpetual:
# Gate.io does NOT have native XAU/USDT on their standard futures/swap API.
# The XAU/USDT perpetual visible in the Gate.io app is on their TradFi (MT5) platform.
#
# For crypto futures mode (TRADING_MODE=paper/live/testnet):
#   XAU/USDT is automatically mapped to XAUT/USDT (Tether Gold perpetual).
#   XAUT/USDT:USDT IS a valid Gate.io perpetual contract.
#   Tether Gold (XAUT) is backed 1:1 by physical gold and tracks the gold price.
#   Max leverage on XAUT perpetual: 20x.
#
# For forex mode (TRADING_MODE=forex_live/forex_demo):
#   Use GateIOTradFiMT5Client with native XAUUSD.
#   Requires MT5 credentials (login/password/server).

GOLD_FUTURES_CONFIG = {
    "XAU/USDT": {
        # Gate.io standard futures mode — maps to XAUT/USDT (Tether Gold perpetual)
        "exchange": "gateio",
        "ccxt_symbol": "XAUT/USDT",  # Actual CCXT symbol used (auto-mapped from XAU/USDT)
        "tokenized": True,            # XAUT is a tokenized gold product
        "contract_size": 1.0,         # 1 XAUT ≈ 1 troy oz gold
        "tick_size": 0.001,
        "min_order_size": 0.01,
        "max_leverage": 20,           # XAUT perpetual on Gate.io supports up to 20x
        "default_leverage": 5,
        "trading_hours": "24/7",
        "volatility_profile": "high",
        "correlation": {
            "DXY": -0.85,  # Strong inverse correlation with dollar
            "SPX": -0.3,   # Mild inverse with stocks
            "BTC": 0.4,    # Moderate positive with Bitcoin
        },
        # Gold-specific strategy preferences
        "preferred_strategies": [
            "gold_momentum_breakout",
            "gold_fibonacci",
            "gold_mean_reversion",
            "gold_rsi_divergence",
            "gold_bollinger_squeeze",
            "gold_ichimoku",
            "fibonacci_retracement",
            "bollinger_squeeze",
            "supertrend",
        ],
        # Risk parameters specific to gold
        "risk": {
            "max_position_size_pct": 10.0,
            "default_stop_loss_pct": 2.0,
            "default_take_profit_pct": 4.0,
            "trailing_stop_pct": 1.5,
            "max_daily_loss_pct": 2.5,
        },
    },
    "XAU/USDT:TRADFI": {
        # Gate.io TradFi MT5 mode — native XAUUSD perpetual
        # Use GateIOTradFiMT5Client with TRADING_MODE=forex_live or forex_demo
        "exchange": "gateio_tradfi_mt5",
        "mt5_symbol": "XAUUSD",
        "contract_size": 1.0,  # 1 oz per contract (Gate.io TradFi specific)
        "tick_size": 0.01,
        "min_order_size": 0.01,
        "max_leverage": 500,   # Gate.io TradFi supports up to 500x
        "default_leverage": 10,
        "trading_hours": "24/7",
        "volatility_profile": "high",
        "correlation": {
            "DXY": -0.85,
            "SPX": -0.3,
            "BTC": 0.4,
        },
        "preferred_strategies": [
            "gold_momentum_breakout",
            "gold_fibonacci",
            "gold_mean_reversion",
        ],
        "risk": {
            "max_position_size_pct": 10.0,
            "default_stop_loss_pct": 1.0,
            "default_take_profit_pct": 2.5,
            "trailing_stop_pct": 0.8,
            "max_daily_loss_pct": 2.5,
        },
    },
    "XAG/USDT": {
        # Gate.io standard futures — maps to XAUT/USDT (no silver token; fallback to gold)
        "exchange": "gateio",
        "ccxt_symbol": "XAUT/USDT",  # Fallback: no XAGT token on Gate.io standard futures
        "tokenized": True,
        "contract_size": 1.0,
        "tick_size": 0.001,
        "min_order_size": 0.01,
        "max_leverage": 20,
        "default_leverage": 5,
        "trading_hours": "24/7",
        "volatility_profile": "very_high",  # Silver more volatile than gold
        "correlation": {
            "DXY": -0.80,
            "SPX": -0.25,
            "XAU": 0.85,  # High correlation with gold
        },
        "preferred_strategies": [
            "gold_momentum_breakout",
            "gold_mean_reversion",
            "supertrend",
        ],
        "risk": {
            "max_position_size_pct": 8.0,
            "default_stop_loss_pct": 2.0,
            "default_take_profit_pct": 4.0,
            "trailing_stop_pct": 1.5,
            "max_daily_loss_pct": 2.0,
        },
    },
}

# Gold market knowledge for AI brain
GOLD_TRADING_KNOWLEDGE = """
Gold (XAU) Trading Knowledge Base:
- Gold is a safe-haven asset that rallies during uncertainty
- Inverse correlation with USD (DXY) is the strongest driver
- Key levels: $2000, $2050, $2100 are major psychological levels
- London session (08:00-16:00 GMT) has highest gold volume
- NFP, CPI, FOMC are the most impactful events for gold
- Central bank buying is a major long-term driver
- Gold respects Fibonacci levels exceptionally well
- ATR for gold is typically $15-30/day
- Gold trends strongly - trend following strategies work well
- Mean reversion works at extreme deviations (>2 std dev from 200 MA)

Gate.io Gold Trading Modes:
- For crypto futures mode (TRADING_MODE=paper/live/testnet):
  XAU/USDT is automatically mapped to XAUT/USDT (Tether Gold perpetual).
  XAUT is backed 1:1 by physical gold and closely tracks the spot price.
  Max leverage: 20x. Traded on Gate.io standard futures API via CCXT.
- For forex mode (TRADING_MODE=forex_live/forex_demo):
  Use GateIOTradFiMT5Client with native XAUUSD (MT5 format).
  Up to 500x leverage. Requires: GATEIO_TRADFI_LOGIN, GATEIO_TRADFI_PASSWORD, GATEIO_TRADFI_SERVER.
  Enable with: ENABLE_FOREX_TRADING=true

IMPORTANT: Gate.io does NOT have native XAU/USDT on their standard futures/swap API.
The XAU/USDT visible in the Gate.io app is on their TradFi (MT5) platform, not the
standard futures API. For crypto futures mode, always use the XAUT/USDT mapping.
"""

