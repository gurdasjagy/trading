"""Tests for the Rust WebSocket parser (Phase 1).

These tests exercise ``rust_trading_engine.ws_parser`` directly and also
verify that the Python ``MarketDataFeed._parse_ticker`` correctly activates
the Rust fast-path when raw bytes are passed instead of a dict.
"""

from __future__ import annotations

import json
import pytest

# Skip entire module when the Rust extension is not installed (CI without maturin).
pytest.importorskip("rust_trading_engine")

from rust_trading_engine.ws_parser import (
    RustTicker,
    detect_significant_move,
    parse_orderbook_message,
    parse_ticker_message,
    parse_trade_message,
    parse_ws_message,
)


# ---------------------------------------------------------------------------
# parse_ticker_message
# ---------------------------------------------------------------------------


class TestParseTickerMessage:
    """Coverage for field-resolution logic across exchange message formats."""

    def _encode(self, obj: dict) -> bytes:
        return json.dumps(obj).encode()

    def test_gate_io_style(self):
        msg = self._encode(
            {
                "last": 42000.5,
                "bid": 41999.0,
                "ask": 42001.0,
                "high": 43000.0,
                "low": 41000.0,
                "volume": 1234.5,
                "timestamp": 1234567890,
            }
        )
        ticker = parse_ticker_message("BTC/USDT", msg)
        assert ticker is not None
        assert isinstance(ticker, RustTicker)
        assert ticker.symbol == "BTC/USDT"
        assert ticker.last == pytest.approx(42000.5)
        assert ticker.bid == pytest.approx(41999.0)
        assert ticker.ask == pytest.approx(42001.0)
        assert ticker.high == pytest.approx(43000.0)
        assert ticker.low == pytest.approx(41000.0)
        assert ticker.volume == pytest.approx(1234.5)
        assert ticker.timestamp == 1234567890

    def test_mexc_style_c_b_a(self):
        """MEXC uses c/b/a/v/ts keys."""
        msg = self._encode({"c": 100.5, "b": 100.3, "a": 100.7, "v": 500.0, "ts": 1234567})
        ticker = parse_ticker_message("ETH/USDT", msg)
        assert ticker is not None
        assert ticker.last == pytest.approx(100.5)
        assert ticker.bid == pytest.approx(100.3)
        assert ticker.ask == pytest.approx(100.7)
        assert ticker.volume == pytest.approx(500.0)
        assert ticker.timestamp == 1234567

    def test_bitget_style_lastPr_bestBid_bestAsk(self):
        """Bitget uses lastPr / bestBid / bestAsk."""
        msg = self._encode({"lastPr": 200.0, "bestBid": 199.9, "bestAsk": 200.1})
        ticker = parse_ticker_message("LTC/USDT", msg)
        assert ticker is not None
        assert ticker.last == pytest.approx(200.0)
        assert ticker.bid == pytest.approx(199.9)
        assert ticker.ask == pytest.approx(200.1)

    def test_string_numbers(self):
        """Gate.io futures sends all numbers as strings."""
        msg = self._encode({"last": "42000.5", "bid": "41999.0", "ask": "42001.0"})
        ticker = parse_ticker_message("BTC/USDT", msg)
        assert ticker is not None
        assert ticker.last == pytest.approx(42000.5)
        assert ticker.bid == pytest.approx(41999.0)

    def test_subscription_confirmation_returns_none(self):
        """Subscription confirmations have no price fields → return None."""
        msg = self._encode({"event": "subscribe", "status": "ok"})
        ticker = parse_ticker_message("BTC/USDT", msg)
        assert ticker is None

    def test_data_wrapper_dict(self):
        """Messages with a ``data`` dict wrapper."""
        msg = self._encode({"time": 123, "data": {"last": 50000.0, "bid": 49999.0}})
        ticker = parse_ticker_message("BTC/USDT", msg)
        assert ticker is not None
        assert ticker.last == pytest.approx(50000.0)

    def test_data_wrapper_list(self):
        """Messages where ``data`` is a list — first element is taken."""
        msg = self._encode({"data": [{"last": 50000.0, "bid": 49999.0, "ask": 50001.0}]})
        ticker = parse_ticker_message("BTC/USDT", msg)
        assert ticker is not None
        assert ticker.last == pytest.approx(50000.0)

    def test_data_wrapper_empty_list_returns_none(self):
        msg = self._encode({"data": []})
        ticker = parse_ticker_message("BTC/USDT", msg)
        assert ticker is None

    def test_bid_fallback_to_last(self):
        """When bid/ask are missing, they fall back to last."""
        msg = self._encode({"last": 100.0})
        ticker = parse_ticker_message("X/USDT", msg)
        assert ticker is not None
        assert ticker.bid == pytest.approx(100.0)
        assert ticker.ask == pytest.approx(100.0)
        assert ticker.high == pytest.approx(100.0)
        assert ticker.low == pytest.approx(100.0)
        assert ticker.volume == pytest.approx(0.0)
        assert ticker.timestamp == 0

    def test_funding_rate_field(self):
        msg = self._encode({"last": 100.0, "fundingRate": 0.0001})
        ticker = parse_ticker_message("BTC/USDT", msg)
        assert ticker is not None
        assert ticker.funding_rate == pytest.approx(0.0001)

    def test_open_interest_field(self):
        msg = self._encode({"last": 100.0, "openInterest": 12345.0})
        ticker = parse_ticker_message("BTC/USDT", msg)
        assert ticker is not None
        assert ticker.open_interest == pytest.approx(12345.0)

    def test_missing_funding_rate_is_none(self):
        msg = self._encode({"last": 100.0})
        ticker = parse_ticker_message("BTC/USDT", msg)
        assert ticker is not None
        assert ticker.funding_rate is None
        assert ticker.open_interest is None

    def test_repr(self):
        msg = self._encode({"last": 42000.0, "bid": 41999.0, "ask": 42001.0})
        ticker = parse_ticker_message("BTC/USDT", msg)
        r = repr(ticker)
        assert "RustTicker" in r
        assert "BTC/USDT" in r


# ---------------------------------------------------------------------------
# parse_orderbook_message
# ---------------------------------------------------------------------------


class TestParseOrderbookMessage:
    def _encode(self, obj: dict) -> bytes:
        return json.dumps(obj).encode()

    def test_basic(self):
        msg = self._encode(
            {
                "bids": [[41999.0, 1.5], [41998.0, 2.0]],
                "asks": [[42001.0, 1.0], [42002.0, 3.0]],
                "timestamp": 999,
            }
        )
        result = parse_orderbook_message(msg)
        assert result is not None
        assert len(result["bids"]) == 2
        assert len(result["asks"]) == 2
        assert result["bids"][0][0] == pytest.approx(41999.0)
        assert result["asks"][0][0] == pytest.approx(42001.0)
        assert result["timestamp"] == 999

    def test_data_wrapper(self):
        msg = self._encode(
            {
                "data": {
                    "bids": [[100.0, 5.0]],
                    "asks": [[101.0, 3.0]],
                    "t": 12345,
                }
            }
        )
        result = parse_orderbook_message(msg)
        assert result is not None
        assert result["bids"][0][0] == pytest.approx(100.0)
        assert result["timestamp"] == 12345

    def test_empty_sides(self):
        msg = self._encode({"bids": [], "asks": []})
        result = parse_orderbook_message(msg)
        assert result is not None
        assert result["bids"] == []
        assert result["asks"] == []


# ---------------------------------------------------------------------------
# parse_ws_message
# ---------------------------------------------------------------------------


class TestParseWsMessage:
    def test_basic_message(self):
        raw = json.dumps({"channel": "futures.tickers", "event": "update"}).encode()
        result = parse_ws_message(raw)
        assert result["channel"] == "futures.tickers"
        assert result["event"] == "update"

    def test_nested_data(self):
        raw = json.dumps({"data": [{"last": "42000"}]}).encode()
        result = parse_ws_message(raw)
        assert isinstance(result, dict)
        assert "data" in result


# ---------------------------------------------------------------------------
# parse_trade_message
# ---------------------------------------------------------------------------


class TestParseTradeMessage:
    def test_trade_with_data_wrapper(self):
        raw = json.dumps(
            {"data": {"price": "100.5", "size": "1.0", "side": "buy"}}
        ).encode()
        result = parse_trade_message(raw)
        assert result is not None
        assert result["price"] == "100.5"

    def test_trade_top_level(self):
        raw = json.dumps({"price": "50.0", "amount": "2.5"}).encode()
        result = parse_trade_message(raw)
        assert result is not None


# ---------------------------------------------------------------------------
# detect_significant_move
# ---------------------------------------------------------------------------


class TestDetectSignificantMove:
    def test_move_above_threshold(self):
        """0.2 % move with 0.1 % threshold → True."""
        assert detect_significant_move(100.0, 100.2, 0.001) is True

    def test_move_below_threshold(self):
        """0.05 % move with 0.1 % threshold → False."""
        assert detect_significant_move(100.0, 100.05, 0.001) is False

    def test_exactly_at_threshold(self):
        """Move >= threshold → True (use integer-friendly values)."""
        # 110 / 100 - 1 = 0.1 exactly in floating point → 10 % move, 0.1 threshold
        assert detect_significant_move(1000.0, 1001.0, 0.001) is True

    def test_negative_move(self):
        """Downward move is also detected."""
        assert detect_significant_move(100.0, 99.8, 0.001) is True

    def test_zero_old_price(self):
        """old_price == 0 → False (avoids division by zero)."""
        assert detect_significant_move(0.0, 100.0, 0.001) is False


# ---------------------------------------------------------------------------
# Integration: Python MarketDataFeed._parse_ticker with Rust fast-path
# ---------------------------------------------------------------------------


class TestMarketDataFeedRustFastPath:
    """Verify that MarketDataFeed._parse_ticker activates the Rust fast-path
    when raw bytes are provided."""

    def test_rust_fastpath_bytes(self):
        from crypto_trading_bot.exchange.websocket_feeds import MarketDataFeed, _USE_RUST_PARSER
        assert _USE_RUST_PARSER is True, "Rust parser should be available in tests"

        raw = json.dumps({"last": 42000.0, "bid": 41999.0, "ask": 42001.0}).encode()
        ticker = MarketDataFeed._parse_ticker("BTC/USDT", raw)
        assert ticker is not None
        assert ticker.last == pytest.approx(42000.0)

    def test_python_fallback_dict(self):
        """When a dict is passed the Python path is used (no raw bytes)."""
        from crypto_trading_bot.exchange.websocket_feeds import MarketDataFeed
        data = {"data": {"last": 100.0, "bid": 99.5, "ask": 100.5}}
        ticker = MarketDataFeed._parse_ticker("ETH/USDT", data)
        assert ticker is not None
        assert ticker.last == pytest.approx(100.0)

    def test_none_on_no_price_fields(self):
        from crypto_trading_bot.exchange.websocket_feeds import MarketDataFeed
        data = {"event": "subscribe", "status": "ok"}
        ticker = MarketDataFeed._parse_ticker("BTC/USDT", data)
        assert ticker is None
