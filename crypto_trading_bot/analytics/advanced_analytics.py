"""Advanced market analytics including market impact models, order book analysis,
portfolio attribution, and institutional-grade performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from loguru import logger


@dataclass
class MarketImpactEstimate:
    """Market impact estimation result."""

    estimated_slippage_pct: float
    estimated_slippage_usdt: float
    market_depth_score: float  # 0-1, higher = better liquidity
    recommended_execution_method: str
    estimated_execution_time_minutes: float


class MarketImpactModel:
    """Estimate market impact and slippage for order execution.

    Uses order book depth analysis and historical data to predict
    the price impact of executing a large order.
    """

    def estimate_impact(
        self,
        order_size_usdt: float,
        orderbook: Dict,
        ticker: any,
        daily_volume_usdt: float,
    ) -> MarketImpactEstimate:
        """Estimate market impact for a given order size.

        Args:
            order_size_usdt: Order size in USDT
            orderbook: Order book data with bids/asks
            ticker: Ticker data
            daily_volume_usdt: 24h trading volume in USDT

        Returns:
            MarketImpactEstimate with impact prediction
        """
        mid_price = (ticker.bid + ticker.ask) / 2.0
        spread_pct = (ticker.ask - ticker.bid) / mid_price * 100 if mid_price > 0 else 0.1

        # Calculate order book depth
        relevant_side = orderbook.get("asks", []) if order_size_usdt > 0 else orderbook.get("bids", [])

        total_depth_usdt = sum(level[0] * level[1] for level in relevant_side[:20])

        # Market depth score
        depth_ratio = total_depth_usdt / max(order_size_usdt, 1.0)
        market_depth_score = min(1.0, depth_ratio / 5.0)  # Perfect score if 5x order size available

        # Volume ratio
        volume_ratio = order_size_usdt / max(daily_volume_usdt, 1.0)

        # Estimate slippage using multiple factors
        # 1. Spread component (immediate cost)
        spread_component = spread_pct / 2.0  # Half spread on average

        # 2. Depth component (walking the book)
        if depth_ratio >= 5.0:
            depth_component = 0.0  # Sufficient liquidity
        elif depth_ratio >= 2.0:
            depth_component = 0.05 * (5.0 - depth_ratio) / 3.0  # Small impact
        elif depth_ratio >= 1.0:
            depth_component = 0.10 * (2.0 - depth_ratio)  # Medium impact
        else:
            depth_component = 0.20 * (1.0 - depth_ratio)  # High impact

        # 3. Volume component (temporary price impact)
        if volume_ratio < 0.01:  # < 1% of daily volume
            volume_component = 0.0
        elif volume_ratio < 0.05:  # 1-5% of daily volume
            volume_component = 0.05 * volume_ratio / 0.05
        else:  # > 5% of daily volume
            volume_component = 0.10 + 0.15 * min(volume_ratio - 0.05, 0.10) / 0.10

        # Total estimated slippage
        estimated_slippage_pct = spread_component + depth_component + volume_component
        estimated_slippage_usdt = order_size_usdt * estimated_slippage_pct / 100

        # Recommend execution method
        if volume_ratio > 0.05 or depth_ratio < 1.5:
            method = "VWAP"
            exec_time = 60.0  # 1 hour
        elif volume_ratio > 0.02 or depth_ratio < 3.0:
            method = "TWAP"
            exec_time = 30.0  # 30 minutes
        elif order_size_usdt > total_depth_usdt * 0.3:
            method = "Iceberg"
            exec_time = 20.0  # 20 minutes
        else:
            method = "Market"
            exec_time = 1.0  # Immediate

        logger.debug(
            f"Market impact estimate: slippage {estimated_slippage_pct:.3f}%, "
            f"depth score {market_depth_score:.2f}, recommended: {method}"
        )

        return MarketImpactEstimate(
            estimated_slippage_pct=estimated_slippage_pct,
            estimated_slippage_usdt=estimated_slippage_usdt,
            market_depth_score=market_depth_score,
            recommended_execution_method=method,
            estimated_execution_time_minutes=exec_time,
        )


class OrderBookAnalytics:
    """Advanced order book analytics for market microstructure analysis."""

    @staticmethod
    def calculate_order_flow_toxicity(
        orderbook_snapshots: List[Dict],
        trades: List[Dict],
    ) -> float:
        """Calculate order flow toxicity (VPIN metric).

        High toxicity indicates informed trading and potential adverse selection.

        Returns:
            Toxicity score (0-1), higher = more toxic
        """
        if not trades or len(trades) < 10:
            return 0.5  # Neutral default

        # Calculate volume-synchronized probability of informed trading (VPIN)
        # Simplified version: ratio of buy vs sell imbalance
        buy_volume = sum(t.get("amount", 0) for t in trades if t.get("side") == "buy")
        sell_volume = sum(t.get("amount", 0) for t in trades if t.get("side") == "sell")

        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.5

        imbalance = abs(buy_volume - sell_volume) / total_volume
        toxicity = min(1.0, imbalance * 2.0)  # Scale to 0-1

        return toxicity

    @staticmethod
    def calculate_effective_spread(
        execution_price: float,
        mid_price: float,
        side: str,
    ) -> float:
        """Calculate effective spread - actual execution cost.

        Args:
            execution_price: Actual fill price
            mid_price: Mid price at time of order
            side: 'buy' or 'sell'

        Returns:
            Effective spread in percentage
        """
        if mid_price == 0:
            return 0.0

        if side == "buy":
            spread_pct = (execution_price - mid_price) / mid_price * 100
        else:
            spread_pct = (mid_price - execution_price) / mid_price * 100

        return max(0.0, spread_pct)

    @staticmethod
    def calculate_order_book_imbalance(orderbook: Dict) -> float:
        """Calculate order book imbalance.

        Returns:
            Imbalance ratio (-1 to 1), positive = more bids, negative = more asks
        """
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        bid_volume = sum(bid[1] for bid in bids[:10])
        ask_volume = sum(ask[1] for ask in asks[:10])

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0

        imbalance = (bid_volume - ask_volume) / total_volume
        return imbalance


class PortfolioAttributionAnalyzer:
    """Analyze portfolio returns and attribute to different factors."""

    @staticmethod
    def calculate_sharpe_attribution(
        strategy_returns: Dict[str, pd.Series],
        weights: Dict[str, float],
    ) -> Dict[str, float]:
        """Calculate Sharpe ratio attribution by strategy.

        Args:
            strategy_returns: Dict of strategy_name -> returns series
            weights: Dict of strategy_name -> weight

        Returns:
            Dict of strategy_name -> Sharpe contribution
        """
        sharpe_contributions = {}

        for strategy_name, returns in strategy_returns.items():
            if len(returns) < 2:
                sharpe_contributions[strategy_name] = 0.0
                continue

            # Calculate Sharpe ratio for this strategy
            mean_return = returns.mean()
            std_return = returns.std()

            if std_return == 0:
                sharpe = 0.0
            else:
                # Annualized Sharpe (assuming daily returns)
                sharpe = (mean_return / std_return) * np.sqrt(365)

            weight = weights.get(strategy_name, 0.0)
            sharpe_contributions[strategy_name] = sharpe * weight

        return sharpe_contributions

    @staticmethod
    def factor_attribution(
        portfolio_returns: pd.Series,
        factor_returns: Dict[str, pd.Series],
    ) -> Dict[str, float]:
        """Perform factor attribution analysis.

        Args:
            portfolio_returns: Portfolio returns series
            factor_returns: Dict of factor_name -> factor returns series

        Returns:
            Dict of factor_name -> beta (exposure)
        """
        if len(portfolio_returns) < 30:
            logger.warning("Insufficient data for factor attribution")
            return {}

        factor_betas = {}

        for factor_name, factor_ret in factor_returns.items():
            # Align series
            aligned = pd.concat([portfolio_returns, factor_ret], axis=1).dropna()

            if len(aligned) < 10:
                continue

            port_ret = aligned.iloc[:, 0]
            fact_ret = aligned.iloc[:, 1]

            # Calculate beta (exposure to this factor)
            covariance = np.cov(port_ret, fact_ret)[0, 1]
            factor_variance = np.var(fact_ret)

            if factor_variance > 0:
                beta = covariance / factor_variance
                factor_betas[factor_name] = beta

        return factor_betas


class AdvancedPerformanceMetrics:
    """Calculate institutional-grade performance metrics."""

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 365) -> float:
        """Calculate Calmar ratio (return / max drawdown).

        Args:
            returns: Returns series
            periods_per_year: Number of periods per year for annualization

        Returns:
            Calmar ratio
        """
        if len(returns) < 2:
            return 0.0

        # Annualized return
        total_return = (1 + returns).prod() - 1
        years = len(returns) / periods_per_year
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        if max_drawdown == 0:
            return 0.0

        calmar = annual_return / max_drawdown
        return calmar

    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series,
        target_return: float = 0.0,
        periods_per_year: int = 365,
    ) -> float:
        """Calculate Sortino ratio (downside risk-adjusted return).

        Args:
            returns: Returns series
            target_return: Minimum acceptable return
            periods_per_year: Number of periods per year

        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        excess_returns = returns - target_return
        mean_excess = excess_returns.mean()

        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0

        if downside_std == 0:
            return 0.0

        # Annualized Sortino
        sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
        return sortino

    @staticmethod
    def calculate_omega_ratio(
        returns: pd.Series,
        threshold: float = 0.0,
    ) -> float:
        """Calculate Omega ratio (probability-weighted ratio of gains vs losses).

        Args:
            returns: Returns series
            threshold: Threshold return level

        Returns:
            Omega ratio
        """
        if len(returns) < 2:
            return 1.0

        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns < threshold]

        gains_sum = gains.sum() if len(gains) > 0 else 0.0
        losses_sum = losses.sum() if len(losses) > 0 else 1e-10

        omega = gains_sum / losses_sum
        return omega

    @staticmethod
    def calculate_information_ratio(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        periods_per_year: int = 365,
    ) -> float:
        """Calculate Information Ratio (active return / tracking error).

        Args:
            portfolio_returns: Portfolio returns series
            benchmark_returns: Benchmark returns series
            periods_per_year: Number of periods per year

        Returns:
            Information ratio
        """
        # Align series
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()

        if len(aligned) < 2:
            return 0.0

        excess_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        mean_excess = excess_returns.mean()
        tracking_error = excess_returns.std()

        if tracking_error == 0:
            return 0.0

        # Annualized Information Ratio
        info_ratio = (mean_excess / tracking_error) * np.sqrt(periods_per_year)
        return info_ratio


class RegimeDetectionAdvanced:
    """Enhanced regime detection with confidence intervals."""

    @staticmethod
    def detect_regime_with_confidence(
        prices: pd.Series,
        window: int = 20,
    ) -> Tuple[str, float]:
        """Detect market regime with confidence level.

        Returns:
            Tuple of (regime, confidence) where regime is 'trending_up',
            'trending_down', 'ranging', or 'volatile'
        """
        if len(prices) < window:
            return "unknown", 0.0

        returns = prices.pct_change().dropna()

        # Calculate metrics
        volatility = returns.std()
        trend_strength = abs(returns.mean()) / (volatility + 1e-10)

        # Calculate directional move
        price_change_pct = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] * 100

        # Trend detection with ADX-like logic
        if trend_strength > 1.0:  # Strong trend
            confidence = min(1.0, trend_strength / 2.0)
            if price_change_pct > 2.0:
                return "trending_up", confidence
            elif price_change_pct < -2.0:
                return "trending_down", confidence
            else:
                return "ranging", confidence * 0.7
        else:
            # Check volatility
            if volatility > returns.std() * 1.5:
                return "volatile", 0.7
            else:
                return "ranging", 0.6

    @staticmethod
    def detect_volatility_regime(
        returns: pd.Series,
        window: int = 20,
    ) -> Tuple[str, float]:
        """Detect volatility regime.

        Returns:
            Tuple of (regime, current_volatility) where regime is
            'low', 'normal', or 'high'
        """
        if len(returns) < window * 2:
            return "normal", 0.0

        recent_vol = returns.iloc[-window:].std()
        historical_vol = returns.iloc[-window*2:-window].std()

        vol_ratio = recent_vol / (historical_vol + 1e-10)

        if vol_ratio > 1.5:
            regime = "high"
        elif vol_ratio < 0.7:
            regime = "low"
        else:
            regime = "normal"

        return regime, recent_vol * np.sqrt(365) * 100  # Annualized vol %
