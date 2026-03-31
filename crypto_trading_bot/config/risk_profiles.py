"""Risk profiles for conservative, moderate, and aggressive trading styles."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class RiskProfile(str, Enum):
    """Enumeration of supported risk profiles."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class RiskProfileConfig(BaseModel):
    """Full risk parameter set for a given trading style."""

    profile: RiskProfile
    max_position_size_pct: float
    max_open_positions: int
    max_leverage: int
    max_daily_loss_pct: float
    daily_profit_target_pct: float
    max_drawdown_pct: float
    default_stop_loss_pct: float
    default_take_profit_pct: float
    trailing_stop_pct: float
    risk_reward_min: float
    use_kelly_criterion: bool
    max_correlation: float
    circuit_breaker_loss_pct: float
    cooldown_after_loss_minutes: int


CONSERVATIVE_PROFILE = RiskProfileConfig(
    profile=RiskProfile.CONSERVATIVE,
    max_position_size_pct=5.0,
    max_open_positions=3,
    max_leverage=3,
    max_daily_loss_pct=1.0,
    daily_profit_target_pct=0.5,
    max_drawdown_pct=5.0,
    default_stop_loss_pct=1.5,
    default_take_profit_pct=2.0,
    trailing_stop_pct=1.0,
    risk_reward_min=1.05,
    use_kelly_criterion=True,
    max_correlation=0.5,
    circuit_breaker_loss_pct=3.0,
    cooldown_after_loss_minutes=60,
)

MODERATE_PROFILE = RiskProfileConfig(
    profile=RiskProfile.MODERATE,
    max_position_size_pct=10.0,
    max_open_positions=5,
    max_leverage=5,
    max_daily_loss_pct=2.0,
    daily_profit_target_pct=1.0,
    max_drawdown_pct=10.0,
    default_stop_loss_pct=2.0,
    default_take_profit_pct=3.0,
    trailing_stop_pct=1.05,
    risk_reward_min=1.5,
    use_kelly_criterion=True,
    max_correlation=0.7,
    circuit_breaker_loss_pct=5.0,
    cooldown_after_loss_minutes=30,
)

AGGRESSIVE_PROFILE = RiskProfileConfig(
    profile=RiskProfile.AGGRESSIVE,
    max_position_size_pct=20.0,
    max_open_positions=10,
    max_leverage=10,
    max_daily_loss_pct=3.0,
    daily_profit_target_pct=2.0,
    max_drawdown_pct=20.0,
    default_stop_loss_pct=3.0,
    default_take_profit_pct=5.0,
    trailing_stop_pct=2.0,
    risk_reward_min=1.08,
    use_kelly_criterion=False,
    max_correlation=0.85,
    circuit_breaker_loss_pct=8.0,
    cooldown_after_loss_minutes=15,
)

_PROFILE_MAP: dict[RiskProfile, RiskProfileConfig] = {
    RiskProfile.CONSERVATIVE: CONSERVATIVE_PROFILE,
    RiskProfile.MODERATE: MODERATE_PROFILE,
    RiskProfile.AGGRESSIVE: AGGRESSIVE_PROFILE,
}


def get_risk_profile(profile: RiskProfile) -> RiskProfileConfig:
    """Return the :class:`RiskProfileConfig` for the given *profile*.

    Args:
        profile: One of the :class:`RiskProfile` enum values.

    Returns:
        The corresponding :class:`RiskProfileConfig` instance.

    Raises:
        ValueError: If *profile* is not a valid :class:`RiskProfile` member.
    """
    if profile not in _PROFILE_MAP:
        raise ValueError(f"Unknown risk profile: {profile!r}")
    return _PROFILE_MAP[profile]
