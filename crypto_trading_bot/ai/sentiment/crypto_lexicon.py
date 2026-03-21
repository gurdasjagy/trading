"""Crypto-specific sentiment lexicon for domain-aware scoring."""

import re
from typing import Dict, List, Tuple

from loguru import logger

from ai.sentiment.analyzer import _LABEL_THRESHOLDS, SentimentResult

# ---------------------------------------------------------------------------
# Crypto-specific lexicons
# ---------------------------------------------------------------------------

_BULLISH_TERMS: Dict[str, float] = {
    # Strong positive signals
    "moon": 0.8,
    "mooning": 0.85,
    "to the moon": 0.9,
    "lambo": 0.7,
    "ath": 0.8,
    "all time high": 0.85,
    "all-time high": 0.85,
    "hodl": 0.5,
    "accumulate": 0.6,
    "accumulation": 0.6,
    "fomo": 0.5,
    "ape in": 0.6,
    "aping": 0.5,
    "gem": 0.65,
    "hidden gem": 0.7,
    "bullish": 0.7,
    "pump": 0.6,
    "pumping": 0.65,
    "green": 0.4,
    "greens": 0.4,
    "breakout": 0.7,
    "ripping": 0.6,
    "rip": 0.5,
    "flying": 0.6,
    "surge": 0.65,
    "surging": 0.65,
    "rally": 0.6,
    "rallying": 0.65,
    "institutional buying": 0.8,
    "etf approved": 0.9,
    "adoption": 0.6,
    "mainstream": 0.55,
    "buy the dip": 0.6,
    "btd": 0.6,
    "degen": 0.4,
    "wagmi": 0.7,
    "we're all gonna make it": 0.75,
    "gigachad": 0.5,
    "based": 0.4,
    "short squeeze": 0.7,
    "rekt shorts": 0.7,
    "liquidated shorts": 0.65,
    "golden cross": 0.7,
    "support held": 0.6,
    "oversold": 0.5,
}

_BEARISH_TERMS: Dict[str, float] = {
    # Strong negative signals
    "rekt": 0.7,
    "getting rekt": 0.75,
    "dump": 0.65,
    "dumping": 0.7,
    "rug pull": 0.95,
    "rug": 0.8,
    "rugged": 0.85,
    "exit scam": 0.95,
    "scam": 0.85,
    "fud": 0.55,
    "fear uncertainty doubt": 0.6,
    "crash": 0.8,
    "crashing": 0.85,
    "bearish": 0.7,
    "bear market": 0.75,
    "red": 0.35,
    "bleeding": 0.65,
    "bleed": 0.6,
    "capitulation": 0.75,
    "capitulating": 0.75,
    "ngmi": 0.65,
    "not gonna make it": 0.7,
    "rekt longs": 0.7,
    "liquidated longs": 0.65,
    "hack": 0.85,
    "hacked": 0.85,
    "exploit": 0.8,
    "exploited": 0.8,
    "vulnerability": 0.65,
    "ban": 0.75,
    "banned": 0.8,
    "regulation crackdown": 0.8,
    "sec lawsuit": 0.85,
    "death cross": 0.7,
    "resistance rejected": 0.6,
    "overbought": 0.5,
    "ponzi": 0.9,
    "wash trading": 0.75,
    "paper hands": 0.5,
    "sell the news": 0.5,
    "dump the news": 0.55,
    "shakeout": 0.5,
    "stop hunt": 0.55,
}

_NEUTRAL_AMPLIFIERS: Tuple[str, ...] = (
    "massive",
    "huge",
    "significant",
    "major",
    "historic",
    "insane",
    "extreme",
    "wild",
)


def _score_to_label(score: float) -> str:
    for low, high, label in _LABEL_THRESHOLDS:
        if low <= score < high:
            return label
    return "neutral"


class CryptoLexicon:
    """Domain-specific crypto sentiment lexicon analyzer.

    Provides fast keyword-based scoring that captures crypto slang and
    community terminology not well-handled by generic sentiment models.
    """

    def __init__(self) -> None:
        self._lexicon: Dict[str, float] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _load_lexicon(self) -> Dict[str, float]:
        """Build and return the combined lexicon dict.

        Bullish terms have positive scores; bearish terms have negative scores.
        Scores are in [-1, 1].
        """
        if self._loaded:
            return self._lexicon
        lexicon: Dict[str, float] = {}
        for term, strength in _BULLISH_TERMS.items():
            lexicon[term] = strength
        for term, strength in _BEARISH_TERMS.items():
            lexicon[term] = -strength
        self._lexicon = lexicon
        self._loaded = True
        logger.debug(f"CryptoLexicon loaded {len(lexicon)} terms")
        return lexicon

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_crypto_score(self, text: str) -> float:
        """Return a weighted sentiment score in [-1, 1] for *text*.

        Scans the text for all matching terms and returns a weighted average of
        their scores.  Returns 0.0 if no terms are found.

        Args:
            text: Raw text to score.

        Returns:
            Float in [-1.0, 1.0].
        """
        lexicon = self._load_lexicon()
        text_lower = text.lower()

        # Check for amplifier nearby (simple heuristic)
        has_amplifier = any(amp in text_lower for amp in _NEUTRAL_AMPLIFIERS)
        amplifier_mult = 1.2 if has_amplifier else 1.0

        matches: List[float] = []
        for term, score in lexicon.items():
            # Use word-boundary-aware matching for short terms
            if len(term) <= 4:
                pattern = rf"\b{re.escape(term)}\b"
                if re.search(pattern, text_lower):
                    matches.append(score * amplifier_mult)
            else:
                if term in text_lower:
                    matches.append(score * amplifier_mult)

        if not matches:
            return 0.0

        raw = sum(matches) / len(matches)
        return max(-1.0, min(1.0, raw))

    def analyze(self, text: str) -> SentimentResult:
        """Analyze *text* using the crypto lexicon.

        Args:
            text: Raw text to analyze.

        Returns:
            :class:`SentimentResult` with score in [-1, 1].
        """
        try:
            score = self.get_crypto_score(text)
            label = _score_to_label(score)
            lexicon = self._load_lexicon()
            text_lower = text.lower()
            matches = sum(1 for term in lexicon if term in text_lower)
            confidence = min(1.0, matches * 0.15 + abs(score) * 0.5)
            return SentimentResult(
                score=score,
                label=label,
                confidence=confidence,
                breakdown={"crypto_lexicon": score},
            )
        except Exception as exc:
            logger.warning(f"CryptoLexicon.analyze error: {exc}")
            return SentimentResult(score=0.0, label="neutral", confidence=0.0, breakdown={})
