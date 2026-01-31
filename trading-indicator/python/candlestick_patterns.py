#!/usr/bin/env python3
"""
CANDLESTICK PATTERN RECOGNITION MODULE
=====================================
Detects 20+ candlestick patterns for trading signals.

Patterns Included:
- Doji (standard, dragonfly, gravestone, long-legged)
- Hammer / Inverted Hammer / Hanging Man / Shooting Star
- Engulfing (bullish/bearish)
- Morning Star / Evening Star
- Three White Soldiers / Three Black Crows
- Harami / Harami Cross
- Piercing Line / Dark Cloud Cover
- Spinning Top
- Marubozu (bullish/bearish)
- Tweezer Top / Tweezer Bottom
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PatternResult:
    """Result of pattern detection."""
    name: str
    direction: str  # 'bullish', 'bearish', 'neutral'
    strength: float  # 0.0 to 1.0
    index: int
    confidence: float


class CandlestickPatternDetector:
    """
    Detects candlestick patterns in OHLC data.
    
    Usage:
        detector = CandlestickPatternDetector()
        patterns = detector.detect_all(open_prices, high_prices, low_prices, close_prices)
    """
    
    def __init__(self, body_threshold: float = 0.1, shadow_threshold: float = 2.0):
        """
        Initialize pattern detector.
        
        Args:
            body_threshold: Minimum body size as % of range for significant candle
            shadow_threshold: Shadow must be this many times body size for hammer/doji
        """
        self.body_threshold = body_threshold
        self.shadow_threshold = shadow_threshold
    
    def _body_size(self, open_price: float, close_price: float) -> float:
        """Calculate absolute body size."""
        return abs(close_price - open_price)
    
    def _candle_range(self, high: float, low: float) -> float:
        """Calculate total candle range."""
        return high - low
    
    def _upper_shadow(self, open_price: float, high: float, close_price: float) -> float:
        """Calculate upper shadow size."""
        return high - max(open_price, close_price)
    
    def _lower_shadow(self, open_price: float, low: float, close_price: float) -> float:
        """Calculate lower shadow size."""
        return min(open_price, close_price) - low
    
    def _is_bullish(self, open_price: float, close_price: float) -> bool:
        """Check if candle is bullish (close > open)."""
        return close_price > open_price
    
    def _is_bearish(self, open_price: float, close_price: float) -> bool:
        """Check if candle is bearish (close < open)."""
        return close_price < open_price
    
    # ==========================================================================
    # SINGLE CANDLE PATTERNS
    # ==========================================================================
    
    def detect_doji(self, o: float, h: float, l: float, c: float) -> Optional[str]:
        """
        Detect Doji pattern (open ≈ close).
        
        Variants:
        - Standard Doji: small body, equal shadows
        - Dragonfly Doji: long lower shadow, no upper shadow
        - Gravestone Doji: long upper shadow, no lower shadow
        - Long-legged Doji: long both shadows
        """
        candle_range = self._candle_range(h, l)
        if candle_range == 0:
            return None
            
        body = self._body_size(o, c)
        body_pct = body / candle_range
        
        if body_pct > 0.1:  # Body too large for doji
            return None
        
        upper = self._upper_shadow(o, h, c)
        lower = self._lower_shadow(o, l, c)
        
        upper_pct = upper / candle_range
        lower_pct = lower / candle_range
        
        # Dragonfly: long lower shadow, no upper
        if lower_pct > 0.6 and upper_pct < 0.1:
            return "dragonfly_doji"
        
        # Gravestone: long upper shadow, no lower
        if upper_pct > 0.6 and lower_pct < 0.1:
            return "gravestone_doji"
        
        # Long-legged: both shadows long
        if upper_pct > 0.3 and lower_pct > 0.3:
            return "long_legged_doji"
        
        return "doji"
    
    def detect_hammer(self, o: float, h: float, l: float, c: float, 
                      prev_trend: str = 'down') -> Optional[str]:
        """
        Detect Hammer/Hanging Man/Inverted Hammer/Shooting Star.
        
        - Hammer: Downtrend, long lower shadow, small body at top
        - Hanging Man: Uptrend, same shape as hammer
        - Inverted Hammer: Downtrend, long upper shadow, small body at bottom
        - Shooting Star: Uptrend, same shape as inverted hammer
        """
        candle_range = self._candle_range(h, l)
        if candle_range == 0:
            return None
            
        body = self._body_size(o, c)
        upper = self._upper_shadow(o, h, c)
        lower = self._lower_shadow(o, l, c)
        
        body_pct = body / candle_range
        upper_pct = upper / candle_range
        lower_pct = lower / candle_range
        
        # Hammer/Hanging Man: long lower shadow (>60%), small upper (<10%)
        if lower_pct > 0.6 and upper_pct < 0.1 and body_pct < 0.3:
            return "hammer" if prev_trend == 'down' else "hanging_man"
        
        # Inverted Hammer/Shooting Star: long upper shadow (>60%), small lower (<10%)
        if upper_pct > 0.6 and lower_pct < 0.1 and body_pct < 0.3:
            return "inverted_hammer" if prev_trend == 'down' else "shooting_star"
        
        return None
    
    def detect_marubozu(self, o: float, h: float, l: float, c: float) -> Optional[str]:
        """
        Detect Marubozu (no shadows, full body candle).
        Strong trend continuation signal.
        """
        candle_range = self._candle_range(h, l)
        if candle_range == 0:
            return None
            
        upper = self._upper_shadow(o, h, c)
        lower = self._lower_shadow(o, l, c)
        
        shadow_tolerance = candle_range * 0.05  # 5% tolerance
        
        if upper < shadow_tolerance and lower < shadow_tolerance:
            if self._is_bullish(o, c):
                return "bullish_marubozu"
            else:
                return "bearish_marubozu"
        
        return None
    
    def detect_spinning_top(self, o: float, h: float, l: float, c: float) -> bool:
        """
        Detect Spinning Top (small body, long shadows on both sides).
        Indicates indecision.
        """
        candle_range = self._candle_range(h, l)
        if candle_range == 0:
            return False
            
        body = self._body_size(o, c)
        upper = self._upper_shadow(o, h, c)
        lower = self._lower_shadow(o, l, c)
        
        body_pct = body / candle_range
        upper_pct = upper / candle_range
        lower_pct = lower / candle_range
        
        # Small body (10-30%), both shadows significant (>20%)
        return 0.1 < body_pct < 0.3 and upper_pct > 0.2 and lower_pct > 0.2
    
    # ==========================================================================
    # TWO CANDLE PATTERNS
    # ==========================================================================
    
    def detect_engulfing(self, o1: float, h1: float, l1: float, c1: float,
                         o2: float, h2: float, l2: float, c2: float) -> Optional[str]:
        """
        Detect Engulfing pattern (second candle completely engulfs first).
        
        - Bullish Engulfing: Bearish candle followed by larger bullish candle
        - Bearish Engulfing: Bullish candle followed by larger bearish candle
        """
        body1 = self._body_size(o1, c1)
        body2 = self._body_size(o2, c2)
        
        # Second body must be larger
        if body2 <= body1:
            return None
        
        # Bullish Engulfing
        if self._is_bearish(o1, c1) and self._is_bullish(o2, c2):
            if o2 <= c1 and c2 >= o1:
                return "bullish_engulfing"
        
        # Bearish Engulfing
        if self._is_bullish(o1, c1) and self._is_bearish(o2, c2):
            if o2 >= c1 and c2 <= o1:
                return "bearish_engulfing"
        
        return None
    
    def detect_harami(self, o1: float, h1: float, l1: float, c1: float,
                      o2: float, h2: float, l2: float, c2: float) -> Optional[str]:
        """
        Detect Harami pattern (second candle contained within first).
        
        - Bullish Harami: Large bearish followed by small bullish inside
        - Bearish Harami: Large bullish followed by small bearish inside
        """
        body1 = self._body_size(o1, c1)
        body2 = self._body_size(o2, c2)
        
        # First body must be significantly larger
        if body1 <= body2 * 1.5:
            return None
        
        # Check if second candle is contained within first
        top1 = max(o1, c1)
        bottom1 = min(o1, c1)
        top2 = max(o2, c2)
        bottom2 = min(o2, c2)
        
        if not (top2 < top1 and bottom2 > bottom1):
            return None
        
        # Bullish Harami
        if self._is_bearish(o1, c1) and self._is_bullish(o2, c2):
            return "bullish_harami"
        
        # Bearish Harami
        if self._is_bullish(o1, c1) and self._is_bearish(o2, c2):
            return "bearish_harami"
        
        # Harami Cross (second candle is doji)
        if body2 / max(body1, 0.001) < 0.1:
            if self._is_bearish(o1, c1):
                return "bullish_harami_cross"
            else:
                return "bearish_harami_cross"
        
        return None
    
    def detect_piercing_dark_cloud(self, o1: float, h1: float, l1: float, c1: float,
                                    o2: float, h2: float, l2: float, c2: float) -> Optional[str]:
        """
        Detect Piercing Line / Dark Cloud Cover.
        
        - Piercing Line: Bearish candle, then bullish that opens below and closes above midpoint
        - Dark Cloud: Bullish candle, then bearish that opens above and closes below midpoint
        """
        midpoint1 = (o1 + c1) / 2
        
        # Piercing Line
        if self._is_bearish(o1, c1) and self._is_bullish(o2, c2):
            if o2 < c1 and c2 > midpoint1 and c2 < o1:
                return "piercing_line"
        
        # Dark Cloud Cover
        if self._is_bullish(o1, c1) and self._is_bearish(o2, c2):
            if o2 > c1 and c2 < midpoint1 and c2 > o1:
                return "dark_cloud_cover"
        
        return None
    
    def detect_tweezer(self, o1: float, h1: float, l1: float, c1: float,
                       o2: float, h2: float, l2: float, c2: float) -> Optional[str]:
        """
        Detect Tweezer Top/Bottom (matching highs or lows).
        """
        tolerance = abs(h1 - l1) * 0.05  # 5% tolerance
        
        # Tweezer Bottom: matching lows
        if abs(l1 - l2) < tolerance:
            if self._is_bearish(o1, c1) and self._is_bullish(o2, c2):
                return "tweezer_bottom"
        
        # Tweezer Top: matching highs
        if abs(h1 - h2) < tolerance:
            if self._is_bullish(o1, c1) and self._is_bearish(o2, c2):
                return "tweezer_top"
        
        return None
    
    # ==========================================================================
    # THREE CANDLE PATTERNS
    # ==========================================================================
    
    def detect_morning_evening_star(self, 
                                     o1: float, h1: float, l1: float, c1: float,
                                     o2: float, h2: float, l2: float, c2: float,
                                     o3: float, h3: float, l3: float, c3: float) -> Optional[str]:
        """
        Detect Morning Star / Evening Star (3-candle reversal patterns).
        
        - Morning Star: Bearish, small body/doji, bullish (bullish reversal)
        - Evening Star: Bullish, small body/doji, bearish (bearish reversal)
        """
        body1 = self._body_size(o1, c1)
        body2 = self._body_size(o2, c2)
        body3 = self._body_size(o3, c3)
        
        # Middle candle should be small
        if body2 > min(body1, body3) * 0.5:
            return None
        
        # Morning Star
        if (self._is_bearish(o1, c1) and 
            self._is_bullish(o3, c3) and 
            c3 > (o1 + c1) / 2):  # Third candle closes above midpoint of first
            return "morning_star"
        
        # Evening Star
        if (self._is_bullish(o1, c1) and 
            self._is_bearish(o3, c3) and 
            c3 < (o1 + c1) / 2):  # Third candle closes below midpoint of first
            return "evening_star"
        
        return None
    
    def detect_three_soldiers_crows(self,
                                     o1: float, h1: float, l1: float, c1: float,
                                     o2: float, h2: float, l2: float, c2: float,
                                     o3: float, h3: float, l3: float, c3: float) -> Optional[str]:
        """
        Detect Three White Soldiers / Three Black Crows.
        
        - Three White Soldiers: Three consecutive bullish candles, each opening within prev body
        - Three Black Crows: Three consecutive bearish candles
        """
        # Three White Soldiers
        if (self._is_bullish(o1, c1) and 
            self._is_bullish(o2, c2) and 
            self._is_bullish(o3, c3)):
            # Each opens within previous body and closes higher
            if (o2 > o1 and o2 < c1 and c2 > c1 and
                o3 > o2 and o3 < c2 and c3 > c2):
                return "three_white_soldiers"
        
        # Three Black Crows
        if (self._is_bearish(o1, c1) and 
            self._is_bearish(o2, c2) and 
            self._is_bearish(o3, c3)):
            # Each opens within previous body and closes lower
            if (o2 < o1 and o2 > c1 and c2 < c1 and
                o3 < o2 and o3 > c2 and c3 < c2):
                return "three_black_crows"
        
        return None
    
    # ==========================================================================
    # MAIN DETECTION METHODS
    # ==========================================================================
    
    def detect_all_at_index(self, opens: np.ndarray, highs: np.ndarray, 
                            lows: np.ndarray, closes: np.ndarray,
                            index: int) -> List[PatternResult]:
        """
        Detect all patterns at a specific index.
        
        Args:
            opens, highs, lows, closes: Price arrays
            index: Index to check (must have at least 2 candles before it)
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        if index < 2:
            return patterns
        
        # Current candle
        o, h, l, c = opens[index], highs[index], lows[index], closes[index]
        
        # Previous candles
        o1, h1, l1, c1 = opens[index-1], highs[index-1], lows[index-1], closes[index-1]
        o2, h2, l2, c2 = opens[index-2], highs[index-2], lows[index-2], closes[index-2]
        
        # Determine trend (simple: compare to 5-period lookback)
        lookback = min(5, index)
        prev_trend = 'down' if closes[index] < closes[index - lookback] else 'up'
        
        # Single candle patterns
        doji = self.detect_doji(o, h, l, c)
        if doji:
            direction = 'neutral'
            if doji == 'dragonfly_doji':
                direction = 'bullish'
            elif doji == 'gravestone_doji':
                direction = 'bearish'
            patterns.append(PatternResult(doji, direction, 0.6, index, 0.7))
        
        hammer = self.detect_hammer(o, h, l, c, prev_trend)
        if hammer:
            direction = 'bullish' if hammer in ['hammer', 'inverted_hammer'] else 'bearish'
            patterns.append(PatternResult(hammer, direction, 0.7, index, 0.75))
        
        marubozu = self.detect_marubozu(o, h, l, c)
        if marubozu:
            direction = 'bullish' if 'bullish' in marubozu else 'bearish'
            patterns.append(PatternResult(marubozu, direction, 0.8, index, 0.8))
        
        if self.detect_spinning_top(o, h, l, c):
            patterns.append(PatternResult('spinning_top', 'neutral', 0.4, index, 0.6))
        
        # Two candle patterns
        engulfing = self.detect_engulfing(o1, h1, l1, c1, o, h, l, c)
        if engulfing:
            direction = 'bullish' if 'bullish' in engulfing else 'bearish'
            patterns.append(PatternResult(engulfing, direction, 0.85, index, 0.85))
        
        harami = self.detect_harami(o1, h1, l1, c1, o, h, l, c)
        if harami:
            direction = 'bullish' if 'bullish' in harami else 'bearish'
            patterns.append(PatternResult(harami, direction, 0.65, index, 0.7))
        
        piercing = self.detect_piercing_dark_cloud(o1, h1, l1, c1, o, h, l, c)
        if piercing:
            direction = 'bullish' if piercing == 'piercing_line' else 'bearish'
            patterns.append(PatternResult(piercing, direction, 0.75, index, 0.75))
        
        tweezer = self.detect_tweezer(o1, h1, l1, c1, o, h, l, c)
        if tweezer:
            direction = 'bullish' if 'bottom' in tweezer else 'bearish'
            patterns.append(PatternResult(tweezer, direction, 0.7, index, 0.7))
        
        # Three candle patterns
        star = self.detect_morning_evening_star(o2, h2, l2, c2, o1, h1, l1, c1, o, h, l, c)
        if star:
            direction = 'bullish' if 'morning' in star else 'bearish'
            patterns.append(PatternResult(star, direction, 0.9, index, 0.85))
        
        soldiers = self.detect_three_soldiers_crows(o2, h2, l2, c2, o1, h1, l1, c1, o, h, l, c)
        if soldiers:
            direction = 'bullish' if 'soldiers' in soldiers else 'bearish'
            patterns.append(PatternResult(soldiers, direction, 0.85, index, 0.8))
        
        return patterns
    
    def detect_all(self, opens: np.ndarray, highs: np.ndarray, 
                   lows: np.ndarray, closes: np.ndarray) -> Dict[int, List[PatternResult]]:
        """
        Detect all patterns across entire dataset.
        
        Returns:
            Dictionary mapping index to list of patterns found at that index
        """
        all_patterns = {}
        
        for i in range(2, len(opens)):
            patterns = self.detect_all_at_index(opens, highs, lows, closes, i)
            if patterns:
                all_patterns[i] = patterns
        
        return all_patterns
    
    def get_signal_at_index(self, opens: np.ndarray, highs: np.ndarray,
                            lows: np.ndarray, closes: np.ndarray,
                            index: int) -> Tuple[int, float, List[str]]:
        """
        Get aggregated signal at index based on patterns.
        
        Returns:
            Tuple of (signal, confidence, pattern_names)
            signal: 1 for bullish, -1 for bearish, 0 for neutral
        """
        patterns = self.detect_all_at_index(opens, highs, lows, closes, index)
        
        if not patterns:
            return 0, 0.0, []
        
        bullish_score = 0.0
        bearish_score = 0.0
        pattern_names = []
        
        for p in patterns:
            pattern_names.append(p.name)
            weight = p.strength * p.confidence
            
            if p.direction == 'bullish':
                bullish_score += weight
            elif p.direction == 'bearish':
                bearish_score += weight
        
        total_score = bullish_score + bearish_score
        if total_score == 0:
            return 0, 0.0, pattern_names
        
        if bullish_score > bearish_score * 1.2:  # 20% stronger
            signal = 1
            confidence = bullish_score / total_score
        elif bearish_score > bullish_score * 1.2:
            signal = -1
            confidence = bearish_score / total_score
        else:
            signal = 0
            confidence = 0.5
        
        return signal, confidence, pattern_names


def generate_pattern_signals(opens: np.ndarray, highs: np.ndarray,
                             lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """
    Generate pattern-based signals for entire dataset.
    
    Returns:
        Array of signals: 1 (bullish), -1 (bearish), 0 (neutral)
    """
    detector = CandlestickPatternDetector()
    signals = np.zeros(len(opens))
    
    for i in range(2, len(opens)):
        signal, confidence, _ = detector.get_signal_at_index(opens, highs, lows, closes, i)
        if confidence > 0.6:  # Only use high-confidence signals
            signals[i] = signal
    
    return signals


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("CANDLESTICK PATTERN RECOGNITION - DEMO")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n = 100
    
    # Simulate price data
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    opens = closes + np.random.randn(n) * 0.3
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 0.2)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 0.2)
    
    # Detect patterns
    detector = CandlestickPatternDetector()
    all_patterns = detector.detect_all(opens, highs, lows, closes)
    
    print(f"\nDetected {len(all_patterns)} bars with patterns\n")
    
    # Show recent patterns
    print("Recent Patterns (last 20 bars):")
    print("-" * 60)
    for idx in sorted(all_patterns.keys())[-10:]:
        patterns = all_patterns[idx]
        for p in patterns:
            print(f"Bar {idx:3d}: {p.name:25s} | {p.direction:8s} | "
                  f"Strength: {p.strength:.2f} | Conf: {p.confidence:.2f}")
    
    # Generate signals
    print("\n" + "=" * 60)
    print("SIGNAL SUMMARY")
    print("=" * 60)
    
    signals = generate_pattern_signals(opens, highs, lows, closes)
    bullish = np.sum(signals == 1)
    bearish = np.sum(signals == -1)
    neutral = np.sum(signals == 0)
    
    print(f"Bullish Signals: {bullish}")
    print(f"Bearish Signals: {bearish}")
    print(f"Neutral:         {neutral}")
    
    print("\n[OK] Candlestick Pattern Detection Module Working!")
