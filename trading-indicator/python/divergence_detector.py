#!/usr/bin/env python3
"""
DIVERGENCE DETECTION MODULE
===========================
Detects RSI, MACD, and Stochastic divergences for reversal signals.

Types of Divergence:
- Regular Bullish: Price makes lower low, indicator makes higher low (reversal up)
- Regular Bearish: Price makes higher high, indicator makes lower high (reversal down)
- Hidden Bullish: Price makes higher low, indicator makes lower low (continuation up)
- Hidden Bearish: Price makes lower high, indicator makes higher high (continuation down)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DivergenceResult:
    """Result of divergence detection."""
    type: str           # 'regular_bullish', 'regular_bearish', 'hidden_bullish', 'hidden_bearish'
    indicator: str      # 'rsi', 'macd', 'stochastic'
    start_idx: int      # Starting index of divergence
    end_idx: int        # Ending index of divergence
    strength: float     # 0.0 to 1.0
    price_move: float   # Price change percentage
    indicator_move: float  # Indicator change


class TechnicalIndicators:
    """Calculate technical indicators for divergence analysis."""
    
    @staticmethod
    def calculate_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI."""
        deltas = np.diff(closes)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.zeros(len(closes))
        avg_loss = np.zeros(len(closes))
        
        # Initial SMA
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # EMA for subsequent values
        for i in range(period + 1, len(closes)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
        rsi = 100 - (100 / (1 + rs))
        rsi[:period] = 50  # Fill initial values
        
        return rsi
    
    @staticmethod
    def calculate_macd(closes: np.ndarray, fast: int = 12, slow: int = 26, 
                       signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD, Signal, and Histogram."""
        def ema(data, period):
            result = np.zeros_like(data)
            multiplier = 2 / (period + 1)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = (data[i] * multiplier) + (result[i-1] * (1 - multiplier))
            return result
        
        ema_fast = ema(closes, fast)
        ema_slow = ema(closes, slow)
        macd_line = ema_fast - ema_slow
        signal_line = ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def calculate_stochastic(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                             k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic %K and %D."""
        k = np.zeros(len(closes))
        
        for i in range(k_period - 1, len(closes)):
            highest_high = np.max(highs[i - k_period + 1:i + 1])
            lowest_low = np.min(lows[i - k_period + 1:i + 1])
            
            if highest_high != lowest_low:
                k[i] = 100 * (closes[i] - lowest_low) / (highest_high - lowest_low)
            else:
                k[i] = 50
        
        # %D is SMA of %K
        d = np.zeros(len(closes))
        for i in range(d_period - 1, len(closes)):
            d[i] = np.mean(k[i - d_period + 1:i + 1])
        
        return k, d


class DivergenceDetector:
    """
    Detects divergences between price and indicators.
    
    Usage:
        detector = DivergenceDetector()
        divergences = detector.detect_all(highs, lows, closes)
    """
    
    def __init__(self, lookback: int = 20, min_bars: int = 3, max_bars: int = 50):
        """
        Initialize divergence detector.
        
        Args:
            lookback: Bars to look back for pivot points
            min_bars: Minimum bars between pivots
            max_bars: Maximum bars between pivots
        """
        self.lookback = lookback
        self.min_bars = min_bars
        self.max_bars = max_bars
        self.indicators = TechnicalIndicators()
    
    def _find_pivot_highs(self, data: np.ndarray, left: int = 5, right: int = 5) -> List[int]:
        """Find pivot high indices."""
        pivots = []
        for i in range(left, len(data) - right):
            is_pivot = True
            for j in range(1, left + 1):
                if data[i] <= data[i - j]:
                    is_pivot = False
                    break
            for j in range(1, right + 1):
                if data[i] <= data[i + j]:
                    is_pivot = False
                    break
            if is_pivot:
                pivots.append(i)
        return pivots
    
    def _find_pivot_lows(self, data: np.ndarray, left: int = 5, right: int = 5) -> List[int]:
        """Find pivot low indices."""
        pivots = []
        for i in range(left, len(data) - right):
            is_pivot = True
            for j in range(1, left + 1):
                if data[i] >= data[i - j]:
                    is_pivot = False
                    break
            for j in range(1, right + 1):
                if data[i] >= data[i + j]:
                    is_pivot = False
                    break
            if is_pivot:
                pivots.append(i)
        return pivots
    
    def _detect_divergence_pair(self, price_pivot1: float, price_pivot2: float,
                                 ind_pivot1: float, ind_pivot2: float,
                                 is_high: bool) -> Optional[str]:
        """
        Detect divergence type between two pivot pairs.
        
        Args:
            price_pivot1, price_pivot2: Price at pivot points
            ind_pivot1, ind_pivot2: Indicator value at pivot points
            is_high: True if analyzing highs, False for lows
            
        Returns:
            Divergence type or None
        """
        price_higher = price_pivot2 > price_pivot1
        price_lower = price_pivot2 < price_pivot1
        ind_higher = ind_pivot2 > ind_pivot1
        ind_lower = ind_pivot2 < ind_pivot1
        
        if is_high:
            # Analyzing highs
            if price_higher and ind_lower:
                return 'regular_bearish'
            elif price_lower and ind_higher:
                return 'hidden_bearish'
        else:
            # Analyzing lows
            if price_lower and ind_higher:
                return 'regular_bullish'
            elif price_higher and ind_lower:
                return 'hidden_bullish'
        
        return None
    
    def detect_rsi_divergence(self, closes: np.ndarray, highs: np.ndarray = None, 
                              lows: np.ndarray = None) -> List[DivergenceResult]:
        """Detect RSI divergences."""
        if highs is None:
            highs = closes
        if lows is None:
            lows = closes
            
        rsi = self.indicators.calculate_rsi(closes)
        
        return self._detect_indicator_divergence(highs, lows, rsi, 'rsi')
    
    def detect_macd_divergence(self, closes: np.ndarray, highs: np.ndarray = None,
                               lows: np.ndarray = None) -> List[DivergenceResult]:
        """Detect MACD divergences (using histogram)."""
        if highs is None:
            highs = closes
        if lows is None:
            lows = closes
            
        _, _, macd_hist = self.indicators.calculate_macd(closes)
        
        return self._detect_indicator_divergence(highs, lows, macd_hist, 'macd')
    
    def detect_stochastic_divergence(self, closes: np.ndarray, highs: np.ndarray,
                                     lows: np.ndarray) -> List[DivergenceResult]:
        """Detect Stochastic divergences."""
        stoch_k, _ = self.indicators.calculate_stochastic(highs, lows, closes)
        
        return self._detect_indicator_divergence(highs, lows, stoch_k, 'stochastic')
    
    def _detect_indicator_divergence(self, highs: np.ndarray, lows: np.ndarray,
                                      indicator: np.ndarray, 
                                      indicator_name: str) -> List[DivergenceResult]:
        """
        Generic divergence detection for any indicator.
        """
        divergences = []
        
        # Find pivots
        price_pivot_highs = self._find_pivot_highs(highs)
        price_pivot_lows = self._find_pivot_lows(lows)
        ind_pivot_highs = self._find_pivot_highs(indicator)
        ind_pivot_lows = self._find_pivot_lows(indicator)
        
        # Check high pivots for bearish divergence
        for i in range(len(price_pivot_highs) - 1):
            idx1 = price_pivot_highs[i]
            idx2 = price_pivot_highs[i + 1]
            
            # Check bar count constraint
            if idx2 - idx1 < self.min_bars or idx2 - idx1 > self.max_bars:
                continue
            
            # Find nearest indicator pivot highs
            ind_idx1 = min(ind_pivot_highs, key=lambda x: abs(x - idx1), default=None)
            ind_idx2 = min(ind_pivot_highs, key=lambda x: abs(x - idx2), default=None)
            
            if ind_idx1 is None or ind_idx2 is None:
                continue
            
            # Check if indicator pivots are close enough to price pivots
            if abs(ind_idx1 - idx1) > 3 or abs(ind_idx2 - idx2) > 3:
                continue
            
            div_type = self._detect_divergence_pair(
                highs[idx1], highs[idx2],
                indicator[ind_idx1], indicator[ind_idx2],
                is_high=True
            )
            
            if div_type:
                price_move = (highs[idx2] - highs[idx1]) / highs[idx1] * 100
                ind_move = indicator[ind_idx2] - indicator[ind_idx1]
                
                # Calculate strength based on divergence magnitude
                strength = min(1.0, abs(price_move - ind_move) / 10)
                
                divergences.append(DivergenceResult(
                    type=div_type,
                    indicator=indicator_name,
                    start_idx=idx1,
                    end_idx=idx2,
                    strength=strength,
                    price_move=price_move,
                    indicator_move=ind_move
                ))
        
        # Check low pivots for bullish divergence
        for i in range(len(price_pivot_lows) - 1):
            idx1 = price_pivot_lows[i]
            idx2 = price_pivot_lows[i + 1]
            
            if idx2 - idx1 < self.min_bars or idx2 - idx1 > self.max_bars:
                continue
            
            ind_idx1 = min(ind_pivot_lows, key=lambda x: abs(x - idx1), default=None)
            ind_idx2 = min(ind_pivot_lows, key=lambda x: abs(x - idx2), default=None)
            
            if ind_idx1 is None or ind_idx2 is None:
                continue
            
            if abs(ind_idx1 - idx1) > 3 or abs(ind_idx2 - idx2) > 3:
                continue
            
            div_type = self._detect_divergence_pair(
                lows[idx1], lows[idx2],
                indicator[ind_idx1], indicator[ind_idx2],
                is_high=False
            )
            
            if div_type:
                price_move = (lows[idx2] - lows[idx1]) / lows[idx1] * 100
                ind_move = indicator[ind_idx2] - indicator[ind_idx1]
                strength = min(1.0, abs(price_move - ind_move) / 10)
                
                divergences.append(DivergenceResult(
                    type=div_type,
                    indicator=indicator_name,
                    start_idx=idx1,
                    end_idx=idx2,
                    strength=strength,
                    price_move=price_move,
                    indicator_move=ind_move
                ))
        
        return divergences
    
    def detect_all(self, highs: np.ndarray, lows: np.ndarray, 
                   closes: np.ndarray) -> Dict[str, List[DivergenceResult]]:
        """
        Detect all types of divergences.
        
        Returns:
            Dictionary with keys 'rsi', 'macd', 'stochastic'
        """
        return {
            'rsi': self.detect_rsi_divergence(closes, highs, lows),
            'macd': self.detect_macd_divergence(closes, highs, lows),
            'stochastic': self.detect_stochastic_divergence(closes, highs, lows)
        }
    
    def get_signal_at_index(self, highs: np.ndarray, lows: np.ndarray,
                            closes: np.ndarray, index: int,
                            lookback: int = 5) -> Tuple[int, float, List[str]]:
        """
        Get divergence signal at specific index.
        
        Args:
            index: Target index
            lookback: Bars to look back for recent divergences
            
        Returns:
            Tuple of (signal, confidence, divergence_types)
        """
        all_divs = self.detect_all(highs, lows, closes)
        
        bullish_score = 0.0
        bearish_score = 0.0
        div_types = []
        
        for indicator, divs in all_divs.items():
            for div in divs:
                # Check if divergence ends near our target index
                if index - lookback <= div.end_idx <= index:
                    div_types.append(f"{indicator}_{div.type}")
                    
                    if 'bullish' in div.type:
                        # Regular bullish is stronger
                        weight = 1.0 if 'regular' in div.type else 0.7
                        bullish_score += div.strength * weight
                    else:
                        weight = 1.0 if 'regular' in div.type else 0.7
                        bearish_score += div.strength * weight
        
        total = bullish_score + bearish_score
        if total == 0:
            return 0, 0.0, []
        
        if bullish_score > bearish_score * 1.3:
            return 1, bullish_score / (bullish_score + bearish_score), div_types
        elif bearish_score > bullish_score * 1.3:
            return -1, bearish_score / (bullish_score + bearish_score), div_types
        
        return 0, 0.5, div_types


def generate_divergence_signals(highs: np.ndarray, lows: np.ndarray,
                                closes: np.ndarray) -> np.ndarray:
    """
    Generate divergence-based signals for entire dataset.
    
    Returns:
        Array of signals: 1 (bullish), -1 (bearish), 0 (neutral)
    """
    detector = DivergenceDetector()
    signals = np.zeros(len(closes))
    
    # Get all divergences
    all_divs = detector.detect_all(highs, lows, closes)
    
    # Mark signals at divergence endpoints
    for indicator, divs in all_divs.items():
        for div in divs:
            if 'bullish' in div.type and div.strength > 0.5:
                signals[div.end_idx] = max(signals[div.end_idx], 1)
            elif 'bearish' in div.type and div.strength > 0.5:
                signals[div.end_idx] = min(signals[div.end_idx], -1)
    
    return signals


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("DIVERGENCE DETECTION MODULE - DEMO")
    print("=" * 60)
    
    # Generate sample data with trend
    np.random.seed(42)
    n = 200
    
    # Create trending data with pullbacks
    trend = np.linspace(0, 20, n)
    noise = np.cumsum(np.random.randn(n) * 0.5)
    closes = 100 + trend + noise
    
    # Add cycles for divergence opportunities
    cycles = 3 * np.sin(np.linspace(0, 8 * np.pi, n))
    closes += cycles
    
    highs = closes + np.abs(np.random.randn(n) * 0.3)
    lows = closes - np.abs(np.random.randn(n) * 0.3)
    
    # Detect divergences
    detector = DivergenceDetector()
    all_divergences = detector.detect_all(highs, lows, closes)
    
    # Print results
    total_divs = sum(len(divs) for divs in all_divergences.values())
    print(f"\nTotal Divergences Detected: {total_divs}")
    
    print("\n" + "-" * 60)
    print("DIVERGENCES BY INDICATOR")
    print("-" * 60)
    
    for indicator, divs in all_divergences.items():
        print(f"\n{indicator.upper()} Divergences: {len(divs)}")
        for div in divs[:5]:  # Show first 5
            print(f"  {div.type:20s} | Bars {div.start_idx:3d}-{div.end_idx:3d} | "
                  f"Strength: {div.strength:.2f} | Price: {div.price_move:+.2f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("DIVERGENCE SUMMARY")
    print("=" * 60)
    
    bullish_regular = sum(1 for divs in all_divergences.values() 
                         for d in divs if d.type == 'regular_bullish')
    bearish_regular = sum(1 for divs in all_divergences.values() 
                         for d in divs if d.type == 'regular_bearish')
    bullish_hidden = sum(1 for divs in all_divergences.values() 
                        for d in divs if d.type == 'hidden_bullish')
    bearish_hidden = sum(1 for divs in all_divergences.values() 
                        for d in divs if d.type == 'hidden_bearish')
    
    print(f"Regular Bullish: {bullish_regular}")
    print(f"Regular Bearish: {bearish_regular}")
    print(f"Hidden Bullish:  {bullish_hidden}")
    print(f"Hidden Bearish:  {bearish_hidden}")
    
    # Generate signals
    signals = generate_divergence_signals(highs, lows, closes)
    buy_signals = np.sum(signals == 1)
    sell_signals = np.sum(signals == -1)
    
    print(f"\nBuy Signals:  {buy_signals}")
    print(f"Sell Signals: {sell_signals}")
    
    print("\n[OK] Divergence Detection Module Working!")
