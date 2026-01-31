"""
Market Regime Detection Module
Uses ADX and other indicators to classify market conditions

Regimes:
1. Strong Trending (ADX > 25, clear direction)
2. Weak Trending (ADX 20-25, some direction)
3. Ranging/Sideways (ADX < 20)
4. High Volatility (ATR spike)
5. Low Volatility (ATR compression)

Strategy Implications:
- Strong Trend: Use trend-following strategies
- Ranging: Use mean-reversion strategies or avoid trading
- High Volatility: Widen stops, reduce position size
- Low Volatility: Expect breakout, tighten stops
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class MarketRegime(Enum):
    """Market regime classifications."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    WEAK_DOWNTREND = "weak_downtrend"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT_POTENTIAL = "breakout_potential"


@dataclass
class RegimeAnalysis:
    """Result of regime analysis."""
    regime: MarketRegime
    adx: float
    plus_di: float
    minus_di: float
    atr: float
    atr_percentile: float  # Where current ATR falls in historical range
    volatility_state: str  # 'high', 'normal', 'low'
    trend_direction: str  # 'bullish', 'bearish', 'neutral'
    trend_strength: str  # 'strong', 'weak', 'none'
    confidence: float  # 0-100
    strategy_recommendation: str
    position_size_multiplier: float  # 1.0 = normal, <1 = reduce, >1 = increase
    stop_loss_multiplier: float  # Multiply normal stop by this


class RegimeDetector:
    """
    Detects market regime using ADX, ATR, and price action.
    
    ADX Interpretation:
    - 0-20: Weak or no trend (ranging market)
    - 20-25: Weak trend starting
    - 25-50: Strong trend
    - 50-75: Very strong trend
    - 75-100: Extremely strong trend (rare)
    
    +DI vs -DI:
    - +DI > -DI: Bullish pressure
    - -DI > +DI: Bearish pressure
    - Crossovers signal trend changes
    """
    
    def __init__(self, adx_period: int = 14, atr_period: int = 14,
                 lookback_volatility: int = 100):
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.lookback_volatility = lookback_volatility
    
    def calculate_directional_movement(self, highs: List[float], lows: List[float], 
                                        closes: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """
        Calculate +DM, -DM, and True Range.
        """
        plus_dm = []
        minus_dm = []
        tr = []
        
        for i in range(1, len(highs)):
            # Directional Movement
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)
            
            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)
            
            # True Range
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
        
        return plus_dm, minus_dm, tr
    
    def smooth_data(self, data: List[float], period: int) -> List[float]:
        """Wilder's smoothing method."""
        if len(data) < period:
            return data
        
        # Initial sum
        smoothed = [sum(data[:period])]
        
        # Subsequent smoothing
        for i in range(period, len(data)):
            smoothed.append(smoothed[-1] - (smoothed[-1] / period) + data[i])
        
        return smoothed
    
    def calculate_adx(self, highs: List[float], lows: List[float], 
                      closes: List[float]) -> Tuple[float, float, float]:
        """
        Calculate ADX, +DI, and -DI.
        
        Returns:
            Tuple of (ADX, +DI, -DI)
        """
        if len(highs) < self.adx_period * 2:
            return 25, 25, 25  # Default neutral values
        
        plus_dm, minus_dm, tr = self.calculate_directional_movement(highs, lows, closes)
        
        # Smooth the values
        smoothed_plus_dm = self.smooth_data(plus_dm, self.adx_period)
        smoothed_minus_dm = self.smooth_data(minus_dm, self.adx_period)
        smoothed_tr = self.smooth_data(tr, self.adx_period)
        
        if len(smoothed_tr) == 0 or smoothed_tr[-1] == 0:
            return 25, 25, 25
        
        # Calculate +DI and -DI
        plus_di = (smoothed_plus_dm[-1] / smoothed_tr[-1]) * 100
        minus_di = (smoothed_minus_dm[-1] / smoothed_tr[-1]) * 100
        
        # Calculate DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx = 0
        else:
            dx = abs(plus_di - minus_di) / di_sum * 100
        
        # Calculate ADX (smoothed DX)
        # For simplicity, we'll use the current DX as ADX
        # In production, you'd smooth the DX values
        adx = dx
        
        return adx, plus_di, minus_di
    
    def calculate_atr(self, highs: List[float], lows: List[float], 
                      closes: List[float]) -> float:
        """Calculate current ATR."""
        if len(highs) < self.atr_period + 1:
            return 0
        
        tr = []
        for i in range(1, len(highs)):
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
        
        # Wilder's smoothing for ATR
        atr = sum(tr[:self.atr_period]) / self.atr_period
        
        for i in range(self.atr_period, len(tr)):
            atr = (atr * (self.atr_period - 1) + tr[i]) / self.atr_period
        
        return atr
    
    def calculate_atr_percentile(self, highs: List[float], lows: List[float], 
                                  closes: List[float]) -> float:
        """Calculate where current ATR falls in historical range (0-100)."""
        if len(highs) < self.lookback_volatility:
            return 50
        
        # Calculate ATR for each period in lookback
        atr_history = []
        
        for i in range(self.atr_period + 1, len(highs)):
            window_highs = highs[i-self.atr_period:i+1]
            window_lows = lows[i-self.atr_period:i+1]
            window_closes = closes[i-self.atr_period:i+1]
            
            atr = self.calculate_atr(window_highs, window_lows, window_closes)
            atr_history.append(atr)
        
        if len(atr_history) < 2:
            return 50
        
        current_atr = atr_history[-1]
        
        # Calculate percentile
        below_count = sum(1 for atr in atr_history if atr < current_atr)
        percentile = (below_count / len(atr_history)) * 100
        
        return percentile
    
    def detect_squeeze(self, highs: List[float], lows: List[float], 
                       closes: List[float], period: int = 20) -> bool:
        """
        Detect Bollinger Band squeeze (low volatility breakout potential).
        When Bollinger Bands are inside Keltner Channels = squeeze.
        """
        if len(closes) < period:
            return False
        
        # Bollinger Bands
        sma = sum(closes[-period:]) / period
        std = np.std(closes[-period:])
        bb_upper = sma + 2 * std
        bb_lower = sma - 2 * std
        
        # Keltner Channels (simplified with ATR)
        atr = self.calculate_atr(highs[-period-1:], lows[-period-1:], closes[-period-1:])
        kc_upper = sma + 1.5 * atr
        kc_lower = sma - 1.5 * atr
        
        # Squeeze when BB is inside KC
        squeeze = bb_upper < kc_upper and bb_lower > kc_lower
        
        return squeeze
    
    def analyze(self, data: List[Dict]) -> RegimeAnalysis:
        """
        Perform complete regime analysis.
        
        Args:
            data: List of OHLCV dictionaries
        
        Returns:
            RegimeAnalysis with current market regime and recommendations
        """
        if len(data) < max(self.adx_period * 2, self.lookback_volatility):
            return RegimeAnalysis(
                regime=MarketRegime.RANGING,
                adx=25,
                plus_di=25,
                minus_di=25,
                atr=0,
                atr_percentile=50,
                volatility_state='normal',
                trend_direction='neutral',
                trend_strength='none',
                confidence=0,
                strategy_recommendation='Insufficient data',
                position_size_multiplier=1.0,
                stop_loss_multiplier=1.0
            )
        
        highs = [d['high'] for d in data]
        lows = [d['low'] for d in data]
        closes = [d['close'] for d in data]
        
        # Calculate indicators
        adx, plus_di, minus_di = self.calculate_adx(highs, lows, closes)
        atr = self.calculate_atr(highs, lows, closes)
        atr_percentile = self.calculate_atr_percentile(highs, lows, closes)
        is_squeeze = self.detect_squeeze(highs, lows, closes)
        
        # Determine volatility state
        if atr_percentile > 80:
            volatility_state = 'high'
        elif atr_percentile < 20:
            volatility_state = 'low'
        else:
            volatility_state = 'normal'
        
        # Determine trend direction
        if plus_di > minus_di:
            trend_direction = 'bullish'
        elif minus_di > plus_di:
            trend_direction = 'bearish'
        else:
            trend_direction = 'neutral'
        
        # Determine trend strength
        if adx >= 40:
            trend_strength = 'strong'
        elif adx >= 25:
            trend_strength = 'moderate'
        elif adx >= 20:
            trend_strength = 'weak'
        else:
            trend_strength = 'none'
        
        # Determine regime
        regime = self._classify_regime(adx, plus_di, minus_di, atr_percentile, is_squeeze)
        
        # Generate recommendations
        recommendation, pos_mult, sl_mult = self._generate_recommendations(regime, volatility_state)
        
        # Calculate confidence
        confidence = self._calculate_confidence(adx, plus_di, minus_di, atr_percentile)
        
        return RegimeAnalysis(
            regime=regime,
            adx=adx,
            plus_di=plus_di,
            minus_di=minus_di,
            atr=atr,
            atr_percentile=atr_percentile,
            volatility_state=volatility_state,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            confidence=confidence,
            strategy_recommendation=recommendation,
            position_size_multiplier=pos_mult,
            stop_loss_multiplier=sl_mult
        )
    
    def _classify_regime(self, adx: float, plus_di: float, minus_di: float,
                         atr_percentile: float, is_squeeze: bool) -> MarketRegime:
        """Classify the current market regime."""
        
        # Check for squeeze first (potential breakout)
        if is_squeeze:
            return MarketRegime.BREAKOUT_POTENTIAL
        
        # Check for extreme volatility
        if atr_percentile > 85:
            return MarketRegime.HIGH_VOLATILITY
        elif atr_percentile < 15:
            return MarketRegime.LOW_VOLATILITY
        
        # Check trend strength
        if adx < 20:
            return MarketRegime.RANGING
        
        # Determine trend direction and strength
        is_bullish = plus_di > minus_di
        
        if adx >= 40:
            if is_bullish:
                return MarketRegime.STRONG_UPTREND
            else:
                return MarketRegime.STRONG_DOWNTREND
        else:  # 20-40 ADX
            if is_bullish:
                return MarketRegime.WEAK_UPTREND
            else:
                return MarketRegime.WEAK_DOWNTREND
    
    def _generate_recommendations(self, regime: MarketRegime, 
                                   volatility_state: str) -> Tuple[str, float, float]:
        """Generate strategy recommendations based on regime."""
        
        recommendations = {
            MarketRegime.STRONG_UPTREND: (
                "TREND FOLLOWING: Strong uptrend confirmed. Look for pullbacks to buy. "
                "Trail stops below recent swing lows. Hold positions longer.",
                1.2,  # Larger position
                0.8   # Tighter stops (trend is clear)
            ),
            MarketRegime.WEAK_UPTREND: (
                "CAUTIOUS LONG: Weak uptrend - trade smaller size. "
                "Take partial profits quickly. Watch for trend exhaustion.",
                0.8,  # Smaller position
                1.0   # Normal stops
            ),
            MarketRegime.STRONG_DOWNTREND: (
                "TREND FOLLOWING: Strong downtrend confirmed. Look for rallies to short. "
                "Trail stops above recent swing highs. Avoid bottom-picking.",
                1.2,
                0.8
            ),
            MarketRegime.WEAK_DOWNTREND: (
                "CAUTIOUS SHORT: Weak downtrend - trade smaller size. "
                "Take partial profits quickly. Watch for reversal signals.",
                0.8,
                1.0
            ),
            MarketRegime.RANGING: (
                "RANGE TRADING: Market is choppy. Trade support/resistance levels only. "
                "Reduce position size. Consider mean-reversion strategies or stay flat.",
                0.5,  # Much smaller position
                1.2   # Wider stops (more noise)
            ),
            MarketRegime.HIGH_VOLATILITY: (
                "HIGH VOLATILITY: Reduce position size significantly. Widen stops. "
                "Wait for volatility to normalize or trade very short-term only.",
                0.5,
                1.5   # Much wider stops
            ),
            MarketRegime.LOW_VOLATILITY: (
                "LOW VOLATILITY: Market compression detected. Prepare for breakout. "
                "Use tight stops. Position for directional move.",
                0.8,
                0.7   # Tighter stops
            ),
            MarketRegime.BREAKOUT_POTENTIAL: (
                "SQUEEZE DETECTED: Bollinger Bands inside Keltner. Breakout imminent. "
                "Wait for direction confirmation before entry. Be ready for fast move.",
                1.0,
                0.8
            )
        }
        
        rec, pos_mult, sl_mult = recommendations.get(
            regime, 
            ("Unable to determine regime", 1.0, 1.0)
        )
        
        # Adjust for volatility
        if volatility_state == 'high' and regime not in [MarketRegime.HIGH_VOLATILITY]:
            rec += " (CAUTION: Volatility elevated)"
            pos_mult *= 0.8
            sl_mult *= 1.2
        
        return rec, pos_mult, sl_mult
    
    def _calculate_confidence(self, adx: float, plus_di: float, 
                               minus_di: float, atr_percentile: float) -> float:
        """Calculate confidence in regime classification."""
        confidence = 50  # Base confidence
        
        # Higher ADX = more confident in trend classification
        if adx > 30:
            confidence += 20
        elif adx > 20:
            confidence += 10
        
        # Clear DI separation = more confident in direction
        di_diff = abs(plus_di - minus_di)
        if di_diff > 20:
            confidence += 15
        elif di_diff > 10:
            confidence += 8
        
        # Extreme ATR = more confident in volatility classification
        if atr_percentile > 80 or atr_percentile < 20:
            confidence += 10
        
        return min(100, confidence)
    
    def get_regime_history(self, data: List[Dict], lookback: int = 50) -> List[Tuple[str, MarketRegime]]:
        """
        Get regime classification for recent history.
        Useful for understanding regime transitions.
        """
        history = []
        
        min_data = max(self.adx_period * 2, self.lookback_volatility)
        
        for i in range(min_data, len(data)):
            window = data[:i+1]
            analysis = self.analyze(window[-self.lookback_volatility:])
            history.append((data[i].get('date', str(i)), analysis.regime))
        
        return history[-lookback:]


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for Regime Detection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Regime Detection')
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic data')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MARKET REGIME DETECTION")
    print("=" * 60)
    
    # Generate synthetic data with different regimes
    print("\nGenerating synthetic data with multiple regimes...")
    np.random.seed(42)
    
    data = []
    price = 100
    
    # Regime sequence: trending -> ranging -> high vol -> low vol -> breakout
    for i in range(500):
        if i < 100:
            # Strong uptrend
            trend = 0.3
            volatility = 0.2
        elif i < 200:
            # Ranging market
            trend = 0.02 * np.sin(i / 10)  # Oscillate
            volatility = 0.15
        elif i < 300:
            # High volatility
            trend = 0.1 * (-1 if i % 20 < 10 else 1)
            volatility = 0.8
        elif i < 400:
            # Low volatility (squeeze)
            trend = 0.01
            volatility = 0.05
        else:
            # Strong downtrend
            trend = -0.25
            volatility = 0.3
        
        noise = np.random.randn() * volatility
        change = trend + noise
        
        open_price = price
        close_price = price * (1 + change / 100)
        high_price = max(open_price, close_price) * (1 + abs(np.random.randn() * volatility) / 100)
        low_price = min(open_price, close_price) * (1 - abs(np.random.randn() * volatility) / 100)
        
        data.append({
            'date': f'2024-{(i//30)+1:02d}-{(i%30)+1:02d}',
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': int(np.random.uniform(100000, 500000))
        })
        
        price = close_price
    
    print(f"Generated {len(data)} bars of data")
    
    # Create detector
    detector = RegimeDetector()
    
    # Analyze current regime
    print("\n" + "=" * 60)
    print("CURRENT REGIME ANALYSIS")
    print("=" * 60)
    
    analysis = detector.analyze(data)
    
    print(f"\n🎯 REGIME: {analysis.regime.value.upper()}")
    print(f"\n📊 ADX: {analysis.adx:.1f}")
    print(f"   +DI: {analysis.plus_di:.1f}")
    print(f"   -DI: {analysis.minus_di:.1f}")
    print(f"\n📈 ATR: {analysis.atr:.2f}")
    print(f"   ATR Percentile: {analysis.atr_percentile:.0f}%")
    print(f"   Volatility: {analysis.volatility_state.upper()}")
    print(f"\n🔄 Trend: {analysis.trend_direction.upper()} ({analysis.trend_strength})")
    print(f"   Confidence: {analysis.confidence:.0f}%")
    
    print("\n" + "-" * 60)
    print("📋 STRATEGY RECOMMENDATION:")
    print(f"   {analysis.strategy_recommendation}")
    print(f"\n   Position Size: {analysis.position_size_multiplier:.1f}x normal")
    print(f"   Stop Loss: {analysis.stop_loss_multiplier:.1f}x normal")
    
    # Show regime history
    print("\n" + "=" * 60)
    print("REGIME HISTORY (Last 20 bars)")
    print("=" * 60)
    
    history = detector.get_regime_history(data, lookback=20)
    
    regime_colors = {
        MarketRegime.STRONG_UPTREND: "🟢",
        MarketRegime.WEAK_UPTREND: "🟡",
        MarketRegime.STRONG_DOWNTREND: "🔴",
        MarketRegime.WEAK_DOWNTREND: "🟠",
        MarketRegime.RANGING: "⚪",
        MarketRegime.HIGH_VOLATILITY: "⚡",
        MarketRegime.LOW_VOLATILITY: "💤",
        MarketRegime.BREAKOUT_POTENTIAL: "💥"
    }
    
    for date, regime in history:
        emoji = regime_colors.get(regime, "❓")
        print(f"  {date}: {emoji} {regime.value}")
    
    print("\n✅ Regime Detection complete!")


if __name__ == "__main__":
    main()
