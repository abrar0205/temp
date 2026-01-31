"""
Multi-Timeframe Analysis Module
Aligns signals across multiple timeframes for higher probability trades

Concept:
- Higher timeframes define the trend direction
- Lower timeframes provide entry timing
- Signals are stronger when all timeframes agree
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Timeframe(Enum):
    """Supported timeframes."""
    M5 = "5m"      # 5 minutes
    M15 = "15m"    # 15 minutes  
    H1 = "1h"      # 1 hour
    H4 = "4h"      # 4 hours
    D1 = "1d"      # 1 day
    W1 = "1w"      # 1 week


@dataclass
class TimeframeSignal:
    """Signal from a single timeframe."""
    timeframe: Timeframe
    trend: str  # 'bullish', 'bearish', 'neutral'
    trend_strength: float  # 0-100
    rsi: float
    ema_alignment: str  # 'bullish', 'bearish', 'mixed'
    supertrend_direction: int  # 1 = bullish, -1 = bearish
    volume_trend: str  # 'increasing', 'decreasing', 'neutral'
    key_level_nearby: bool  # Near support/resistance


@dataclass
class MTFSignal:
    """Multi-timeframe combined signal."""
    signal: str  # 'strong_long', 'long', 'neutral', 'short', 'strong_short'
    confidence: float  # 0-100
    alignment_score: float  # How well timeframes agree (0-100)
    timeframe_signals: Dict[str, TimeframeSignal]
    entry_timeframe: Timeframe
    trend_timeframe: Timeframe
    reasoning: List[str]


class MultiTimeframeAnalyzer:
    """
    Analyzes price action across multiple timeframes.
    
    Strategy:
    1. Daily/4H determines overall trend
    2. 1H confirms intermediate trend
    3. 15M/5M provides entry timing
    
    Signal Strength:
    - All timeframes agree: Strong signal (80-100% confidence)
    - Entry + Trend agree: Medium signal (60-80%)
    - Conflicting signals: Weak/Neutral (0-60%)
    """
    
    def __init__(self, entry_tf: Timeframe = Timeframe.M15, 
                 trend_tf: Timeframe = Timeframe.H4):
        self.entry_tf = entry_tf
        self.trend_tf = trend_tf
        
        # Timeframe hierarchy (higher index = higher timeframe)
        self.tf_hierarchy = {
            Timeframe.M5: 1,
            Timeframe.M15: 2,
            Timeframe.H1: 3,
            Timeframe.H4: 4,
            Timeframe.D1: 5,
            Timeframe.W1: 6
        }
    
    def resample_data(self, data: List[Dict], source_tf: Timeframe, 
                      target_tf: Timeframe) -> List[Dict]:
        """
        Resample data from lower timeframe to higher timeframe.
        
        For simplicity, this uses bar count approximation:
        - 5m to 15m: aggregate 3 bars
        - 15m to 1h: aggregate 4 bars
        - 1h to 4h: aggregate 4 bars
        - 4h to 1d: aggregate 6 bars
        - 1d to 1w: aggregate 5 bars
        """
        ratios = {
            (Timeframe.M5, Timeframe.M15): 3,
            (Timeframe.M5, Timeframe.H1): 12,
            (Timeframe.M5, Timeframe.H4): 48,
            (Timeframe.M5, Timeframe.D1): 288,
            (Timeframe.M15, Timeframe.H1): 4,
            (Timeframe.M15, Timeframe.H4): 16,
            (Timeframe.M15, Timeframe.D1): 96,
            (Timeframe.H1, Timeframe.H4): 4,
            (Timeframe.H1, Timeframe.D1): 24,
            (Timeframe.H4, Timeframe.D1): 6,
            (Timeframe.D1, Timeframe.W1): 5,
        }
        
        ratio = ratios.get((source_tf, target_tf), 1)
        
        if ratio == 1:
            return data
        
        resampled = []
        
        for i in range(0, len(data) - ratio + 1, ratio):
            window = data[i:i + ratio]
            
            resampled.append({
                'date': window[-1]['date'],
                'open': window[0]['open'],
                'high': max(d['high'] for d in window),
                'low': min(d['low'] for d in window),
                'close': window[-1]['close'],
                'volume': sum(d['volume'] for d in window)
            })
        
        return resampled
    
    def calculate_ema(self, closes: List[float], period: int) -> List[float]:
        """Calculate EMA."""
        if len(closes) < period:
            return [closes[0]] * len(closes)
        
        ema = [sum(closes[:period]) / period]
        multiplier = 2 / (period + 1)
        
        for price in closes[period:]:
            ema.append((price * multiplier) + (ema[-1] * (1 - multiplier)))
        
        # Pad to match input length
        return [ema[0]] * (period - 1) + ema
    
    def calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """Calculate RSI for the most recent bar."""
        if len(closes) < period + 1:
            return 50
        
        changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [max(c, 0) for c in changes[-period:]]
        losses = [abs(min(c, 0)) for c in changes[-period:]]
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_supertrend(self, highs: List[float], lows: List[float], 
                            closes: List[float], period: int = 10, 
                            multiplier: float = 3.0) -> int:
        """Calculate SuperTrend direction (1 = bullish, -1 = bearish)."""
        if len(closes) < period + 1:
            return 1
        
        # Calculate ATR
        tr = []
        for i in range(1, len(closes)):
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
        
        atr = sum(tr[-period:]) / period
        
        # Calculate bands
        hl2 = (highs[-1] + lows[-1]) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Determine direction based on close vs bands
        if closes[-1] > upper_band:
            return 1
        elif closes[-1] < lower_band:
            return -1
        else:
            # Use previous close comparison
            if closes[-1] > closes[-2]:
                return 1
            else:
                return -1
    
    def analyze_timeframe(self, data: List[Dict], timeframe: Timeframe) -> TimeframeSignal:
        """Analyze a single timeframe."""
        if len(data) < 50:
            return TimeframeSignal(
                timeframe=timeframe,
                trend='neutral',
                trend_strength=0,
                rsi=50,
                ema_alignment='mixed',
                supertrend_direction=0,
                volume_trend='neutral',
                key_level_nearby=False
            )
        
        closes = [d['close'] for d in data]
        highs = [d['high'] for d in data]
        lows = [d['low'] for d in data]
        volumes = [d['volume'] for d in data]
        
        # Calculate indicators
        ema9 = self.calculate_ema(closes, 9)
        ema21 = self.calculate_ema(closes, 21)
        ema50 = self.calculate_ema(closes, 50)
        
        rsi = self.calculate_rsi(closes, 14)
        supertrend_dir = self.calculate_supertrend(highs, lows, closes)
        
        # Determine trend
        price = closes[-1]
        ema9_val = ema9[-1]
        ema21_val = ema21[-1]
        ema50_val = ema50[-1]
        
        # EMA alignment
        if ema9_val > ema21_val > ema50_val:
            ema_alignment = 'bullish'
        elif ema9_val < ema21_val < ema50_val:
            ema_alignment = 'bearish'
        else:
            ema_alignment = 'mixed'
        
        # Trend direction
        if price > ema21_val and ema9_val > ema21_val and supertrend_dir == 1:
            trend = 'bullish'
        elif price < ema21_val and ema9_val < ema21_val and supertrend_dir == -1:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        # Trend strength (0-100)
        # Based on distance from EMA and RSI extremity
        ema_distance = abs(price - ema21_val) / ema21_val * 100
        rsi_strength = abs(rsi - 50) / 50 * 100
        trend_strength = min((ema_distance * 5 + rsi_strength) / 2, 100)
        
        # Volume trend
        vol_current = volumes[-1]
        vol_avg = sum(volumes[-20:]) / 20
        
        if vol_current > vol_avg * 1.2:
            volume_trend = 'increasing'
        elif vol_current < vol_avg * 0.8:
            volume_trend = 'decreasing'
        else:
            volume_trend = 'neutral'
        
        # Key level detection (simplified - checks recent highs/lows)
        recent_high = max(highs[-20:])
        recent_low = min(lows[-20:])
        threshold = (recent_high - recent_low) * 0.05
        
        key_level_nearby = (abs(price - recent_high) < threshold or 
                           abs(price - recent_low) < threshold)
        
        return TimeframeSignal(
            timeframe=timeframe,
            trend=trend,
            trend_strength=trend_strength,
            rsi=rsi,
            ema_alignment=ema_alignment,
            supertrend_direction=supertrend_dir,
            volume_trend=volume_trend,
            key_level_nearby=key_level_nearby
        )
    
    def analyze(self, data_by_tf: Dict[Timeframe, List[Dict]]) -> MTFSignal:
        """
        Perform multi-timeframe analysis.
        
        Args:
            data_by_tf: Dictionary mapping timeframes to their price data
        
        Returns:
            MTFSignal with combined analysis
        """
        # Analyze each timeframe
        tf_signals = {}
        
        for tf, data in data_by_tf.items():
            tf_signals[tf.value] = self.analyze_timeframe(data, tf)
        
        # Calculate alignment
        bullish_count = sum(1 for s in tf_signals.values() if s.trend == 'bullish')
        bearish_count = sum(1 for s in tf_signals.values() if s.trend == 'bearish')
        total = len(tf_signals)
        
        alignment_score = max(bullish_count, bearish_count) / total * 100 if total > 0 else 0
        
        # Determine overall signal
        reasoning = []
        
        # Get entry and trend timeframe signals
        entry_signal = tf_signals.get(self.entry_tf.value)
        trend_signal = tf_signals.get(self.trend_tf.value)
        
        if entry_signal is None or trend_signal is None:
            return MTFSignal(
                signal='neutral',
                confidence=0,
                alignment_score=0,
                timeframe_signals=tf_signals,
                entry_timeframe=self.entry_tf,
                trend_timeframe=self.trend_tf,
                reasoning=['Insufficient timeframe data']
            )
        
        # Rule 1: Trend timeframe determines direction
        if trend_signal.trend == 'bullish':
            reasoning.append(f"✓ {self.trend_tf.value} trend is BULLISH")
            base_direction = 'long'
        elif trend_signal.trend == 'bearish':
            reasoning.append(f"✓ {self.trend_tf.value} trend is BEARISH")
            base_direction = 'short'
        else:
            reasoning.append(f"⚠ {self.trend_tf.value} trend is NEUTRAL - be cautious")
            base_direction = 'neutral'
        
        # Rule 2: Entry timeframe must align
        if base_direction != 'neutral':
            expected_entry_trend = 'bullish' if base_direction == 'long' else 'bearish'
            
            if entry_signal.trend == expected_entry_trend:
                reasoning.append(f"✓ {self.entry_tf.value} confirms direction")
            else:
                reasoning.append(f"⚠ {self.entry_tf.value} does NOT confirm - wait for alignment")
                base_direction = 'neutral'
        
        # Rule 3: Calculate confidence based on alignment
        confidence = 0
        
        if base_direction != 'neutral':
            # Base confidence from alignment
            confidence = alignment_score * 0.5
            
            # Bonus for strong trend
            if trend_signal.trend_strength > 50:
                confidence += 15
                reasoning.append(f"✓ Strong trend (strength: {trend_signal.trend_strength:.0f}%)")
            
            # Bonus for volume confirmation
            expected_volume = 'increasing'
            if entry_signal.volume_trend == expected_volume:
                confidence += 10
                reasoning.append("✓ Volume confirms move")
            
            # Bonus for RSI not extreme
            if 30 < entry_signal.rsi < 70:
                confidence += 10
                reasoning.append(f"✓ RSI in healthy zone ({entry_signal.rsi:.0f})")
            else:
                reasoning.append(f"⚠ RSI extreme ({entry_signal.rsi:.0f}) - potential reversal")
                confidence -= 10
            
            # Penalty for key level nearby (potential reversal zone)
            if entry_signal.key_level_nearby:
                reasoning.append("⚠ Near key level - watch for reversal")
                confidence -= 5
        
        confidence = max(0, min(100, confidence))
        
        # Determine signal strength
        if base_direction == 'neutral':
            signal = 'neutral'
        elif confidence >= 75:
            signal = f'strong_{base_direction}'
        elif confidence >= 50:
            signal = base_direction
        else:
            signal = 'neutral'
            reasoning.append("Confidence too low for signal")
        
        return MTFSignal(
            signal=signal,
            confidence=confidence,
            alignment_score=alignment_score,
            timeframe_signals=tf_signals,
            entry_timeframe=self.entry_tf,
            trend_timeframe=self.trend_tf,
            reasoning=reasoning
        )
    
    def analyze_from_single_data(self, data: List[Dict], 
                                  base_tf: Timeframe = Timeframe.M15) -> MTFSignal:
        """
        Analyze using data from a single timeframe, resampling as needed.
        
        This is useful when you only have one timeframe's data and want
        to simulate multi-timeframe analysis.
        """
        # Define timeframes to analyze
        timeframes_to_analyze = [Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1]
        
        data_by_tf = {}
        
        for tf in timeframes_to_analyze:
            if self.tf_hierarchy[tf] >= self.tf_hierarchy[base_tf]:
                # Resample to higher timeframe
                resampled = self.resample_data(data, base_tf, tf)
                if len(resampled) > 0:
                    data_by_tf[tf] = resampled
            else:
                # Can't go to lower timeframe
                continue
        
        # Always include base timeframe
        data_by_tf[base_tf] = data
        
        return self.analyze(data_by_tf)


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for Multi-Timeframe Analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Timeframe Analysis')
    parser.add_argument('--demo', action='store_true', help='Run demo with synthetic data')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MULTI-TIMEFRAME ANALYSIS")
    print("=" * 60)
    
    # Generate synthetic data
    print("\nGenerating synthetic price data (simulating 15m bars)...")
    np.random.seed(42)
    
    data = []
    price = 100
    trend = 1  # Start bullish
    
    for i in range(1000):  # ~10 days of 15m data
        # Change trend periodically
        if i % 200 == 0:
            trend *= -1
        
        noise = np.random.randn() * 0.3
        change = trend * 0.05 + noise
        
        open_price = price
        close_price = price * (1 + change / 100)
        high_price = max(open_price, close_price) * (1 + abs(np.random.randn() * 0.1) / 100)
        low_price = min(open_price, close_price) * (1 - abs(np.random.randn() * 0.1) / 100)
        
        data.append({
            'date': f'2024-01-{(i//96)+1:02d} {(i%96)*15//60:02d}:{(i%96)*15%60:02d}',
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': int(np.random.uniform(100000, 500000))
        })
        
        price = close_price
    
    print(f"Generated {len(data)} bars of 15m data")
    
    # Create analyzer
    analyzer = MultiTimeframeAnalyzer(
        entry_tf=Timeframe.M15,
        trend_tf=Timeframe.H4
    )
    
    # Perform analysis
    print("\n" + "=" * 60)
    print("ANALYZING MULTIPLE TIMEFRAMES")
    print("=" * 60)
    
    result = analyzer.analyze_from_single_data(data, Timeframe.M15)
    
    print(f"\n📊 SIGNAL: {result.signal.upper()}")
    print(f"📈 Confidence: {result.confidence:.1f}%")
    print(f"🔄 Alignment Score: {result.alignment_score:.1f}%")
    
    print("\n--- Timeframe Breakdown ---")
    for tf_name, signal in result.timeframe_signals.items():
        trend_emoji = "🟢" if signal.trend == 'bullish' else "🔴" if signal.trend == 'bearish' else "⚪"
        print(f"  {tf_name:5s}: {trend_emoji} {signal.trend:8s} | RSI: {signal.rsi:5.1f} | "
              f"ST: {'↑' if signal.supertrend_direction == 1 else '↓'} | "
              f"Vol: {signal.volume_trend}")
    
    print("\n--- Reasoning ---")
    for reason in result.reasoning:
        print(f"  {reason}")
    
    print("\n✅ Multi-Timeframe Analysis complete!")


if __name__ == "__main__":
    main()
