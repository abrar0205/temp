#!/usr/bin/env python3
"""
SMART MONEY CONCEPTS (SMC) MODULE
=================================
Implements institutional trading concepts:

1. Fair Value Gaps (FVG) - Imbalances where price moved too fast
2. Order Blocks (OB) - Institutional entry zones
3. Liquidity Zones - Areas with stop losses (targets for smart money)
4. Break of Structure (BOS) - Trend continuation signal
5. Change of Character (CHOCH) - Trend reversal signal
6. Premium/Discount Zones - Based on Fibonacci retracement

Based on ICT (Inner Circle Trader) methodology.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ZoneType(Enum):
    """Types of SMC zones."""
    BULLISH_FVG = "bullish_fvg"
    BEARISH_FVG = "bearish_fvg"
    BULLISH_OB = "bullish_ob"
    BEARISH_OB = "bearish_ob"
    BUY_LIQUIDITY = "buy_liquidity"
    SELL_LIQUIDITY = "sell_liquidity"
    BOS_BULLISH = "bos_bullish"
    BOS_BEARISH = "bos_bearish"
    CHOCH_BULLISH = "choch_bullish"
    CHOCH_BEARISH = "choch_bearish"


@dataclass
class SMCZone:
    """Represents a Smart Money Concept zone."""
    zone_type: ZoneType
    start_idx: int
    end_idx: int
    upper_price: float
    lower_price: float
    strength: float
    is_mitigated: bool = False
    mitigation_idx: Optional[int] = None


class SmartMoneyAnalyzer:
    """
    Analyzes price action using Smart Money Concepts.
    
    Usage:
        analyzer = SmartMoneyAnalyzer()
        zones = analyzer.analyze(opens, highs, lows, closes)
    """
    
    def __init__(self, swing_lookback: int = 10, fvg_min_size: float = 0.001):
        """
        Initialize analyzer.
        
        Args:
            swing_lookback: Bars to look back for swing points
            fvg_min_size: Minimum FVG size as % of price
        """
        self.swing_lookback = swing_lookback
        self.fvg_min_size = fvg_min_size
    
    # ==========================================================================
    # SWING POINT DETECTION
    # ==========================================================================
    
    def find_swing_highs(self, highs: np.ndarray, left: int = 5, 
                         right: int = 5) -> List[int]:
        """Find swing high indices."""
        swings = []
        for i in range(left, len(highs) - right):
            is_swing = True
            for j in range(1, left + 1):
                if highs[i] < highs[i - j]:
                    is_swing = False
                    break
            if is_swing:
                for j in range(1, right + 1):
                    if highs[i] < highs[i + j]:
                        is_swing = False
                        break
            if is_swing:
                swings.append(i)
        return swings
    
    def find_swing_lows(self, lows: np.ndarray, left: int = 5, 
                        right: int = 5) -> List[int]:
        """Find swing low indices."""
        swings = []
        for i in range(left, len(lows) - right):
            is_swing = True
            for j in range(1, left + 1):
                if lows[i] > lows[i - j]:
                    is_swing = False
                    break
            if is_swing:
                for j in range(1, right + 1):
                    if lows[i] > lows[i + j]:
                        is_swing = False
                        break
            if is_swing:
                swings.append(i)
        return swings
    
    # ==========================================================================
    # FAIR VALUE GAPS (FVG)
    # ==========================================================================
    
    def detect_fvg(self, opens: np.ndarray, highs: np.ndarray, 
                   lows: np.ndarray, closes: np.ndarray) -> List[SMCZone]:
        """
        Detect Fair Value Gaps (imbalances).
        
        Bullish FVG: Gap between candle 1's high and candle 3's low
        Bearish FVG: Gap between candle 1's low and candle 3's high
        """
        fvgs = []
        
        for i in range(2, len(closes)):
            # Bullish FVG: Candle 3's low > Candle 1's high
            if lows[i] > highs[i-2]:
                gap_size = lows[i] - highs[i-2]
                if gap_size / closes[i] > self.fvg_min_size:
                    # Check if middle candle is bullish (strong FVG)
                    strength = 0.7
                    if closes[i-1] > opens[i-1]:
                        strength = 0.9
                    
                    fvgs.append(SMCZone(
                        zone_type=ZoneType.BULLISH_FVG,
                        start_idx=i-2,
                        end_idx=i,
                        upper_price=lows[i],
                        lower_price=highs[i-2],
                        strength=strength
                    ))
            
            # Bearish FVG: Candle 3's high < Candle 1's low
            if highs[i] < lows[i-2]:
                gap_size = lows[i-2] - highs[i]
                if gap_size / closes[i] > self.fvg_min_size:
                    strength = 0.7
                    if closes[i-1] < opens[i-1]:
                        strength = 0.9
                    
                    fvgs.append(SMCZone(
                        zone_type=ZoneType.BEARISH_FVG,
                        start_idx=i-2,
                        end_idx=i,
                        upper_price=lows[i-2],
                        lower_price=highs[i],
                        strength=strength
                    ))
        
        return fvgs
    
    # ==========================================================================
    # ORDER BLOCKS (OB)
    # ==========================================================================
    
    def detect_order_blocks(self, opens: np.ndarray, highs: np.ndarray,
                            lows: np.ndarray, closes: np.ndarray) -> List[SMCZone]:
        """
        Detect Order Blocks (last opposite candle before impulse move).
        
        Bullish OB: Last bearish candle before strong bullish move
        Bearish OB: Last bullish candle before strong bearish move
        """
        obs = []
        
        # Calculate ATR for impulse detection
        tr = np.maximum(highs - lows, 
                       np.maximum(np.abs(highs - np.roll(closes, 1)),
                                 np.abs(lows - np.roll(closes, 1))))
        atr = np.zeros(len(closes))
        for i in range(14, len(closes)):
            atr[i] = np.mean(tr[i-14:i])
        
        for i in range(3, len(closes) - 1):
            # Skip if ATR not calculated yet
            if atr[i] == 0:
                continue
            
            # Check for impulse move (price moved > 2 ATR)
            move = closes[i] - closes[i-3]
            
            # Bullish impulse
            if move > 2 * atr[i]:
                # Find last bearish candle before impulse
                for j in range(i-1, max(i-10, 0), -1):
                    if closes[j] < opens[j]:  # Bearish candle
                        obs.append(SMCZone(
                            zone_type=ZoneType.BULLISH_OB,
                            start_idx=j,
                            end_idx=j,
                            upper_price=highs[j],
                            lower_price=lows[j],
                            strength=min(1.0, move / (3 * atr[i]))
                        ))
                        break
            
            # Bearish impulse
            elif move < -2 * atr[i]:
                # Find last bullish candle before impulse
                for j in range(i-1, max(i-10, 0), -1):
                    if closes[j] > opens[j]:  # Bullish candle
                        obs.append(SMCZone(
                            zone_type=ZoneType.BEARISH_OB,
                            start_idx=j,
                            end_idx=j,
                            upper_price=highs[j],
                            lower_price=lows[j],
                            strength=min(1.0, abs(move) / (3 * atr[i]))
                        ))
                        break
        
        return obs
    
    # ==========================================================================
    # LIQUIDITY ZONES
    # ==========================================================================
    
    def detect_liquidity_zones(self, highs: np.ndarray, lows: np.ndarray,
                               lookback: int = 20) -> List[SMCZone]:
        """
        Detect liquidity zones (equal highs/lows where stops accumulate).
        
        Buy-side liquidity: Above equal highs (stop losses from shorts)
        Sell-side liquidity: Below equal lows (stop losses from longs)
        """
        zones = []
        tolerance = 0.001  # 0.1% tolerance for "equal" prices
        
        swing_highs = self.find_swing_highs(highs)
        swing_lows = self.find_swing_lows(lows)
        
        # Find equal highs (buy-side liquidity above)
        for i in range(len(swing_highs)):
            for j in range(i + 1, min(i + 5, len(swing_highs))):
                h1 = highs[swing_highs[i]]
                h2 = highs[swing_highs[j]]
                
                if abs(h1 - h2) / h1 < tolerance:
                    avg_high = (h1 + h2) / 2
                    zones.append(SMCZone(
                        zone_type=ZoneType.BUY_LIQUIDITY,
                        start_idx=swing_highs[i],
                        end_idx=swing_highs[j],
                        upper_price=avg_high * 1.002,  # Slightly above
                        lower_price=avg_high,
                        strength=0.8
                    ))
        
        # Find equal lows (sell-side liquidity below)
        for i in range(len(swing_lows)):
            for j in range(i + 1, min(i + 5, len(swing_lows))):
                l1 = lows[swing_lows[i]]
                l2 = lows[swing_lows[j]]
                
                if abs(l1 - l2) / l1 < tolerance:
                    avg_low = (l1 + l2) / 2
                    zones.append(SMCZone(
                        zone_type=ZoneType.SELL_LIQUIDITY,
                        start_idx=swing_lows[i],
                        end_idx=swing_lows[j],
                        upper_price=avg_low,
                        lower_price=avg_low * 0.998,  # Slightly below
                        strength=0.8
                    ))
        
        return zones
    
    # ==========================================================================
    # BREAK OF STRUCTURE (BOS) / CHANGE OF CHARACTER (CHOCH)
    # ==========================================================================
    
    def detect_structure_breaks(self, highs: np.ndarray, lows: np.ndarray,
                                closes: np.ndarray) -> List[SMCZone]:
        """
        Detect Break of Structure (BOS) and Change of Character (CHOCH).
        
        BOS: Break of recent swing in trend direction (continuation)
        CHOCH: Break of recent swing against trend (reversal)
        """
        breaks = []
        
        swing_highs = self.find_swing_highs(highs, left=3, right=3)
        swing_lows = self.find_swing_lows(lows, left=3, right=3)
        
        # Track market structure
        current_trend = 'neutral'
        last_swing_high = None
        last_swing_low = None
        
        for i in range(20, len(closes)):
            # Update last swings
            recent_highs = [sh for sh in swing_highs if sh < i and sh > i - 20]
            recent_lows = [sl for sl in swing_lows if sl < i and sl > i - 20]
            
            if recent_highs:
                last_swing_high = recent_highs[-1]
            if recent_lows:
                last_swing_low = recent_lows[-1]
            
            if last_swing_high is None or last_swing_low is None:
                continue
            
            # Determine current trend (higher highs/lows = bullish)
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                hh = highs[recent_highs[-1]] > highs[recent_highs[-2]]
                hl = lows[recent_lows[-1]] > lows[recent_lows[-2]]
                lh = highs[recent_highs[-1]] < highs[recent_highs[-2]]
                ll = lows[recent_lows[-1]] < lows[recent_lows[-2]]
                
                if hh and hl:
                    current_trend = 'bullish'
                elif lh and ll:
                    current_trend = 'bearish'
            
            # Check for breaks
            # Break above last swing high
            if closes[i] > highs[last_swing_high] and closes[i-1] <= highs[last_swing_high]:
                if current_trend == 'bearish':
                    # CHOCH: Breaking high in downtrend = reversal
                    breaks.append(SMCZone(
                        zone_type=ZoneType.CHOCH_BULLISH,
                        start_idx=last_swing_high,
                        end_idx=i,
                        upper_price=highs[last_swing_high],
                        lower_price=lows[last_swing_low],
                        strength=0.9
                    ))
                else:
                    # BOS: Breaking high in uptrend = continuation
                    breaks.append(SMCZone(
                        zone_type=ZoneType.BOS_BULLISH,
                        start_idx=last_swing_high,
                        end_idx=i,
                        upper_price=highs[last_swing_high],
                        lower_price=lows[last_swing_low],
                        strength=0.7
                    ))
            
            # Break below last swing low
            if closes[i] < lows[last_swing_low] and closes[i-1] >= lows[last_swing_low]:
                if current_trend == 'bullish':
                    # CHOCH: Breaking low in uptrend = reversal
                    breaks.append(SMCZone(
                        zone_type=ZoneType.CHOCH_BEARISH,
                        start_idx=last_swing_low,
                        end_idx=i,
                        upper_price=highs[last_swing_high],
                        lower_price=lows[last_swing_low],
                        strength=0.9
                    ))
                else:
                    # BOS: Breaking low in downtrend = continuation
                    breaks.append(SMCZone(
                        zone_type=ZoneType.BOS_BEARISH,
                        start_idx=last_swing_low,
                        end_idx=i,
                        upper_price=highs[last_swing_high],
                        lower_price=lows[last_swing_low],
                        strength=0.7
                    ))
        
        return breaks
    
    # ==========================================================================
    # PREMIUM/DISCOUNT ZONES
    # ==========================================================================
    
    def get_premium_discount_zones(self, highs: np.ndarray, lows: np.ndarray,
                                   lookback: int = 50) -> Dict[str, Tuple[float, float]]:
        """
        Calculate Premium and Discount zones based on recent range.
        
        Premium: Upper 50% of range (expensive - look to sell)
        Discount: Lower 50% of range (cheap - look to buy)
        """
        recent_high = np.max(highs[-lookback:])
        recent_low = np.min(lows[-lookback:])
        
        mid_price = (recent_high + recent_low) / 2
        
        return {
            'premium': (mid_price, recent_high),
            'discount': (recent_low, mid_price),
            'equilibrium': mid_price
        }
    
    # ==========================================================================
    # MAIN ANALYSIS
    # ==========================================================================
    
    def analyze(self, opens: np.ndarray, highs: np.ndarray, 
                lows: np.ndarray, closes: np.ndarray) -> Dict[str, List[SMCZone]]:
        """
        Run full Smart Money Concepts analysis.
        
        Returns:
            Dictionary with all SMC zones categorized
        """
        return {
            'fvg': self.detect_fvg(opens, highs, lows, closes),
            'order_blocks': self.detect_order_blocks(opens, highs, lows, closes),
            'liquidity': self.detect_liquidity_zones(highs, lows),
            'structure': self.detect_structure_breaks(highs, lows, closes)
        }
    
    def get_signal_at_index(self, opens: np.ndarray, highs: np.ndarray,
                            lows: np.ndarray, closes: np.ndarray,
                            index: int, lookback: int = 10) -> Tuple[int, float, List[str]]:
        """
        Get SMC-based signal at specific index.
        
        Returns:
            Tuple of (signal, confidence, zone_types)
        """
        all_zones = self.analyze(opens, highs, lows, closes)
        
        bullish_score = 0.0
        bearish_score = 0.0
        zone_types = []
        
        current_price = closes[index]
        
        for category, zones in all_zones.items():
            for zone in zones:
                # Skip zones that formed after our index
                if zone.end_idx > index:
                    continue
                
                # Check if price is near zone
                in_zone = zone.lower_price <= current_price <= zone.upper_price
                near_zone = (abs(current_price - zone.upper_price) / current_price < 0.005 or
                            abs(current_price - zone.lower_price) / current_price < 0.005)
                
                if not (in_zone or near_zone):
                    continue
                
                # Recent zones are more important
                recency = max(0, 1 - (index - zone.end_idx) / 50)
                weight = zone.strength * recency
                
                zone_types.append(zone.zone_type.value)
                
                if zone.zone_type in [ZoneType.BULLISH_FVG, ZoneType.BULLISH_OB,
                                       ZoneType.BOS_BULLISH, ZoneType.CHOCH_BULLISH]:
                    bullish_score += weight
                elif zone.zone_type in [ZoneType.BEARISH_FVG, ZoneType.BEARISH_OB,
                                         ZoneType.BOS_BEARISH, ZoneType.CHOCH_BEARISH]:
                    bearish_score += weight
        
        # Check premium/discount
        pd_zones = self.get_premium_discount_zones(highs[:index+1], lows[:index+1])
        if current_price < pd_zones['discount'][1]:
            bullish_score += 0.3  # In discount zone
        elif current_price > pd_zones['premium'][0]:
            bearish_score += 0.3  # In premium zone
        
        total = bullish_score + bearish_score
        if total == 0:
            return 0, 0.0, []
        
        if bullish_score > bearish_score * 1.3:
            return 1, bullish_score / total, zone_types
        elif bearish_score > bullish_score * 1.3:
            return -1, bearish_score / total, zone_types
        
        return 0, 0.5, zone_types


def generate_smc_signals(opens: np.ndarray, highs: np.ndarray,
                         lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """
    Generate SMC-based signals for entire dataset.
    
    Returns:
        Array of signals: 1 (bullish), -1 (bearish), 0 (neutral)
    """
    analyzer = SmartMoneyAnalyzer()
    signals = np.zeros(len(closes))
    
    for i in range(50, len(closes)):
        signal, conf, _ = analyzer.get_signal_at_index(opens, highs, lows, closes, i)
        if conf > 0.6:
            signals[i] = signal
    
    return signals


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("SMART MONEY CONCEPTS (SMC) - DEMO")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n = 200
    
    # Trending data with swings
    trend = np.linspace(0, 30, n)
    swings = 5 * np.sin(np.linspace(0, 10 * np.pi, n))
    noise = np.cumsum(np.random.randn(n) * 0.3)
    
    closes = 100 + trend + swings + noise
    opens = np.roll(closes, 1) + np.random.randn(n) * 0.2
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 0.5)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 0.5)
    
    # Analyze
    analyzer = SmartMoneyAnalyzer()
    all_zones = analyzer.analyze(opens, highs, lows, closes)
    
    # Print results
    print("\n" + "=" * 60)
    print("SMC ZONES DETECTED")
    print("=" * 60)
    
    for category, zones in all_zones.items():
        print(f"\n{category.upper()}: {len(zones)} zones")
        for zone in zones[:3]:  # Show first 3
            print(f"  {zone.zone_type.value:20s} | Bars {zone.start_idx:3d}-{zone.end_idx:3d} | "
                  f"${zone.lower_price:.2f}-${zone.upper_price:.2f} | Str: {zone.strength:.2f}")
    
    # Premium/Discount
    pd_zones = analyzer.get_premium_discount_zones(highs, lows)
    print(f"\n{'='*60}")
    print("PREMIUM/DISCOUNT ZONES")
    print("=" * 60)
    print(f"Premium Zone:  ${pd_zones['premium'][0]:.2f} - ${pd_zones['premium'][1]:.2f}")
    print(f"Discount Zone: ${pd_zones['discount'][0]:.2f} - ${pd_zones['discount'][1]:.2f}")
    print(f"Equilibrium:   ${pd_zones['equilibrium']:.2f}")
    
    # Generate signals
    signals = generate_smc_signals(opens, highs, lows, closes)
    buy_signals = np.sum(signals == 1)
    sell_signals = np.sum(signals == -1)
    
    print(f"\n{'='*60}")
    print("SIGNAL SUMMARY")
    print("=" * 60)
    print(f"Buy Signals:  {buy_signals}")
    print(f"Sell Signals: {sell_signals}")
    
    print("\n[OK] Smart Money Concepts Module Working!")
