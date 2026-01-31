"""
Improved Trading Indicator with Research-Based Strategy
Based on "Ultimate Trading Indicator Research.pdf" findings

Key improvements from research:
1. Triple SuperTrend (10/1.0, 11/2.0, 12/3.0) for multi-timeframe confirmation
2. EMA(200) as trend filter (improves Sharpe from 0.43 to 0.91)
3. Stochastic RSI for entry timing
4. Realistic transaction costs (0.1% round-trip minimum)
5. Monte Carlo simulation for robustness testing
6. Walk-forward analysis for validation

Expected performance (from research):
- Win rate: 42-48% (realistic, not 70%)
- Profit factor: 1.8-2.2
- Sharpe ratio: 0.7-1.0
- Max drawdown: 25-35%
"""

import json
import random
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import argparse


@dataclass
class TradeResult:
    """Represents a single trade result."""
    entry_date: str
    exit_date: str
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_percent: float
    exit_reason: str
    indicators_at_entry: Dict = field(default_factory=dict)


@dataclass
class ImprovedBacktestConfig:
    """
    Configuration based on research PDF recommendations.
    
    Key differences from original:
    - Triple SuperTrend with specific multipliers
    - EMA(200) as primary trend filter
    - Stochastic RSI for timing
    - Higher transaction costs (realistic for India)
    """
    # Triple SuperTrend settings (from research)
    st1_period: int = 10
    st1_mult: float = 1.0
    st2_period: int = 11  
    st2_mult: float = 2.0
    st3_period: int = 12
    st3_mult: float = 3.0
    
    # EMA trend filter (research shows EMA 200 improves Sharpe by 2x)
    ema_period: int = 200
    
    # Stochastic RSI settings
    rsi_period: int = 14
    stoch_period: int = 3
    stoch_smooth: int = 3
    oversold_level: int = 28
    overbought_level: int = 78
    
    # ATR for stops
    atr_period: int = 14
    stop_loss_atr_mult: float = 1.5  # Research recommends 1.5x ATR
    take_profit_atr_mult: float = 3.0  # 2:1 reward-risk as per research
    
    # Volume filter
    use_volume_filter: bool = True
    volume_ma_period: int = 20
    
    # Transaction costs (CRITICAL - research shows 0.1% minimum)
    # Includes: brokerage (0.03%), STT (0.025%), exchange fees, GST
    commission_percent: float = 0.05  # 0.05% per side
    slippage_percent: float = 0.05    # 0.05% slippage
    # Total round-trip: 0.2% (higher than original 0.03%)
    
    # Position sizing
    initial_capital: float = 500000.0  # ₹5 lakh realistic
    risk_per_trade_percent: float = 2.0  # 2% risk per trade (industry standard)
    
    # Required confirmations (research: need 2/3 SuperTrends agreeing)
    min_supertrend_agreement: int = 2


@dataclass
class BacktestResults:
    """Results with Monte Carlo compatible metrics."""
    trades: list = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    total_return_percent: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_trade_duration_days: float = 0.0
    # Monte Carlo metrics
    mc_median_drawdown: float = 0.0
    mc_95th_percentile_drawdown: float = 0.0
    probabilistic_sharpe_ratio: float = 0.0


# ============================================================================
# INDICATOR CALCULATIONS
# ============================================================================

def calculate_ema(data: List[float], period: int) -> List[Optional[float]]:
    """Calculate Exponential Moving Average."""
    if len(data) < period:
        return [None] * len(data)
    
    ema = [None] * (period - 1)
    multiplier = 2 / (period + 1)
    
    # First EMA is SMA
    sma = sum(data[:period]) / period
    ema.append(sma)
    
    for i in range(period, len(data)):
        ema_val = (data[i] * multiplier) + (ema[-1] * (1 - multiplier))
        ema.append(ema_val)
    
    return ema


def calculate_sma(data: List[float], period: int) -> List[Optional[float]]:
    """Calculate Simple Moving Average."""
    sma = []
    for i in range(len(data)):
        if i < period - 1:
            sma.append(None)
        else:
            window = data[i - period + 1:i + 1]
            if any(v is None for v in window):
                sma.append(None)
            else:
                sma.append(sum(window) / period)
    return sma


def calculate_rsi(closes: List[float], period: int) -> List[Optional[float]]:
    """Calculate RSI."""
    if len(closes) < period + 1:
        return [None] * len(closes)
    
    rsi = [None] * period
    changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [max(c, 0) for c in changes]
    losses = [abs(min(c, 0)) for c in changes]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    if avg_loss == 0:
        rsi.append(100)
    else:
        rs = avg_gain / avg_loss
        rsi.append(100 - (100 / (1 + rs)))
    
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100 - (100 / (1 + rs)))
    
    return rsi


def calculate_stochastic_rsi(closes: List[float], rsi_period: int, 
                              stoch_period: int, smooth_k: int) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Calculate Stochastic RSI (K and D lines).
    This is the key timing indicator from the research.
    """
    rsi = calculate_rsi(closes, rsi_period)
    
    k_line = []
    for i in range(len(rsi)):
        if i < stoch_period - 1 or rsi[i] is None:
            k_line.append(None)
            continue
        
        # Get RSI values for stochastic period
        rsi_window = [r for r in rsi[i-stoch_period+1:i+1] if r is not None]
        if len(rsi_window) < stoch_period:
            k_line.append(None)
            continue
        
        rsi_min = min(rsi_window)
        rsi_max = max(rsi_window)
        
        if rsi_max == rsi_min:
            k_line.append(50)
        else:
            stoch_rsi = ((rsi[i] - rsi_min) / (rsi_max - rsi_min)) * 100
            k_line.append(stoch_rsi)
    
    # Smooth K line to get actual K
    k_smoothed = calculate_sma(k_line, smooth_k)
    
    # D line is SMA of smoothed K
    d_line = calculate_sma(k_smoothed, smooth_k)
    
    return k_smoothed, d_line


def calculate_atr(highs: List[float], lows: List[float], 
                  closes: List[float], period: int) -> List[Optional[float]]:
    """Calculate Average True Range."""
    tr = [highs[0] - lows[0]]
    
    for i in range(1, len(closes)):
        tr_val = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        tr.append(tr_val)
    
    return calculate_sma(tr, period)


def calculate_supertrend(highs: List[float], lows: List[float], 
                         closes: List[float], period: int, 
                         multiplier: float) -> Tuple[List[Optional[float]], List[int]]:
    """
    Calculate SuperTrend indicator.
    Returns (supertrend_values, direction) where direction: 1=bullish, -1=bearish
    """
    atr = calculate_atr(highs, lows, closes, period)
    
    supertrend = [None] * len(closes)
    direction = [0] * len(closes)
    upper_band = [None] * len(closes)
    lower_band = [None] * len(closes)
    
    for i in range(len(closes)):
        if atr[i] is None:
            continue
        
        hl2 = (highs[i] + lows[i]) / 2
        upper_band[i] = hl2 + (multiplier * atr[i])
        lower_band[i] = hl2 - (multiplier * atr[i])
        
        if i == 0 or upper_band[i-1] is None:
            supertrend[i] = lower_band[i]
            direction[i] = 1
            continue
        
        # Determine trend direction
        if closes[i-1] > upper_band[i-1]:
            direction[i] = 1
        elif closes[i-1] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        # Adjust bands
        if direction[i] == 1:
            if lower_band[i-1]:
                lower_band[i] = max(lower_band[i], lower_band[i-1])
            supertrend[i] = lower_band[i]
        else:
            if upper_band[i-1]:
                upper_band[i] = min(upper_band[i], upper_band[i-1])
            supertrend[i] = upper_band[i]
    
    return supertrend, direction


# ============================================================================
# IMPROVED BACKTEST ENGINE
# ============================================================================

def run_improved_backtest(data: List[Dict], config: ImprovedBacktestConfig) -> BacktestResults:
    """
    Run backtest with research-based multi-confirmation strategy.
    
    Strategy logic (from research PDF):
    1. EMA(200) as trend filter - only trade in trend direction
    2. 2/3 SuperTrends must agree on direction
    3. Stochastic RSI crossover in oversold/overbought zone for timing
    4. Volume above 20-period average for confirmation
    """
    results = BacktestResults()
    min_bars = max(config.ema_period, 
                   config.st3_period + config.atr_period,
                   config.rsi_period + config.stoch_period * 2) + 5
    
    if len(data) < min_bars:
        print(f"Insufficient data: need {min_bars} bars, have {len(data)}")
        return results
    
    # Extract price data
    dates = [d["date"] for d in data]
    opens = [d["open"] for d in data]
    highs = [d["high"] for d in data]
    lows = [d["low"] for d in data]
    closes = [d["close"] for d in data]
    volumes = [d["volume"] for d in data]
    
    # Calculate indicators
    ema200 = calculate_ema(closes, config.ema_period)
    st1, dir1 = calculate_supertrend(highs, lows, closes, config.st1_period, config.st1_mult)
    st2, dir2 = calculate_supertrend(highs, lows, closes, config.st2_period, config.st2_mult)
    st3, dir3 = calculate_supertrend(highs, lows, closes, config.st3_period, config.st3_mult)
    stoch_k, stoch_d = calculate_stochastic_rsi(closes, config.rsi_period, 
                                                  config.stoch_period, config.stoch_smooth)
    atr = calculate_atr(highs, lows, closes, config.atr_period)
    vol_ma = calculate_sma(volumes, config.volume_ma_period)
    
    # Trading state
    position = None
    entry_price = 0
    entry_date = ""
    stop_loss = 0
    take_profit = 0
    position_size = 0
    
    trades = []
    current_capital = config.initial_capital
    peak_capital = config.initial_capital
    max_drawdown = 0
    equity_curve = [config.initial_capital]
    
    # Start trading after all indicators are valid
    start_idx = min_bars
    
    for i in range(start_idx, len(data)):
        # Check if all indicators valid
        if (ema200[i] is None or st1[i] is None or st2[i] is None or 
            st3[i] is None or stoch_k[i] is None or stoch_d[i] is None or
            atr[i] is None or vol_ma[i] is None):
            continue
        
        # Count bullish/bearish SuperTrends (research: need 2/3 agreement)
        bullish_st = sum(1 for d in [dir1[i], dir2[i], dir3[i]] if d == 1)
        bearish_st = sum(1 for d in [dir1[i], dir2[i], dir3[i]] if d == -1)
        
        # EMA trend filter (CRITICAL - research shows 2x Sharpe improvement)
        above_ema = closes[i] > ema200[i]
        below_ema = closes[i] < ema200[i]
        
        # Stochastic RSI conditions
        k_crossed_above_d = (stoch_k[i-1] is not None and stoch_d[i-1] is not None and
                            stoch_k[i-1] < stoch_d[i-1] and stoch_k[i] > stoch_d[i])
        k_crossed_below_d = (stoch_k[i-1] is not None and stoch_d[i-1] is not None and
                            stoch_k[i-1] > stoch_d[i-1] and stoch_k[i] < stoch_d[i])
        oversold = stoch_k[i] < config.oversold_level
        overbought = stoch_k[i] > config.overbought_level
        
        # Volume confirmation
        volume_confirm = not config.use_volume_filter or volumes[i] > vol_ma[i]
        
        # ================================================================
        # ENTRY SIGNALS (Research-based Multi-Confirmation)
        # ================================================================
        
        # LONG: Above EMA + 2/3 SuperTrends bullish + Stoch RSI oversold cross up + Volume
        long_signal = (above_ema and 
                       bullish_st >= config.min_supertrend_agreement and
                       k_crossed_above_d and oversold and
                       volume_confirm)
        
        # SHORT: Below EMA + 2/3 SuperTrends bearish + Stoch RSI overbought cross down + Volume
        short_signal = (below_ema and 
                        bearish_st >= config.min_supertrend_agreement and
                        k_crossed_below_d and overbought and
                        volume_confirm)
        
        # ================================================================
        # EXIT CONDITIONS
        # ================================================================
        
        # Exit long if trend reverses or hits stops
        exit_long = (below_ema or bearish_st >= 2)
        
        # Exit short if trend reverses or hits stops  
        exit_short = (above_ema or bullish_st >= 2)
        
        # ================================================================
        # POSITION MANAGEMENT
        # ================================================================
        
        # Check existing position
        if position == 'long':
            # Check stop loss
            if lows[i] <= stop_loss:
                exit_price = stop_loss * (1 - config.slippage_percent / 100)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trade_pnl = position_size * (exit_price / entry_price - 1)
                trade_pnl -= position_size * config.commission_percent / 100  # Exit commission
                
                trades.append(TradeResult(
                    entry_date=entry_date, exit_date=dates[i], direction='long',
                    entry_price=entry_price, exit_price=exit_price,
                    stop_loss=stop_loss, take_profit=take_profit,
                    pnl=trade_pnl, pnl_percent=pnl_pct, exit_reason='stop_loss',
                    indicators_at_entry={}
                ))
                current_capital += trade_pnl
                position = None
                
            # Check take profit
            elif highs[i] >= take_profit:
                exit_price = take_profit * (1 - config.slippage_percent / 100)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trade_pnl = position_size * (exit_price / entry_price - 1)
                trade_pnl -= position_size * config.commission_percent / 100
                
                trades.append(TradeResult(
                    entry_date=entry_date, exit_date=dates[i], direction='long',
                    entry_price=entry_price, exit_price=exit_price,
                    stop_loss=stop_loss, take_profit=take_profit,
                    pnl=trade_pnl, pnl_percent=pnl_pct, exit_reason='take_profit',
                    indicators_at_entry={}
                ))
                current_capital += trade_pnl
                position = None
                
            # Check signal exit
            elif exit_long:
                exit_price = closes[i] * (1 - config.slippage_percent / 100)
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                trade_pnl = position_size * (exit_price / entry_price - 1)
                trade_pnl -= position_size * config.commission_percent / 100
                
                trades.append(TradeResult(
                    entry_date=entry_date, exit_date=dates[i], direction='long',
                    entry_price=entry_price, exit_price=exit_price,
                    stop_loss=stop_loss, take_profit=take_profit,
                    pnl=trade_pnl, pnl_percent=pnl_pct, exit_reason='signal',
                    indicators_at_entry={}
                ))
                current_capital += trade_pnl
                position = None
        
        elif position == 'short':
            # Check stop loss
            if highs[i] >= stop_loss:
                exit_price = stop_loss * (1 + config.slippage_percent / 100)
                pnl_pct = (entry_price - exit_price) / entry_price * 100
                trade_pnl = position_size * (entry_price / exit_price - 1)
                trade_pnl -= position_size * config.commission_percent / 100
                
                trades.append(TradeResult(
                    entry_date=entry_date, exit_date=dates[i], direction='short',
                    entry_price=entry_price, exit_price=exit_price,
                    stop_loss=stop_loss, take_profit=take_profit,
                    pnl=trade_pnl, pnl_percent=pnl_pct, exit_reason='stop_loss',
                    indicators_at_entry={}
                ))
                current_capital += trade_pnl
                position = None
                
            # Check take profit
            elif lows[i] <= take_profit:
                exit_price = take_profit * (1 + config.slippage_percent / 100)
                pnl_pct = (entry_price - exit_price) / entry_price * 100
                trade_pnl = position_size * (entry_price / exit_price - 1)
                trade_pnl -= position_size * config.commission_percent / 100
                
                trades.append(TradeResult(
                    entry_date=entry_date, exit_date=dates[i], direction='short',
                    entry_price=entry_price, exit_price=exit_price,
                    stop_loss=stop_loss, take_profit=take_profit,
                    pnl=trade_pnl, pnl_percent=pnl_pct, exit_reason='take_profit',
                    indicators_at_entry={}
                ))
                current_capital += trade_pnl
                position = None
                
            # Check signal exit
            elif exit_short:
                exit_price = closes[i] * (1 + config.slippage_percent / 100)
                pnl_pct = (entry_price - exit_price) / entry_price * 100
                trade_pnl = position_size * (entry_price / exit_price - 1)
                trade_pnl -= position_size * config.commission_percent / 100
                
                trades.append(TradeResult(
                    entry_date=entry_date, exit_date=dates[i], direction='short',
                    entry_price=entry_price, exit_price=exit_price,
                    stop_loss=stop_loss, take_profit=take_profit,
                    pnl=trade_pnl, pnl_percent=pnl_pct, exit_reason='signal',
                    indicators_at_entry={}
                ))
                current_capital += trade_pnl
                position = None
        
        # Open new position
        if position is None:
            if long_signal:
                entry_price = closes[i] * (1 + config.slippage_percent / 100)
                entry_date = dates[i]
                stop_loss = entry_price - (atr[i] * config.stop_loss_atr_mult)
                take_profit = entry_price + (atr[i] * config.take_profit_atr_mult)
                
                # Position sizing based on risk (Kelly-inspired but conservative)
                risk_amount = current_capital * config.risk_per_trade_percent / 100
                risk_per_unit = entry_price - stop_loss
                if risk_per_unit > 0:
                    position_size = risk_amount / risk_per_unit * entry_price
                else:
                    position_size = current_capital * 0.1  # Fallback
                
                # Apply entry commission
                current_capital -= position_size * config.commission_percent / 100
                position = 'long'
                
            elif short_signal:
                entry_price = closes[i] * (1 - config.slippage_percent / 100)
                entry_date = dates[i]
                stop_loss = entry_price + (atr[i] * config.stop_loss_atr_mult)
                take_profit = entry_price - (atr[i] * config.take_profit_atr_mult)
                
                risk_amount = current_capital * config.risk_per_trade_percent / 100
                risk_per_unit = stop_loss - entry_price
                if risk_per_unit > 0:
                    position_size = risk_amount / risk_per_unit * entry_price
                else:
                    position_size = current_capital * 0.1
                
                current_capital -= position_size * config.commission_percent / 100
                position = 'short'
        
        # Update equity and drawdown
        equity_curve.append(current_capital)
        if current_capital > peak_capital:
            peak_capital = current_capital
        dd = (peak_capital - current_capital) / peak_capital * 100
        if dd > max_drawdown:
            max_drawdown = dd
    
    # Close any open position
    if position is not None:
        exit_price = closes[-1]
        if position == 'long':
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            trade_pnl = position_size * (exit_price / entry_price - 1)
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100
            trade_pnl = position_size * (entry_price / exit_price - 1)
        
        trade_pnl -= position_size * config.commission_percent / 100
        trades.append(TradeResult(
            entry_date=entry_date, exit_date=dates[-1], direction=position,
            entry_price=entry_price, exit_price=exit_price,
            stop_loss=stop_loss, take_profit=take_profit,
            pnl=trade_pnl, pnl_percent=pnl_pct, exit_reason='end_of_data',
            indicators_at_entry={}
        ))
        current_capital += trade_pnl
    
    # Calculate results
    results.trades = trades
    results.total_trades = len(trades)
    results.winning_trades = sum(1 for t in trades if t.pnl > 0)
    results.losing_trades = sum(1 for t in trades if t.pnl <= 0)
    results.win_rate = results.winning_trades / results.total_trades * 100 if results.total_trades > 0 else 0
    
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    results.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    results.total_pnl = current_capital - config.initial_capital
    results.total_return_percent = (current_capital / config.initial_capital - 1) * 100
    results.max_drawdown_percent = max_drawdown
    
    # Average win/loss
    winning = [t.pnl for t in trades if t.pnl > 0]
    losing = [t.pnl for t in trades if t.pnl < 0]
    results.avg_win = sum(winning) / len(winning) if winning else 0
    results.avg_loss = sum(losing) / len(losing) if losing else 0
    
    # Consecutive wins/losses
    max_cw, max_cl, cw, cl = 0, 0, 0, 0
    for t in trades:
        if t.pnl > 0:
            cw += 1
            cl = 0
            max_cw = max(max_cw, cw)
        else:
            cl += 1
            cw = 0
            max_cl = max(max_cl, cl)
    results.max_consecutive_wins = max_cw
    results.max_consecutive_losses = max_cl
    
    # Sharpe ratio
    if len(trades) > 1:
        returns = [t.pnl_percent for t in trades]
        avg_ret = sum(returns) / len(returns)
        std_dev = math.sqrt(sum((r - avg_ret) ** 2 for r in returns) / (len(returns) - 1)) if len(returns) > 1 else 1
        results.sharpe_ratio = (avg_ret / std_dev) * math.sqrt(252 / max(len(trades), 1)) if std_dev > 0 else 0
    
    # Trade duration
    if trades:
        durations = []
        for t in trades:
            try:
                entry = datetime.strptime(t.entry_date, "%Y-%m-%d")
                exit = datetime.strptime(t.exit_date, "%Y-%m-%d")
                durations.append((exit - entry).days)
            except:
                pass
        results.avg_trade_duration_days = sum(durations) / len(durations) if durations else 0
    
    return results


# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

def run_monte_carlo(trades: List[TradeResult], initial_capital: float, 
                    num_simulations: int = 1000, skip_percentage: float = 0.10) -> Dict:
    """
    Run Monte Carlo simulation to stress test the strategy.
    
    From research: "If Monte Carlo median drawdown is 3x higher than backtest drawdown -> overfitted"
    
    Args:
        trades: List of trade results from backtest
        initial_capital: Starting capital
        num_simulations: Number of Monte Carlo runs (1000+ recommended)
        skip_percentage: Percentage of trades to randomly skip (simulates missed entries)
    
    Returns:
        Dictionary with Monte Carlo statistics
    """
    if not trades:
        return {"error": "No trades for Monte Carlo"}
    
    drawdowns = []
    final_capitals = []
    
    for _ in range(num_simulations):
        # Shuffle trades
        shuffled = trades.copy()
        random.shuffle(shuffled)
        
        # Skip some trades randomly
        num_skip = int(len(shuffled) * skip_percentage)
        indices_to_skip = set(random.sample(range(len(shuffled)), num_skip))
        
        capital = initial_capital
        peak = initial_capital
        max_dd = 0
        
        for idx, trade in enumerate(shuffled):
            if idx in indices_to_skip:
                continue
            
            capital += trade.pnl
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        drawdowns.append(max_dd)
        final_capitals.append(capital)
    
    drawdowns.sort()
    final_capitals.sort()
    
    return {
        "simulations": num_simulations,
        "skip_percentage": skip_percentage,
        "drawdown_median": drawdowns[len(drawdowns) // 2],
        "drawdown_75th": drawdowns[int(len(drawdowns) * 0.75)],
        "drawdown_95th": drawdowns[int(len(drawdowns) * 0.95)],
        "drawdown_99th": drawdowns[int(len(drawdowns) * 0.99)],
        "final_capital_median": final_capitals[len(final_capitals) // 2],
        "final_capital_5th": final_capitals[int(len(final_capitals) * 0.05)],
        "probability_of_profit": sum(1 for c in final_capitals if c > initial_capital) / len(final_capitals) * 100
    }


# ============================================================================
# WALK-FORWARD ANALYSIS
# ============================================================================

def walk_forward_analysis(data: List[Dict], config: ImprovedBacktestConfig,
                          train_ratio: float = 0.7) -> Dict:
    """
    Perform walk-forward analysis.
    
    From research: "Optimize on in-sample, test on out-of-sample WITHOUT re-optimization"
    
    Args:
        data: Full dataset
        train_ratio: Percentage of data for training (0.7 = 70% train, 30% test)
    
    Returns:
        Dictionary with in-sample and out-of-sample results
    """
    split_idx = int(len(data) * train_ratio)
    
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    # Run backtest on training data (in-sample)
    train_results = run_improved_backtest(train_data, config)
    
    # Run backtest on test data (out-of-sample) with SAME parameters
    test_results = run_improved_backtest(test_data, config)
    
    # Calculate degradation
    if train_results.sharpe_ratio > 0 and test_results.sharpe_ratio:
        sharpe_degradation = (train_results.sharpe_ratio - test_results.sharpe_ratio) / train_results.sharpe_ratio * 100
    else:
        sharpe_degradation = 100
    
    return {
        "train_period": f"{data[0]['date']} to {data[split_idx-1]['date']}" if split_idx > 0 else "N/A",
        "test_period": f"{data[split_idx]['date']} to {data[-1]['date']}" if split_idx < len(data) else "N/A",
        "train_trades": train_results.total_trades,
        "test_trades": test_results.total_trades,
        "train_win_rate": train_results.win_rate,
        "test_win_rate": test_results.win_rate,
        "train_sharpe": train_results.sharpe_ratio,
        "test_sharpe": test_results.sharpe_ratio,
        "train_profit_factor": train_results.profit_factor,
        "test_profit_factor": test_results.profit_factor,
        "train_return": train_results.total_return_percent,
        "test_return": test_results.total_return_percent,
        "sharpe_degradation_percent": sharpe_degradation,
        "is_overfitted": sharpe_degradation > 50,  # >50% degradation suggests overfitting
        "train_results": train_results,
        "test_results": test_results
    }


# ============================================================================
# SAMPLE DATA GENERATION (for testing)
# ============================================================================

def generate_sample_data(symbol: str, start_date: str, end_date: str, 
                         seed: int = 42) -> List[Dict]:
    """Generate realistic sample OHLCV data."""
    random.seed(seed)
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    base_prices = {
        "NIFTY50": 18000, "BANKNIFTY": 42000, "TCS": 3500,
        "RELIANCE": 2400, "HDFC": 1600, "SPY": 450, 
        "AAPL": 180, "GOLD": 1900
    }
    base_price = base_prices.get(symbol.upper(), 1000)
    
    data = []
    current_date = start
    current_price = base_price
    
    # Add trending and ranging regimes for realistic simulation
    trend_strength = 0
    regime_counter = 0
    
    while current_date <= end:
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        # Change regime every 20-50 days
        regime_counter += 1
        if regime_counter > random.randint(20, 50):
            trend_strength = random.uniform(-0.003, 0.003)  # New trend bias
            regime_counter = 0
        
        daily_volatility = 0.015
        change = random.gauss(trend_strength, daily_volatility)
        
        open_price = current_price
        close_price = current_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
        volume = int(random.uniform(5000000, 20000000))
        
        data.append({
            "date": current_date.strftime("%Y-%m-%d"),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": volume
        })
        
        current_price = close_price
        current_date += timedelta(days=1)
    
    return data


# ============================================================================
# MAIN
# ============================================================================

def print_results(results: BacktestResults, symbol: str) -> None:
    """Print formatted backtest results."""
    print("\n" + "=" * 70)
    print(f"IMPROVED STRATEGY BACKTEST RESULTS - {symbol}")
    print("(Based on Ultimate Trading Indicator Research.pdf)")
    print("=" * 70)
    
    print(f"\n{'Performance Metrics':^70}")
    print("-" * 70)
    print(f"Total Trades:              {results.total_trades}")
    print(f"Winning Trades:            {results.winning_trades}")
    print(f"Losing Trades:             {results.losing_trades}")
    print(f"Win Rate:                  {results.win_rate:.2f}% (target: 42-48%)")
    print(f"Profit Factor:             {results.profit_factor:.2f} (target: 1.8-2.2)")
    print(f"Sharpe Ratio:              {results.sharpe_ratio:.2f} (target: 0.7-1.0)")
    print(f"Total P&L:                 ₹{results.total_pnl:,.2f}")
    print(f"Total Return:              {results.total_return_percent:.2f}%")
    print(f"Max Drawdown:              {results.max_drawdown_percent:.2f}% (target: <35%)")
    
    print(f"\n{'Trade Statistics':^70}")
    print("-" * 70)
    print(f"Average Win:               ₹{results.avg_win:,.2f}")
    print(f"Average Loss:              ₹{results.avg_loss:,.2f}")
    print(f"Avg Trade Duration:        {results.avg_trade_duration_days:.1f} days")
    print(f"Max Consecutive Wins:      {results.max_consecutive_wins}")
    print(f"Max Consecutive Losses:    {results.max_consecutive_losses}")
    
    # Research-based assessment
    print(f"\n{'Research Validation':^70}")
    print("-" * 70)
    win_rate_ok = 35 <= results.win_rate <= 55
    pf_ok = 1.5 <= results.profit_factor <= 3.0
    sharpe_ok = results.sharpe_ratio >= 0.5
    dd_ok = results.max_drawdown_percent <= 40
    
    print(f"Win Rate in Range (35-55%): {'✓ PASS' if win_rate_ok else '✗ FAIL'}")
    print(f"Profit Factor OK (1.5-3.0): {'✓ PASS' if pf_ok else '✗ FAIL'}")
    print(f"Sharpe Ratio OK (>0.5):     {'✓ PASS' if sharpe_ok else '✗ FAIL'}")
    print(f"Max DD Acceptable (<40%):   {'✓ PASS' if dd_ok else '✗ FAIL'}")
    
    print("\n" + "=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Improved Trading Strategy Backtest')
    parser.add_argument('--symbol', type=str, default='NIFTY50')
    parser.add_argument('--start', type=str, default='2022-01-01')
    parser.add_argument('--end', type=str, default='2025-12-31')
    parser.add_argument('--capital', type=float, default=500000)
    parser.add_argument('--monte-carlo', action='store_true', help='Run Monte Carlo simulation')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward analysis')
    parser.add_argument('--output', type=str, help='Output JSON file')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("IMPROVED TRADING STRATEGY - Based on Research PDF")
    print(f"{'='*70}")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Capital: ₹{args.capital:,.2f}")
    
    # Generate data
    print("\nLoading data...")
    data = generate_sample_data(args.symbol, args.start, args.end)
    print(f"Loaded {len(data)} bars")
    
    # Run backtest
    config = ImprovedBacktestConfig(initial_capital=args.capital)
    print("\nRunning backtest...")
    results = run_improved_backtest(data, config)
    
    print_results(results, args.symbol)
    
    # Monte Carlo
    if args.monte_carlo and results.trades:
        print("\nRunning Monte Carlo Simulation (1000 runs)...")
        mc_results = run_monte_carlo(results.trades, args.capital)
        print(f"  Median Drawdown:     {mc_results['drawdown_median']:.2f}%")
        print(f"  95th Percentile DD:  {mc_results['drawdown_95th']:.2f}%")
        print(f"  Probability of Profit: {mc_results['probability_of_profit']:.1f}%")
        
        # Overfitting check
        if mc_results['drawdown_median'] > results.max_drawdown_percent * 3:
            print("  ⚠️  WARNING: Monte Carlo suggests strategy may be overfitted")
    
    # Walk-forward
    if args.walk_forward:
        print("\nRunning Walk-Forward Analysis...")
        wf_results = walk_forward_analysis(data, config)
        print(f"  Training Period: {wf_results['train_period']}")
        print(f"  Test Period: {wf_results['test_period']}")
        print(f"  Train Sharpe: {wf_results['train_sharpe']:.2f}")
        print(f"  Test Sharpe: {wf_results['test_sharpe']:.2f}")
        print(f"  Sharpe Degradation: {wf_results['sharpe_degradation_percent']:.1f}%")
        
        if wf_results['is_overfitted']:
            print("  ⚠️  WARNING: >50% Sharpe degradation suggests overfitting")
        else:
            print("  ✓ Strategy passes walk-forward validation")
    
    # Export
    if args.output:
        output_data = {
            "symbol": args.symbol,
            "period": f"{args.start} to {args.end}",
            "config": {
                "ema_period": config.ema_period,
                "st1": f"{config.st1_period}/{config.st1_mult}",
                "st2": f"{config.st2_period}/{config.st2_mult}",
                "st3": f"{config.st3_period}/{config.st3_mult}",
                "transaction_cost": f"{(config.commission_percent + config.slippage_percent) * 2:.2f}% round-trip"
            },
            "results": {
                "total_trades": results.total_trades,
                "win_rate": round(results.win_rate, 2),
                "profit_factor": round(results.profit_factor, 2),
                "sharpe_ratio": round(results.sharpe_ratio, 2),
                "total_return_percent": round(results.total_return_percent, 2),
                "max_drawdown_percent": round(results.max_drawdown_percent, 2)
            },
            "trades": [
                {
                    "entry": t.entry_date,
                    "exit": t.exit_date,
                    "direction": t.direction,
                    "pnl": round(t.pnl, 2),
                    "exit_reason": t.exit_reason
                }
                for t in results.trades
            ]
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    return results


if __name__ == "__main__":
    main()
