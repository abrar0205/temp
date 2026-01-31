"""
Universal Multi-Confirmation Trading Indicator (UMCI) - Python Backtesting Framework
Version 1.0

This module provides backtesting capabilities for the UMCI indicator.
Supports multiple data sources and comprehensive performance analysis.

Dependencies:
- pandas
- numpy
- matplotlib (for visualization)
- yfinance (for data download, optional)

Usage:
    python backtest.py --symbol NIFTY50 --start 2021-01-01 --end 2026-01-31

Author: Trading Indicator Research Project
License: MIT
"""

import json
import argparse
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import math


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
    exit_reason: str  # 'stop_loss', 'take_profit', 'signal', 'end_of_data'


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Trend settings
    fast_ema: int = 9
    slow_ema: int = 21
    supertrend_period: int = 10
    supertrend_mult: float = 3.0
    
    # Momentum settings
    rsi_period: int = 14
    rsi_oversold: int = 40
    rsi_overbought: int = 60
    
    # Volatility settings
    atr_period: int = 14
    atr_min_mult: float = 0.5
    atr_max_mult: float = 3.0
    
    # Volume settings
    use_volume_filter: bool = True
    obv_ema_period: int = 20
    
    # Risk management
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    
    # Capital settings
    initial_capital: float = 100000.0
    position_size_percent: float = 10.0  # % of capital per trade
    commission_percent: float = 0.02  # 0.02% commission
    slippage_percent: float = 0.01  # 0.01% slippage


@dataclass
class BacktestResults:
    """Results from backtesting."""
    trades: list = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_pnl: float = 0.0
    total_return_percent: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_trade_duration_days: float = 0.0


def calculate_ema(data: list, period: int) -> list:
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


def calculate_sma(data: list, period: int) -> list:
    """Calculate Simple Moving Average."""
    if len(data) < period:
        return [None] * len(data)
    
    sma = []
    for i in range(len(data)):
        if i < period - 1:
            sma.append(None)
        else:
            window = data[i - period + 1:i + 1]
            # Skip if any value in window is None
            if any(v is None for v in window):
                sma.append(None)
            else:
                sma.append(sum(window) / period)
    
    return sma


def calculate_rsi(closes: list, period: int) -> list:
    """Calculate Relative Strength Index."""
    if len(closes) < period + 1:
        return [None] * len(closes)
    
    rsi = [None] * period
    
    # Calculate price changes
    changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    
    # Separate gains and losses
    gains = [max(c, 0) for c in changes]
    losses = [abs(min(c, 0)) for c in changes]
    
    # First average gain and loss
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    if avg_loss == 0:
        rsi.append(100)
    else:
        rs = avg_gain / avg_loss
        rsi.append(100 - (100 / (1 + rs)))
    
    # Calculate remaining RSI values using smoothed averages
    for i in range(period, len(changes)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            rsi.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100 - (100 / (1 + rs)))
    
    return rsi


def calculate_atr(highs: list, lows: list, closes: list, period: int) -> list:
    """Calculate Average True Range."""
    if len(closes) < period + 1:
        return [None] * len(closes)
    
    tr = [highs[0] - lows[0]]  # First TR is just high - low
    
    for i in range(1, len(closes)):
        tr_val = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        tr.append(tr_val)
    
    atr = calculate_sma(tr, period)
    return atr


def calculate_supertrend(highs: list, lows: list, closes: list, period: int, multiplier: float) -> tuple:
    """
    Calculate SuperTrend indicator.
    Returns (supertrend_values, trend_direction) where direction is 1 for bullish, -1 for bearish.
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
        
        # Adjust bands based on previous values
        if closes[i-1] > upper_band[i-1]:
            direction[i] = 1
        elif closes[i-1] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            lower_band[i] = max(lower_band[i], lower_band[i-1]) if lower_band[i-1] else lower_band[i]
            supertrend[i] = lower_band[i]
        else:
            upper_band[i] = min(upper_band[i], upper_band[i-1]) if upper_band[i-1] else upper_band[i]
            supertrend[i] = upper_band[i]
    
    return supertrend, direction


def calculate_obv(closes: list, volumes: list) -> list:
    """Calculate On Balance Volume."""
    obv = [0]
    
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    
    return obv


def generate_sample_data(symbol: str, start_date: str, end_date: str) -> list:
    """
    Generate sample OHLCV data for backtesting.
    In production, replace with actual data from yfinance or other sources.
    """
    import random
    random.seed(42)  # For reproducibility
    
    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Initial price based on symbol
    base_prices = {
        "NIFTY50": 18000,
        "BANKNIFTY": 42000,
        "TCS": 3500,
        "RELIANCE": 2400,
        "HDFC": 1600,
        "SPY": 450,
        "AAPL": 180,
        "GOLD": 1900,
    }
    
    base_price = base_prices.get(symbol.upper(), 1000)
    
    data = []
    current_date = start
    current_price = base_price
    
    while current_date <= end:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        # Generate random OHLCV data with realistic characteristics
        daily_volatility = 0.015  # 1.5% daily volatility
        trend_bias = random.uniform(-0.001, 0.002)  # Slight upward bias
        
        change = random.gauss(trend_bias, daily_volatility)
        
        open_price = current_price
        close_price = current_price * (1 + change)
        high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.005))
        low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.005))
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


def run_backtest(data: list, config: BacktestConfig) -> BacktestResults:
    """
    Run backtest on OHLCV data with the given configuration.
    
    Args:
        data: List of OHLCV dictionaries with keys: date, open, high, low, close, volume
        config: BacktestConfig object with indicator parameters
    
    Returns:
        BacktestResults object with performance metrics
    """
    results = BacktestResults()
    
    if len(data) < max(config.slow_ema, config.supertrend_period, config.rsi_period, config.atr_period) + 1:
        print("Insufficient data for backtesting")
        return results
    
    # Extract price data
    dates = [d["date"] for d in data]
    opens = [d["open"] for d in data]
    highs = [d["high"] for d in data]
    lows = [d["low"] for d in data]
    closes = [d["close"] for d in data]
    volumes = [d["volume"] for d in data]
    
    # Calculate indicators
    fast_ema = calculate_ema(closes, config.fast_ema)
    slow_ema = calculate_ema(closes, config.slow_ema)
    rsi = calculate_rsi(closes, config.rsi_period)
    atr = calculate_atr(highs, lows, closes, config.atr_period)
    atr_20 = calculate_sma(atr, 20)
    supertrend, st_direction = calculate_supertrend(highs, lows, closes, 
                                                     config.supertrend_period, 
                                                     config.supertrend_mult)
    obv = calculate_obv(closes, volumes)
    obv_ema = calculate_ema(obv, config.obv_ema_period)
    
    # Trading state
    position = None  # None, 'long', or 'short'
    entry_price = 0
    entry_date = ""
    stop_loss = 0
    take_profit = 0
    
    trades = []
    equity_curve = [config.initial_capital]
    current_capital = config.initial_capital
    peak_capital = config.initial_capital
    max_drawdown = 0
    
    # Start from index where all indicators are valid
    start_idx = max(config.slow_ema, config.supertrend_period + config.atr_period, 
                   config.rsi_period, config.obv_ema_period + 1, 21)
    
    for i in range(start_idx, len(data)):
        # Check if all indicators are valid
        if (fast_ema[i] is None or slow_ema[i] is None or rsi[i] is None or 
            atr[i] is None or atr_20[i] is None or supertrend[i] is None or 
            obv_ema[i] is None):
            continue
        
        # Calculate indicator conditions
        ema_trend_up = fast_ema[i] > slow_ema[i]
        ema_trend_down = fast_ema[i] < slow_ema[i]
        st_bullish = st_direction[i] == 1
        
        strong_uptrend = ema_trend_up and st_bullish
        strong_downtrend = ema_trend_down and not st_bullish
        
        # RSI conditions
        rsi_bullish = (config.rsi_oversold < rsi[i] < config.rsi_overbought and 
                      rsi[i] > rsi[i-1])
        rsi_bearish = (config.rsi_oversold < rsi[i] < config.rsi_overbought and 
                      rsi[i] < rsi[i-1])
        rsi_overbought = rsi[i] > config.rsi_overbought
        rsi_oversold = rsi[i] < config.rsi_oversold
        
        # Volatility filter
        atr_ratio = atr[i] / atr_20[i] if atr_20[i] else 1
        volatility_ok = config.atr_min_mult <= atr_ratio <= config.atr_max_mult
        
        # Volume confirmation
        volume_confirm_bull = not config.use_volume_filter or obv[i] > obv_ema[i]
        volume_confirm_bear = not config.use_volume_filter or obv[i] < obv_ema[i]
        
        # Previous trend for signal detection
        prev_ema_trend_up = fast_ema[i-1] > slow_ema[i-1] if fast_ema[i-1] and slow_ema[i-1] else False
        prev_st_bullish = st_direction[i-1] == 1
        prev_strong_uptrend = prev_ema_trend_up and prev_st_bullish
        prev_strong_downtrend = (fast_ema[i-1] < slow_ema[i-1] if fast_ema[i-1] and slow_ema[i-1] else False) and not prev_st_bullish
        
        # Generate signals
        buy_signal = (strong_uptrend and rsi_bullish and volatility_ok and 
                     volume_confirm_bull and not prev_strong_uptrend)
        sell_signal = (strong_downtrend and rsi_bearish and volatility_ok and 
                      volume_confirm_bear and not prev_strong_downtrend)
        
        exit_long = strong_downtrend or rsi_overbought
        exit_short = strong_uptrend or rsi_oversold
        
        # Check for stop loss / take profit hit
        if position == 'long':
            if lows[i] <= stop_loss:
                # Stop loss hit
                exit_price = stop_loss * (1 - config.slippage_percent / 100)
                pnl = (exit_price - entry_price) / entry_price * 100
                position_value = (current_capital * config.position_size_percent / 100)
                trade_pnl = position_value * (exit_price / entry_price - 1)
                trade_pnl -= position_value * config.commission_percent / 100 * 2  # Entry + exit commission
                
                trades.append(TradeResult(
                    entry_date=entry_date,
                    exit_date=dates[i],
                    direction='long',
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    pnl=trade_pnl,
                    pnl_percent=pnl,
                    exit_reason='stop_loss'
                ))
                
                current_capital += trade_pnl
                position = None
                
            elif highs[i] >= take_profit:
                # Take profit hit
                exit_price = take_profit * (1 - config.slippage_percent / 100)
                pnl = (exit_price - entry_price) / entry_price * 100
                position_value = (current_capital * config.position_size_percent / 100)
                trade_pnl = position_value * (exit_price / entry_price - 1)
                trade_pnl -= position_value * config.commission_percent / 100 * 2
                
                trades.append(TradeResult(
                    entry_date=entry_date,
                    exit_date=dates[i],
                    direction='long',
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    pnl=trade_pnl,
                    pnl_percent=pnl,
                    exit_reason='take_profit'
                ))
                
                current_capital += trade_pnl
                position = None
                
            elif exit_long:
                # Exit signal
                exit_price = closes[i] * (1 - config.slippage_percent / 100)
                pnl = (exit_price - entry_price) / entry_price * 100
                position_value = (current_capital * config.position_size_percent / 100)
                trade_pnl = position_value * (exit_price / entry_price - 1)
                trade_pnl -= position_value * config.commission_percent / 100 * 2
                
                trades.append(TradeResult(
                    entry_date=entry_date,
                    exit_date=dates[i],
                    direction='long',
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    pnl=trade_pnl,
                    pnl_percent=pnl,
                    exit_reason='signal'
                ))
                
                current_capital += trade_pnl
                position = None
        
        elif position == 'short':
            if highs[i] >= stop_loss:
                # Stop loss hit
                exit_price = stop_loss * (1 + config.slippage_percent / 100)
                pnl = (entry_price - exit_price) / entry_price * 100
                position_value = (current_capital * config.position_size_percent / 100)
                trade_pnl = position_value * (entry_price / exit_price - 1)
                trade_pnl -= position_value * config.commission_percent / 100 * 2
                
                trades.append(TradeResult(
                    entry_date=entry_date,
                    exit_date=dates[i],
                    direction='short',
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    pnl=trade_pnl,
                    pnl_percent=pnl,
                    exit_reason='stop_loss'
                ))
                
                current_capital += trade_pnl
                position = None
                
            elif lows[i] <= take_profit:
                # Take profit hit
                exit_price = take_profit * (1 + config.slippage_percent / 100)
                pnl = (entry_price - exit_price) / entry_price * 100
                position_value = (current_capital * config.position_size_percent / 100)
                trade_pnl = position_value * (entry_price / exit_price - 1)
                trade_pnl -= position_value * config.commission_percent / 100 * 2
                
                trades.append(TradeResult(
                    entry_date=entry_date,
                    exit_date=dates[i],
                    direction='short',
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    pnl=trade_pnl,
                    pnl_percent=pnl,
                    exit_reason='take_profit'
                ))
                
                current_capital += trade_pnl
                position = None
                
            elif exit_short:
                # Exit signal
                exit_price = closes[i] * (1 + config.slippage_percent / 100)
                pnl = (entry_price - exit_price) / entry_price * 100
                position_value = (current_capital * config.position_size_percent / 100)
                trade_pnl = position_value * (entry_price / exit_price - 1)
                trade_pnl -= position_value * config.commission_percent / 100 * 2
                
                trades.append(TradeResult(
                    entry_date=entry_date,
                    exit_date=dates[i],
                    direction='short',
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    pnl=trade_pnl,
                    pnl_percent=pnl,
                    exit_reason='signal'
                ))
                
                current_capital += trade_pnl
                position = None
        
        # Open new position if no current position
        if position is None:
            if buy_signal:
                position = 'long'
                entry_price = closes[i] * (1 + config.slippage_percent / 100)
                entry_date = dates[i]
                stop_loss = entry_price - (atr[i] * config.stop_loss_atr_mult)
                take_profit = entry_price + (atr[i] * config.take_profit_atr_mult)
                
            elif sell_signal:
                position = 'short'
                entry_price = closes[i] * (1 - config.slippage_percent / 100)
                entry_date = dates[i]
                stop_loss = entry_price + (atr[i] * config.stop_loss_atr_mult)
                take_profit = entry_price - (atr[i] * config.take_profit_atr_mult)
        
        # Update equity curve and drawdown
        equity_curve.append(current_capital)
        if current_capital > peak_capital:
            peak_capital = current_capital
        drawdown = (peak_capital - current_capital) / peak_capital * 100
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    
    # Close any open position at end of data
    if position is not None:
        exit_price = closes[-1]
        if position == 'long':
            pnl = (exit_price - entry_price) / entry_price * 100
            position_value = (current_capital * config.position_size_percent / 100)
            trade_pnl = position_value * (exit_price / entry_price - 1)
        else:
            pnl = (entry_price - exit_price) / entry_price * 100
            position_value = (current_capital * config.position_size_percent / 100)
            trade_pnl = position_value * (entry_price / exit_price - 1)
        
        trade_pnl -= position_value * config.commission_percent / 100 * 2
        
        trades.append(TradeResult(
            entry_date=entry_date,
            exit_date=dates[-1],
            direction=position,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            pnl=trade_pnl,
            pnl_percent=pnl,
            exit_reason='end_of_data'
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
    results.max_drawdown = max_drawdown
    results.max_drawdown_percent = max_drawdown
    
    # Average win/loss
    winning_pnls = [t.pnl for t in trades if t.pnl > 0]
    losing_pnls = [t.pnl for t in trades if t.pnl < 0]
    results.avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
    results.avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0
    
    # Consecutive wins/losses
    max_consec_wins = 0
    max_consec_losses = 0
    current_wins = 0
    current_losses = 0
    
    for t in trades:
        if t.pnl > 0:
            current_wins += 1
            current_losses = 0
            max_consec_wins = max(max_consec_wins, current_wins)
        else:
            current_losses += 1
            current_wins = 0
            max_consec_losses = max(max_consec_losses, current_losses)
    
    results.max_consecutive_wins = max_consec_wins
    results.max_consecutive_losses = max_consec_losses
    
    # Calculate Sharpe Ratio (simplified)
    if len(trades) > 1:
        returns = [t.pnl_percent for t in trades]
        avg_return = sum(returns) / len(returns)
        std_dev = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1))
        # Annualize (assuming ~252 trading days)
        trading_days = (datetime.strptime(dates[-1], "%Y-%m-%d") - 
                       datetime.strptime(dates[start_idx], "%Y-%m-%d")).days
        trades_per_year = results.total_trades / max(trading_days / 252, 1)
        annual_return = avg_return * trades_per_year
        annual_std = std_dev * math.sqrt(trades_per_year)
        results.sharpe_ratio = annual_return / annual_std if annual_std > 0 else 0
    
    # Average trade duration
    if trades:
        durations = []
        for t in trades:
            entry = datetime.strptime(t.entry_date, "%Y-%m-%d")
            exit = datetime.strptime(t.exit_date, "%Y-%m-%d")
            durations.append((exit - entry).days)
        results.avg_trade_duration_days = sum(durations) / len(durations)
    
    return results


def print_results(results: BacktestResults, symbol: str) -> None:
    """Print backtest results in a formatted manner."""
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS - {symbol}")
    print("=" * 60)
    
    print(f"\n{'Performance Metrics':^60}")
    print("-" * 60)
    print(f"Total Trades:          {results.total_trades}")
    print(f"Winning Trades:        {results.winning_trades}")
    print(f"Losing Trades:         {results.losing_trades}")
    print(f"Win Rate:              {results.win_rate:.2f}%")
    print(f"Profit Factor:         {results.profit_factor:.2f}")
    print(f"Total P&L:             {results.total_pnl:.2f}")
    print(f"Total Return:          {results.total_return_percent:.2f}%")
    print(f"Max Drawdown:          {results.max_drawdown_percent:.2f}%")
    print(f"Sharpe Ratio:          {results.sharpe_ratio:.2f}")
    
    print(f"\n{'Trade Statistics':^60}")
    print("-" * 60)
    print(f"Average Win:           {results.avg_win:.2f}")
    print(f"Average Loss:          {results.avg_loss:.2f}")
    print(f"Avg Trade Duration:    {results.avg_trade_duration_days:.1f} days")
    print(f"Max Consecutive Wins:  {results.max_consecutive_wins}")
    print(f"Max Consecutive Losses:{results.max_consecutive_losses}")
    
    print("\n" + "=" * 60)


def export_results_json(results: BacktestResults, symbol: str, filename: str) -> None:
    """Export results to JSON file."""
    output = {
        "symbol": symbol,
        "summary": {
            "total_trades": results.total_trades,
            "winning_trades": results.winning_trades,
            "losing_trades": results.losing_trades,
            "win_rate": round(results.win_rate, 2),
            "profit_factor": round(results.profit_factor, 2),
            "total_pnl": round(results.total_pnl, 2),
            "total_return_percent": round(results.total_return_percent, 2),
            "max_drawdown_percent": round(results.max_drawdown_percent, 2),
            "sharpe_ratio": round(results.sharpe_ratio, 2),
            "avg_win": round(results.avg_win, 2),
            "avg_loss": round(results.avg_loss, 2),
            "max_consecutive_wins": results.max_consecutive_wins,
            "max_consecutive_losses": results.max_consecutive_losses,
            "avg_trade_duration_days": round(results.avg_trade_duration_days, 1)
        },
        "trades": [
            {
                "entry_date": t.entry_date,
                "exit_date": t.exit_date,
                "direction": t.direction,
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "stop_loss": round(t.stop_loss, 2),
                "take_profit": round(t.take_profit, 2),
                "pnl": round(t.pnl, 2),
                "pnl_percent": round(t.pnl_percent, 2),
                "exit_reason": t.exit_reason
            }
            for t in results.trades
        ]
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Results exported to {filename}")


def main():
    """Main entry point for backtesting."""
    parser = argparse.ArgumentParser(description='UMCI Backtesting Framework')
    parser.add_argument('--symbol', type=str, default='NIFTY50', 
                       help='Symbol to backtest (default: NIFTY50)')
    parser.add_argument('--start', type=str, default='2021-01-01',
                       help='Start date YYYY-MM-DD (default: 2021-01-01)')
    parser.add_argument('--end', type=str, default='2026-01-31',
                       help='End date YYYY-MM-DD (default: 2026-01-31)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (optional)')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital (default: 100000)')
    
    args = parser.parse_args()
    
    print(f"\nUMCI Backtesting Framework v1.0")
    print(f"Symbol: {args.symbol}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Initial Capital: {args.capital:,.2f}")
    
    # Generate sample data (replace with actual data in production)
    print("\nLoading data...")
    data = generate_sample_data(args.symbol, args.start, args.end)
    print(f"Loaded {len(data)} bars of data")
    
    # Configure backtest
    config = BacktestConfig(initial_capital=args.capital)
    
    # Run backtest
    print("\nRunning backtest...")
    results = run_backtest(data, config)
    
    # Print results
    print_results(results, args.symbol)
    
    # Export if requested
    if args.output:
        export_results_json(results, args.symbol, args.output)
    
    return results


if __name__ == "__main__":
    main()
