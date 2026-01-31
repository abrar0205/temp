# Complete Trading Indicator Project - All Code

## Overview

This document contains all code created for the research-based trading indicator project. Copy any section to share with another AI for improvements.

---

## 1. TradingView Pine Script - Indicator Version (v5)

Copy this entire code block and paste in TradingView Pine Editor:

```pinescript
// Universal Multi-Confirmation Trading Indicator v1.0
// Designed for NSE NIFTY 50, Bank NIFTY, and Global Markets
// TradingView Pine Script v5
// 
// Based on extensive backtesting research combining:
// - Trend Analysis (EMA crossovers, SuperTrend)
// - Momentum Confirmation (RSI with dynamic zones)
// - Volatility Filtering (ATR-based)
// - Volume Confirmation (OBV trend)

//@version=5
indicator("Universal Multi-Confirmation Indicator [v1.0]", shorttitle="UMCI", overlay=true)

// ============================================================================
// INPUT PARAMETERS
// ============================================================================

// Trend Parameters
trendGroup = "Trend Settings"
fastEMA = input.int(9, title="Fast EMA Period", minval=2, maxval=50, group=trendGroup)
slowEMA = input.int(21, title="Slow EMA Period", minval=5, maxval=100, group=trendGroup)
superTrendPeriod = input.int(10, title="SuperTrend Period", minval=1, maxval=50, group=trendGroup)
superTrendMult = input.float(3.0, title="SuperTrend Multiplier", minval=0.5, maxval=5.0, step=0.1, group=trendGroup)

// Momentum Parameters
momentumGroup = "Momentum Settings"
rsiPeriod = input.int(14, title="RSI Period", minval=2, maxval=50, group=momentumGroup)
rsiOversold = input.int(40, title="RSI Oversold Level", minval=10, maxval=50, group=momentumGroup)
rsiOverbought = input.int(60, title="RSI Overbought Level", minval=50, maxval=90, group=momentumGroup)

// Volatility Parameters
volatilityGroup = "Volatility Settings"
atrPeriod = input.int(14, title="ATR Period", minval=1, maxval=50, group=volatilityGroup)
atrMinMultiplier = input.float(0.5, title="ATR Min Multiplier (Filter Low Vol)", minval=0.1, maxval=2.0, step=0.1, group=volatilityGroup)
atrMaxMultiplier = input.float(3.0, title="ATR Max Multiplier (Filter High Vol)", minval=1.0, maxval=5.0, step=0.1, group=volatilityGroup)

// Volume Parameters
volumeGroup = "Volume Settings"
useVolumeFilter = input.bool(true, title="Use Volume Confirmation", group=volumeGroup)
obvEMAPeriod = input.int(20, title="OBV EMA Period", minval=5, maxval=50, group=volumeGroup)

// Risk Management Parameters
riskGroup = "Risk Management"
stopLossATRMult = input.float(2.0, title="Stop Loss (ATR Multiplier)", minval=0.5, maxval=5.0, step=0.1, group=riskGroup)
takeProfitATRMult = input.float(3.0, title="Take Profit (ATR Multiplier)", minval=1.0, maxval=10.0, step=0.5, group=riskGroup)

// Display Settings
displayGroup = "Display Settings"
showSignals = input.bool(true, title="Show Buy/Sell Signals", group=displayGroup)
showTrendBackground = input.bool(true, title="Show Trend Background", group=displayGroup)
showSLTP = input.bool(true, title="Show SL/TP Levels", group=displayGroup)
showInfoTable = input.bool(true, title="Show Info Table", group=displayGroup)

// ============================================================================
// TREND CALCULATIONS
// ============================================================================

// EMA Calculations
emaFast = ta.ema(close, fastEMA)
emaSlow = ta.ema(close, slowEMA)

// EMA Trend Direction
emaTrendUp = emaFast > emaSlow
emaTrendDown = emaFast < emaSlow

// SuperTrend Calculation
atr = ta.atr(superTrendPeriod)
upperBand = hl2 + (superTrendMult * atr)
lowerBand = hl2 - (superTrendMult * atr)

var float superTrendUpperBand = na
var float superTrendLowerBand = na
var int superTrendDirection = 1

superTrendLowerBand := close[1] > nz(superTrendLowerBand[1], lowerBand) ? math.max(lowerBand, nz(superTrendLowerBand[1], lowerBand)) : lowerBand
superTrendUpperBand := close[1] < nz(superTrendUpperBand[1], upperBand) ? math.min(upperBand, nz(superTrendUpperBand[1], upperBand)) : upperBand

superTrendDirection := close > nz(superTrendUpperBand[1], upperBand) ? 1 : close < nz(superTrendLowerBand[1], lowerBand) ? -1 : nz(superTrendDirection[1], 1)

superTrend = superTrendDirection == 1 ? superTrendLowerBand : superTrendUpperBand
superTrendBullish = superTrendDirection == 1

// Combined Trend Signal (EMA + SuperTrend agreement)
strongUptrend = emaTrendUp and superTrendBullish
strongDowntrend = emaTrendDown and not superTrendBullish

// ============================================================================
// MOMENTUM CALCULATIONS
// ============================================================================

// RSI Calculation
rsi = ta.rsi(close, rsiPeriod)

// RSI Conditions
rsiBullish = rsi > rsiOversold and rsi < rsiOverbought and rsi > rsi[1]
rsiBearish = rsi > rsiOversold and rsi < rsiOverbought and rsi < rsi[1]
rsiOversoldCondition = rsi < rsiOversold
rsiOverboughtCondition = rsi > rsiOverbought

// ============================================================================
// VOLATILITY FILTER
// ============================================================================

// ATR for volatility measurement
atrVolatility = ta.atr(atrPeriod)
atr20 = ta.sma(atrVolatility, 20)

// Volatility within acceptable range
atrRatio = atrVolatility / atr20
volatilityOK = atrRatio >= atrMinMultiplier and atrRatio <= atrMaxMultiplier

// ============================================================================
// VOLUME CALCULATIONS
// ============================================================================

// OBV Calculation
obv = ta.obv
obvEMA = ta.ema(obv, obvEMAPeriod)

// Volume Confirmation
volumeConfirmBull = not useVolumeFilter or (obv > obvEMA)
volumeConfirmBear = not useVolumeFilter or (obv < obvEMA)

// ============================================================================
// SIGNAL GENERATION (Multi-Confirmation)
// ============================================================================

// Buy Signal Requirements:
// 1. Strong uptrend (EMA + SuperTrend)
// 2. RSI not overbought and improving
// 3. Volatility within acceptable range
// 4. Volume confirmation (OBV above EMA)
buySignal = strongUptrend and rsiBullish and volatilityOK and volumeConfirmBull and not strongUptrend[1]

// Sell Signal Requirements:
// 1. Strong downtrend (EMA + SuperTrend)
// 2. RSI not oversold and declining
// 3. Volatility within acceptable range
// 4. Volume confirmation (OBV below EMA)
sellSignal = strongDowntrend and rsiBearish and volatilityOK and volumeConfirmBear and not strongDowntrend[1]

// Exit Signals
exitLong = strongDowntrend or rsiOverboughtCondition
exitShort = strongUptrend or rsiOversoldCondition

// ============================================================================
// STOP LOSS / TAKE PROFIT CALCULATION
// ============================================================================

var float entryPrice = na
var float stopLoss = na
var float takeProfit = na
var int tradeDirection = 0  // 1 = long, -1 = short, 0 = flat

if buySignal
    entryPrice := close
    stopLoss := close - (atrVolatility * stopLossATRMult)
    takeProfit := close + (atrVolatility * takeProfitATRMult)
    tradeDirection := 1
else if sellSignal
    entryPrice := close
    stopLoss := close + (atrVolatility * stopLossATRMult)
    takeProfit := close - (atrVolatility * takeProfitATRMult)
    tradeDirection := -1
else if (tradeDirection == 1 and exitLong) or (tradeDirection == -1 and exitShort)
    entryPrice := na
    stopLoss := na
    takeProfit := na
    tradeDirection := 0

// ============================================================================
// VISUAL DISPLAY
// ============================================================================

// Plot EMAs
plot(emaFast, color=color.new(color.blue, 0), linewidth=1, title="Fast EMA")
plot(emaSlow, color=color.new(color.orange, 0), linewidth=1, title="Slow EMA")

// Plot SuperTrend
plot(superTrend, color=superTrendBullish ? color.green : color.red, linewidth=2, title="SuperTrend")

// Trend Background
bgcolor(showTrendBackground ? (strongUptrend ? color.new(color.green, 90) : strongDowntrend ? color.new(color.red, 90) : na) : na)

// Buy/Sell Signals
plotshape(showSignals and buySignal, title="Buy Signal", location=location.belowbar, style=shape.triangleup, size=size.normal, color=color.green, text="BUY")
plotshape(showSignals and sellSignal, title="Sell Signal", location=location.abovebar, style=shape.triangledown, size=size.normal, color=color.red, text="SELL")

// Exit Signals
plotshape(showSignals and exitLong and tradeDirection[1] == 1, title="Exit Long", location=location.abovebar, style=shape.xcross, size=size.small, color=color.orange, text="EXIT")
plotshape(showSignals and exitShort and tradeDirection[1] == -1, title="Exit Short", location=location.belowbar, style=shape.xcross, size=size.small, color=color.orange, text="EXIT")

// SL/TP Lines
plot(showSLTP and tradeDirection != 0 ? stopLoss : na, color=color.red, style=plot.style_linebr, linewidth=1, title="Stop Loss")
plot(showSLTP and tradeDirection != 0 ? takeProfit : na, color=color.green, style=plot.style_linebr, linewidth=1, title="Take Profit")

// ============================================================================
// INFORMATION TABLE
// ============================================================================

if showInfoTable
    var table infoTable = table.new(position.top_right, 2, 10, bgcolor=color.new(color.black, 80), border_width=1)
    
    table.cell(infoTable, 0, 0, "UMCI v1.0", text_color=color.white, bgcolor=color.new(color.blue, 50))
    table.cell(infoTable, 1, 0, "Status", text_color=color.white, bgcolor=color.new(color.blue, 50))
    
    table.cell(infoTable, 0, 1, "Trend", text_color=color.white)
    table.cell(infoTable, 1, 1, strongUptrend ? "BULLISH" : strongDowntrend ? "BEARISH" : "NEUTRAL", text_color=strongUptrend ? color.green : strongDowntrend ? color.red : color.gray)
    
    table.cell(infoTable, 0, 2, "RSI", text_color=color.white)
    table.cell(infoTable, 1, 2, str.tostring(rsi, "#.##"), text_color=rsi > 60 ? color.orange : rsi < 40 ? color.aqua : color.white)
    
    table.cell(infoTable, 0, 3, "ATR Ratio", text_color=color.white)
    table.cell(infoTable, 1, 3, str.tostring(atrRatio, "#.##"), text_color=volatilityOK ? color.green : color.red)
    
    table.cell(infoTable, 0, 4, "Vol Filter", text_color=color.white)
    table.cell(infoTable, 1, 4, volumeConfirmBull ? "BULLISH" : volumeConfirmBear ? "BEARISH" : "NEUTRAL", text_color=volumeConfirmBull ? color.green : volumeConfirmBear ? color.red : color.gray)
    
    table.cell(infoTable, 0, 5, "Signal", text_color=color.white)
    table.cell(infoTable, 1, 5, buySignal ? "BUY" : sellSignal ? "SELL" : "WAIT", text_color=buySignal ? color.green : sellSignal ? color.red : color.gray)
    
    if tradeDirection != 0
        table.cell(infoTable, 0, 6, "Entry", text_color=color.white)
        table.cell(infoTable, 1, 6, str.tostring(entryPrice, "#.##"), text_color=color.white)
        
        table.cell(infoTable, 0, 7, "Stop Loss", text_color=color.white)
        table.cell(infoTable, 1, 7, str.tostring(stopLoss, "#.##"), text_color=color.red)
        
        table.cell(infoTable, 0, 8, "Take Profit", text_color=color.white)
        table.cell(infoTable, 1, 8, str.tostring(takeProfit, "#.##"), text_color=color.green)
        
        riskReward = math.abs(takeProfit - entryPrice) / math.abs(entryPrice - stopLoss)
        table.cell(infoTable, 0, 9, "Risk:Reward", text_color=color.white)
        table.cell(infoTable, 1, 9, "1:" + str.tostring(riskReward, "#.#"), text_color=color.yellow)

// ============================================================================
// ALERTS
// ============================================================================

alertcondition(buySignal, title="Buy Signal", message="UMCI: Buy Signal on {{ticker}} at {{close}}")
alertcondition(sellSignal, title="Sell Signal", message="UMCI: Sell Signal on {{ticker}} at {{close}}")
alertcondition(exitLong, title="Exit Long", message="UMCI: Exit Long on {{ticker}} at {{close}}")
alertcondition(exitShort, title="Exit Short", message="UMCI: Exit Short on {{ticker}} at {{close}}")
```

---

## 2. TradingView Pine Script - Strategy Version (Backtesting)

```pinescript
// Universal Multi-Confirmation Trading Strategy v1.0
// Backtesting Version for TradingView Strategy Tester
// TradingView Pine Script v5

//@version=5
strategy("Universal Multi-Confirmation Strategy [v1.0]", 
         shorttitle="UMCI Strategy",
         overlay=true, 
         initial_capital=100000,
         default_qty_type=strategy.percent_of_equity,
         default_qty_value=10,
         commission_type=strategy.commission.percent,
         commission_value=0.02,
         slippage=2,
         pyramiding=0)

// ============================================================================
// INPUT PARAMETERS
// ============================================================================

trendGroup = "Trend Settings"
fastEMA = input.int(9, title="Fast EMA Period", minval=2, maxval=50, group=trendGroup)
slowEMA = input.int(21, title="Slow EMA Period", minval=5, maxval=100, group=trendGroup)
superTrendPeriod = input.int(10, title="SuperTrend Period", minval=1, maxval=50, group=trendGroup)
superTrendMult = input.float(3.0, title="SuperTrend Multiplier", minval=0.5, maxval=5.0, step=0.1, group=trendGroup)

momentumGroup = "Momentum Settings"
rsiPeriod = input.int(14, title="RSI Period", minval=2, maxval=50, group=momentumGroup)
rsiOversold = input.int(40, title="RSI Oversold Level", minval=10, maxval=50, group=momentumGroup)
rsiOverbought = input.int(60, title="RSI Overbought Level", minval=50, maxval=90, group=momentumGroup)

volatilityGroup = "Volatility Settings"
atrPeriod = input.int(14, title="ATR Period", minval=1, maxval=50, group=volatilityGroup)
atrMinMultiplier = input.float(0.5, title="ATR Min Multiplier", minval=0.1, maxval=2.0, step=0.1, group=volatilityGroup)
atrMaxMultiplier = input.float(3.0, title="ATR Max Multiplier", minval=1.0, maxval=5.0, step=0.1, group=volatilityGroup)

volumeGroup = "Volume Settings"
useVolumeFilter = input.bool(true, title="Use Volume Confirmation", group=volumeGroup)
obvEMAPeriod = input.int(20, title="OBV EMA Period", minval=5, maxval=50, group=volumeGroup)

riskGroup = "Risk Management"
stopLossATRMult = input.float(2.0, title="Stop Loss (ATR Multiplier)", minval=0.5, maxval=5.0, step=0.1, group=riskGroup)
takeProfitATRMult = input.float(3.0, title="Take Profit (ATR Multiplier)", minval=1.0, maxval=10.0, step=0.5, group=riskGroup)
useTrailingStop = input.bool(false, title="Use Trailing Stop", group=riskGroup)
trailingStopATRMult = input.float(1.5, title="Trailing Stop (ATR Multiplier)", minval=0.5, maxval=3.0, step=0.1, group=riskGroup)

backtestGroup = "Backtest Settings"
startDate = input.time(timestamp("2021-01-01"), title="Start Date", group=backtestGroup)
endDate = input.time(timestamp("2026-12-31"), title="End Date", group=backtestGroup)

// ============================================================================
// CALCULATIONS
// ============================================================================

inDateRange = time >= startDate and time <= endDate

emaFast = ta.ema(close, fastEMA)
emaSlow = ta.ema(close, slowEMA)
emaTrendUp = emaFast > emaSlow
emaTrendDown = emaFast < emaSlow

atr = ta.atr(superTrendPeriod)
upperBand = hl2 + (superTrendMult * atr)
lowerBand = hl2 - (superTrendMult * atr)

var float superTrendUpperBand = na
var float superTrendLowerBand = na
var int superTrendDirection = 1

superTrendLowerBand := close[1] > nz(superTrendLowerBand[1], lowerBand) ? math.max(lowerBand, nz(superTrendLowerBand[1], lowerBand)) : lowerBand
superTrendUpperBand := close[1] < nz(superTrendUpperBand[1], upperBand) ? math.min(upperBand, nz(superTrendUpperBand[1], upperBand)) : upperBand
superTrendDirection := close > nz(superTrendUpperBand[1], upperBand) ? 1 : close < nz(superTrendLowerBand[1], lowerBand) ? -1 : nz(superTrendDirection[1], 1)

superTrend = superTrendDirection == 1 ? superTrendLowerBand : superTrendUpperBand
superTrendBullish = superTrendDirection == 1

strongUptrend = emaTrendUp and superTrendBullish
strongDowntrend = emaTrendDown and not superTrendBullish

rsi = ta.rsi(close, rsiPeriod)
rsiBullish = rsi > rsiOversold and rsi < rsiOverbought and rsi > rsi[1]
rsiBearish = rsi > rsiOversold and rsi < rsiOverbought and rsi < rsi[1]
rsiOversoldCondition = rsi < rsiOversold
rsiOverboughtCondition = rsi > rsiOverbought

atrVolatility = ta.atr(atrPeriod)
atr20 = ta.sma(atrVolatility, 20)
atrRatio = atrVolatility / atr20
volatilityOK = atrRatio >= atrMinMultiplier and atrRatio <= atrMaxMultiplier

obv = ta.obv
obvEMA = ta.ema(obv, obvEMAPeriod)
volumeConfirmBull = not useVolumeFilter or (obv > obvEMA)
volumeConfirmBear = not useVolumeFilter or (obv < obvEMA)

// ============================================================================
// SIGNALS
// ============================================================================

buySignal = strongUptrend and rsiBullish and volatilityOK and volumeConfirmBull and not strongUptrend[1]
sellSignal = strongDowntrend and rsiBearish and volatilityOK and volumeConfirmBear and not strongDowntrend[1]
exitLong = strongDowntrend or rsiOverboughtCondition
exitShort = strongUptrend or rsiOversoldCondition

// ============================================================================
// STRATEGY EXECUTION
// ============================================================================

longStopLoss = close - (atrVolatility * stopLossATRMult)
longTakeProfit = close + (atrVolatility * takeProfitATRMult)
shortStopLoss = close + (atrVolatility * stopLossATRMult)
shortTakeProfit = close - (atrVolatility * takeProfitATRMult)

if inDateRange
    if buySignal and strategy.position_size == 0
        strategy.entry("Long", strategy.long)
        strategy.exit("Long Exit", "Long", stop=longStopLoss, limit=longTakeProfit)
    
    if sellSignal and strategy.position_size == 0
        strategy.entry("Short", strategy.short)
        strategy.exit("Short Exit", "Short", stop=shortStopLoss, limit=shortTakeProfit)
    
    if strategy.position_size > 0 and exitLong
        strategy.close("Long", comment="Exit Signal")
    
    if strategy.position_size < 0 and exitShort
        strategy.close("Short", comment="Exit Signal")

if useTrailingStop
    trailAmount = atrVolatility * trailingStopATRMult
    strategy.exit("Long Trail", "Long", trail_offset=trailAmount, trail_points=trailAmount)
    strategy.exit("Short Trail", "Short", trail_offset=trailAmount, trail_points=trailAmount)

// ============================================================================
// VISUAL
// ============================================================================

plot(emaFast, color=color.new(color.blue, 0), linewidth=1, title="Fast EMA")
plot(emaSlow, color=color.new(color.orange, 0), linewidth=1, title="Slow EMA")
plot(superTrend, color=superTrendBullish ? color.green : color.red, linewidth=2, title="SuperTrend")
bgcolor(strongUptrend ? color.new(color.green, 90) : strongDowntrend ? color.new(color.red, 90) : na)
plotshape(buySignal, title="Buy Signal", location=location.belowbar, style=shape.triangleup, size=size.normal, color=color.green, text="BUY")
plotshape(sellSignal, title="Sell Signal", location=location.abovebar, style=shape.triangledown, size=size.normal, color=color.red, text="SELL")

// Performance Table
var table perfTable = table.new(position.bottom_right, 2, 8, bgcolor=color.new(color.black, 80), border_width=1)

if barstate.islast
    table.cell(perfTable, 0, 0, "UMCI Strategy", text_color=color.white, bgcolor=color.new(color.blue, 50))
    table.cell(perfTable, 1, 0, "Performance", text_color=color.white, bgcolor=color.new(color.blue, 50))
    
    table.cell(perfTable, 0, 1, "Net Profit", text_color=color.white)
    netProfit = strategy.netprofit
    table.cell(perfTable, 1, 1, str.tostring(netProfit, "#.##"), text_color=netProfit > 0 ? color.green : color.red)
    
    table.cell(perfTable, 0, 2, "Win Rate", text_color=color.white)
    winRate = strategy.wintrades / math.max(strategy.closedtrades, 1) * 100
    table.cell(perfTable, 1, 2, str.tostring(winRate, "#.##") + "%", text_color=winRate >= 50 ? color.green : color.red)
    
    table.cell(perfTable, 0, 3, "Profit Factor", text_color=color.white)
    profitFactor = strategy.grossprofit / math.max(math.abs(strategy.grossloss), 1)
    table.cell(perfTable, 1, 3, str.tostring(profitFactor, "#.##"), text_color=profitFactor >= 1.5 ? color.green : color.red)
    
    table.cell(perfTable, 0, 4, "Total Trades", text_color=color.white)
    table.cell(perfTable, 1, 4, str.tostring(strategy.closedtrades), text_color=color.white)
```

---

## 3. Python - Improved Strategy with Monte Carlo & Walk-Forward

```python
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
    """Configuration based on research PDF recommendations."""
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
    stop_loss_atr_mult: float = 1.5
    take_profit_atr_mult: float = 3.0  # 2:1 reward-risk
    
    # Volume filter
    use_volume_filter: bool = True
    volume_ma_period: int = 20
    
    # Transaction costs (0.1% minimum for India)
    commission_percent: float = 0.05
    slippage_percent: float = 0.05
    
    # Position sizing
    initial_capital: float = 500000.0
    risk_per_trade_percent: float = 2.0
    
    # Required confirmations
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


# ============================================================================
# INDICATOR CALCULATIONS
# ============================================================================

def calculate_ema(data: List[float], period: int) -> List[Optional[float]]:
    """Calculate Exponential Moving Average."""
    if len(data) < period:
        return [None] * len(data)
    
    ema = [None] * (period - 1)
    multiplier = 2 / (period + 1)
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
    """Calculate Stochastic RSI (K and D lines)."""
    rsi = calculate_rsi(closes, rsi_period)
    
    k_line = []
    for i in range(len(rsi)):
        if i < stoch_period - 1 or rsi[i] is None:
            k_line.append(None)
            continue
        
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
    
    k_smoothed = calculate_sma(k_line, smooth_k)
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
    """Calculate SuperTrend indicator."""
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
        
        if closes[i-1] > upper_band[i-1]:
            direction[i] = 1
        elif closes[i-1] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
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
# MONTE CARLO SIMULATION
# ============================================================================

def run_monte_carlo(trades: List[TradeResult], initial_capital: float, 
                    num_simulations: int = 1000, skip_percentage: float = 0.10) -> Dict:
    """Run Monte Carlo simulation to stress test the strategy."""
    if not trades:
        return {"error": "No trades for Monte Carlo"}
    
    drawdowns = []
    final_capitals = []
    
    for _ in range(num_simulations):
        shuffled = trades.copy()
        random.shuffle(shuffled)
        
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
        "drawdown_median": drawdowns[len(drawdowns) // 2],
        "drawdown_95th": drawdowns[int(len(drawdowns) * 0.95)],
        "probability_of_profit": sum(1 for c in final_capitals if c > initial_capital) / len(final_capitals) * 100
    }


# ============================================================================
# WALK-FORWARD ANALYSIS
# ============================================================================

def walk_forward_analysis(data: List[Dict], config: ImprovedBacktestConfig,
                          train_ratio: float = 0.7) -> Dict:
    """Perform walk-forward analysis."""
    split_idx = int(len(data) * train_ratio)
    
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    train_results = run_improved_backtest(train_data, config)
    test_results = run_improved_backtest(test_data, config)
    
    if train_results.sharpe_ratio > 0 and test_results.sharpe_ratio:
        sharpe_degradation = (train_results.sharpe_ratio - test_results.sharpe_ratio) / train_results.sharpe_ratio * 100
    else:
        sharpe_degradation = 100
    
    return {
        "train_sharpe": train_results.sharpe_ratio,
        "test_sharpe": test_results.sharpe_ratio,
        "sharpe_degradation_percent": sharpe_degradation,
        "is_overfitted": sharpe_degradation > 50
    }


# Usage:
# python improved_strategy.py --symbol NIFTY50 --start 2022-01-01 --end 2025-12-31 --monte-carlo --walk-forward
```

---

## 4. Python - Prediction Validator (Track Signals vs Outcomes)

```python
"""
Prediction Validation System
Generates trading signals and tracks their accuracy over time
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import argparse


@dataclass
class Prediction:
    """A single trading signal prediction."""
    timestamp: str
    symbol: str
    signal: str  # 'long', 'short', 'neutral'
    confidence: float  # 0-100
    price_at_prediction: float
    target_price: float
    stop_loss: float
    expected_direction: str  # 'up', 'down', 'sideways'
    indicators: Dict
    actual_outcome: Optional[str] = None  # 'correct', 'incorrect', 'pending'
    actual_price: Optional[float] = None
    validated_at: Optional[str] = None
    pnl_percent: Optional[float] = None


class PredictionValidator:
    """Validates trading predictions against actual market outcomes."""
    
    def __init__(self, storage_path: str = "predictions.json"):
        self.storage_path = storage_path
        self.predictions: List[Prediction] = []
        self.load_predictions()
    
    def load_predictions(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.predictions = [Prediction(**p) for p in data.get('predictions', [])]
    
    def save_predictions(self):
        with open(self.storage_path, 'w') as f:
            json.dump({
                'last_updated': datetime.now().isoformat(),
                'predictions': [asdict(p) for p in self.predictions]
            }, f, indent=2)
    
    def validate_predictions(self, current_data: Dict) -> List[Prediction]:
        """Validate pending predictions against current market data."""
        validated = []
        current_price = current_data['close']
        
        for pred in self.predictions:
            if pred.actual_outcome is not None or pred.signal == 'neutral':
                continue
            
            pred_time = datetime.fromisoformat(pred.timestamp)
            if datetime.now() - pred_time < timedelta(hours=6):
                continue
            
            if pred.signal == 'long':
                if current_price >= pred.target_price:
                    pred.actual_outcome = 'correct'
                elif current_price <= pred.stop_loss:
                    pred.actual_outcome = 'incorrect'
                pred.pnl_percent = (current_price - pred.price_at_prediction) / pred.price_at_prediction * 100
            
            elif pred.signal == 'short':
                if current_price <= pred.target_price:
                    pred.actual_outcome = 'correct'
                elif current_price >= pred.stop_loss:
                    pred.actual_outcome = 'incorrect'
                pred.pnl_percent = (pred.price_at_prediction - current_price) / pred.price_at_prediction * 100
            
            if pred.actual_outcome:
                pred.actual_price = current_price
                pred.validated_at = current_data.get('date', datetime.now().strftime('%Y-%m-%d'))
                validated.append(pred)
        
        self.save_predictions()
        return validated
    
    def get_performance_stats(self) -> Dict:
        """Calculate performance statistics."""
        validated = [p for p in self.predictions if p.actual_outcome and p.signal != 'neutral']
        
        if not validated:
            return {"total_predictions": len(self.predictions), "validated": 0, "accuracy": 0}
        
        correct = sum(1 for p in validated if p.actual_outcome == 'correct')
        
        return {
            "total_predictions": len(self.predictions),
            "validated": len(validated),
            "correct": correct,
            "accuracy": round(correct / len(validated) * 100, 2)
        }


# Usage:
# python prediction_validator.py --mode generate --symbol NIFTY50 --storage predictions.json
# python prediction_validator.py --mode validate --symbol NIFTY50 --storage predictions.json
# python prediction_validator.py --mode report --symbol NIFTY50 --storage predictions.json
```

---

## 5. GitHub Actions Workflow - Daily Signal Generation

Create file: `.github/workflows/daily-trading-signals.yml`

```yaml
name: Daily Trading Signal Generator & Validator

on:
  schedule:
    - cron: '30 3 * * 1-5'  # 9:00 AM IST (before Indian market opens)
  
  workflow_dispatch:
    inputs:
      mode:
        description: 'Mode to run'
        required: true
        default: 'full'
        type: choice
        options:
          - full
          - generate-only
          - validate-only
          - report-only

jobs:
  trading-signals:
    runs-on: ubuntu-latest
    
    permissions:
      contents: write
      
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Generate trading signals
        if: ${{ github.event.inputs.mode != 'validate-only' && github.event.inputs.mode != 'report-only' }}
        working-directory: trading-indicator/python
        run: |
          echo "=== Generating Trading Signals ==="
          python prediction_validator.py --mode generate --symbol NIFTY50 --storage ../../predictions/nifty50.json
          python prediction_validator.py --mode generate --symbol BANKNIFTY --storage ../../predictions/banknifty.json
      
      - name: Validate previous predictions
        if: ${{ github.event.inputs.mode != 'generate-only' }}
        working-directory: trading-indicator/python
        run: |
          echo "=== Validating Previous Predictions ==="
          python prediction_validator.py --mode validate --symbol NIFTY50 --storage ../../predictions/nifty50.json || true
          python prediction_validator.py --mode validate --symbol BANKNIFTY --storage ../../predictions/banknifty.json || true
      
      - name: Generate performance report
        working-directory: trading-indicator/python
        run: |
          mkdir -p ../../predictions/reports
          python prediction_validator.py --mode report --symbol NIFTY50 --storage ../../predictions/nifty50.json --output ../../predictions/reports/nifty50_report.md || true
      
      - name: Commit and push results
        run: |
          git config --local user.email "github-actions[bot]@users.noreply.github.com"
          git config --local user.name "github-actions[bot]"
          mkdir -p predictions/reports
          git add predictions/ || true
          if git diff --staged --quiet; then
            echo "No changes to commit"
          else
            git commit -m "Daily trading signals - $(date -u '+%Y-%m-%d')"
            git push
          fi
```

---

## 6. Key Research Findings (From PDF Analysis)

### Strategy Parameters (Research-Optimized)
```
Trend Filter:    EMA(200) - Improves Sharpe by 2x (0.43 → 0.91)
SuperTrend 1:    Period=10, Multiplier=1.0
SuperTrend 2:    Period=11, Multiplier=2.0  
SuperTrend 3:    Period=12, Multiplier=3.0
Entry Timing:    Stochastic RSI (14, 3, 3), Oversold=28, Overbought=78
Stop Loss:       1.5x ATR
Take Profit:     3.0x ATR (2:1 reward-risk)
Risk Per Trade:  2% of capital
Transaction Cost: 0.1% round-trip minimum
```

### Realistic Performance Expectations
| Metric | Marketing Claims | Research Reality |
|--------|------------------|------------------|
| Win Rate | 70%+ | 42-48% |
| Profit Factor | 3.0+ | 1.8-2.2 |
| Sharpe Ratio | 2.0+ | 0.7-1.0 |
| Max Drawdown | 5% | 25-35% |

### Key Insight
> "45% win rate with 2:1 reward-risk is MORE profitable than 70% win rate with 1:1"

---

## 7. Suggested Improvements for Another AI

Ask another AI to help with:

1. **Machine Learning Enhancement**
   - Random Forest for signal classification
   - LSTM for price direction prediction
   - Feature importance analysis

2. **Multi-Timeframe Analysis**
   - Align 15m signals with 1h and Daily trends
   - Reject signals when timeframes conflict

3. **Regime Detection**
   - ADX-based trending vs ranging detection
   - Different parameters for different regimes

4. **Real-Time Data Integration**
   - yfinance for historical data
   - NSE API for live NIFTY data
   - Zerodha Kite for order execution

5. **Web Dashboard**
   - Streamlit for visualization
   - Real-time signal display
   - Performance tracking charts

---

## Files in This Project

```
trading-indicator/
├── pinescript/
│   ├── universal_trading_indicator.pine    # TradingView indicator
│   └── universal_trading_strategy.pine     # TradingView strategy
├── python/
│   ├── backtest.py                         # Original backtest engine
│   ├── improved_strategy.py                # Research-based strategy
│   ├── prediction_validator.py             # Signal tracking system
│   ├── real_data_backtest.py               # Real data fetcher
│   └── real_backtest_public_data.py        # AAPL backtest
├── docs/
│   ├── RESEARCH_REPORT.md
│   ├── SETUP_GUIDE.md
│   ├── TRADING_MANUAL.md
│   ├── RISK_DISCLOSURE.md
│   └── PROJECT_SUMMARY.md
└── .github/workflows/
    └── daily-trading-signals.yml           # Automation
```

---

*Document generated for sharing with another AI for improvement suggestions*
