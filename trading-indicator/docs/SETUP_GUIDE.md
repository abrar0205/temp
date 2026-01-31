# UMCI Setup Guide

## Universal Multi-Confirmation Indicator - Installation & Configuration

---

## Table of Contents

1. [TradingView Setup](#1-tradingview-setup)
2. [Python Backtesting Setup](#2-python-backtesting-setup)
3. [Configuration Guide](#3-configuration-guide)
4. [Alert Setup](#4-alert-setup)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. TradingView Setup

### 1.1 Adding the Indicator

**Step 1**: Open TradingView and go to your chart

**Step 2**: Click on "Indicators" (fx icon) at the top of the chart

**Step 3**: Click on "Pine Editor" at the bottom of the screen

**Step 4**: Copy the entire content from `pinescript/universal_trading_indicator.pine`

**Step 5**: Paste it into the Pine Editor

**Step 6**: Click "Add to Chart"

**Step 7**: The indicator should now appear on your chart

### 1.2 Adding the Strategy (for Backtesting)

**Step 1**: Follow steps 1-3 above

**Step 2**: Copy the content from `pinescript/universal_trading_strategy.pine`

**Step 3**: Paste it into a new Pine Editor tab

**Step 4**: Click "Add to Chart"

**Step 5**: Open "Strategy Tester" tab at the bottom to see backtest results

### 1.3 Saving the Script

**Step 1**: In Pine Editor, click the dropdown arrow next to "Save"

**Step 2**: Select "Save As..."

**Step 3**: Name your script (e.g., "UMCI Indicator")

**Step 4**: The script will be saved to your TradingView account

---

## 2. Python Backtesting Setup

### 2.1 Requirements

- Python 3.8 or higher
- No external dependencies required for basic functionality

### 2.2 Installation

```bash
# Navigate to the trading-indicator directory
cd trading-indicator/python

# Run the backtest
python backtest.py --symbol NIFTY50 --start 2021-01-01 --end 2026-01-31
```

### 2.3 Command Line Options

```
Options:
  --symbol    Symbol to backtest (default: NIFTY50)
              Supported: NIFTY50, BANKNIFTY, TCS, RELIANCE, HDFC, SPY, AAPL, GOLD
  
  --start     Start date in YYYY-MM-DD format (default: 2021-01-01)
  
  --end       End date in YYYY-MM-DD format (default: 2026-01-31)
  
  --output    Output JSON file path for results (optional)
  
  --capital   Initial capital (default: 100000)
```

### 2.4 Example Commands

```bash
# Basic backtest on NIFTY50
python backtest.py

# Backtest Bank NIFTY with custom dates
python backtest.py --symbol BANKNIFTY --start 2022-01-01 --end 2024-12-31

# Export results to JSON
python backtest.py --symbol NIFTY50 --output results.json

# Custom initial capital
python backtest.py --capital 500000
```

### 2.5 Using Real Data (Advanced)

To use real market data instead of generated sample data:

```python
# Install yfinance
pip install yfinance

# Modify backtest.py to use yfinance
import yfinance as yf

def load_real_data(symbol, start, end):
    """Load real data from Yahoo Finance."""
    # Map Indian symbols to Yahoo format
    symbol_map = {
        "NIFTY50": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "TCS": "TCS.NS",
        "RELIANCE": "RELIANCE.NS",
        "HDFC": "HDFCBANK.NS",
    }
    
    yf_symbol = symbol_map.get(symbol.upper(), symbol)
    df = yf.download(yf_symbol, start=start, end=end)
    
    data = []
    for idx, row in df.iterrows():
        data.append({
            "date": idx.strftime("%Y-%m-%d"),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": int(row["Volume"])
        })
    
    return data
```

---

## 3. Configuration Guide

### 3.1 Default Parameters

The indicator comes with optimized default parameters:

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Fast EMA | 9 | 5-20 | Short-term trend |
| Slow EMA | 21 | 15-50 | Medium-term trend |
| SuperTrend Period | 10 | 7-14 | Trend calculation period |
| SuperTrend Multiplier | 3.0 | 2.0-4.0 | ATR multiplier for bands |
| RSI Period | 14 | 7-21 | Momentum calculation |
| RSI Oversold | 40 | 30-45 | Buy zone threshold |
| RSI Overbought | 60 | 55-70 | Sell zone threshold |
| ATR Period | 14 | 10-20 | Volatility period |
| ATR Min Multiplier | 0.5 | 0.3-0.8 | Minimum volatility filter |
| ATR Max Multiplier | 3.0 | 2.0-5.0 | Maximum volatility filter |
| OBV EMA Period | 20 | 10-30 | Volume trend period |
| Stop Loss ATR | 2.0 | 1.5-3.0 | Stop loss distance |
| Take Profit ATR | 3.0 | 2.0-5.0 | Take profit distance |

### 3.2 Asset-Specific Settings

#### NIFTY 50 (Default - Optimal)
```
Fast EMA: 9
Slow EMA: 21
SuperTrend: 10, 3.0
RSI: 14 (40/60)
Stop Loss: 2.0 ATR
Take Profit: 3.0 ATR
```

#### Bank NIFTY (Higher Volatility)
```
Fast EMA: 9
Slow EMA: 21
SuperTrend: 10, 2.5  # Lower multiplier for faster signals
RSI: 14 (35/65)  # Wider zones
Stop Loss: 2.5 ATR  # Wider stop
Take Profit: 3.5 ATR
```

#### Individual Stocks (TCS, Reliance, etc.)
```
Fast EMA: 9
Slow EMA: 21
SuperTrend: 10, 3.0
RSI: 14 (40/60)
Stop Loss: 2.0 ATR
Take Profit: 3.0 ATR
```

#### Gold/Commodities
```
Fast EMA: 9
Slow EMA: 21
SuperTrend: 12, 3.5  # Slower, wider
RSI: 14 (35/65)
Stop Loss: 2.5 ATR
Take Profit: 4.0 ATR
```

### 3.3 Timeframe-Specific Settings

#### 5-Minute (Scalping)
```
Fast EMA: 5
Slow EMA: 13
SuperTrend: 7, 2.5
RSI: 7 (35/65)
```

#### 15-Minute (Intraday - Recommended)
```
Use default settings
```

#### 1-Hour (Swing Trading)
```
Fast EMA: 9
Slow EMA: 21
SuperTrend: 10, 3.0
RSI: 14 (40/60)
```

#### Daily (Position Trading)
```
Fast EMA: 12
Slow EMA: 26
SuperTrend: 14, 3.5
RSI: 14 (40/60)
```

---

## 4. Alert Setup

### 4.1 TradingView Alerts

**Step 1**: Right-click on the chart

**Step 2**: Select "Add Alert..."

**Step 3**: In the Condition dropdown, select "UMCI [v1.0]"

**Step 4**: Select the alert type:
- "Buy Signal" - For long entry alerts
- "Sell Signal" - For short entry alerts
- "Exit Long" - For long exit alerts
- "Exit Short" - For short exit alerts

**Step 5**: Configure notification method:
- Email
- Mobile push notification
- SMS (requires TradingView upgrade)
- Webhook (for automated trading)

**Step 6**: Set expiration (recommended: Open-ended)

**Step 7**: Click "Create"

### 4.2 Webhook for Automated Trading

For automated trading via Zerodha Kite or other platforms:

**Step 1**: Create a webhook endpoint (using services like Pipedream, Make.com)

**Step 2**: In TradingView alert settings, enable "Webhook URL"

**Step 3**: Enter your webhook URL

**Step 4**: The alert message format will be:
```
UMCI: Buy Signal on {{ticker}} at {{close}}
```

**Step 5**: Parse this in your webhook handler to execute trades

### 4.3 Multiple Alerts (Recommended Setup)

Create these 4 alerts for complete coverage:

1. **Buy Alert**: Condition = "Buy Signal"
2. **Sell Alert**: Condition = "Sell Signal"
3. **Exit Long Alert**: Condition = "Exit Long"
4. **Exit Short Alert**: Condition = "Exit Short"

---

## 5. Troubleshooting

### 5.1 Common Issues

#### Issue: Indicator not showing on chart
**Solution**: 
- Check if the asset has enough historical data
- Verify the timeframe is supported (minimum 5-minute)
- Try reloading the page

#### Issue: No signals appearing
**Solution**:
- Enable "Show Buy/Sell Signals" in settings
- Check if the current market conditions meet signal criteria
- Verify the asset is liquid (has volume data)

#### Issue: Too many signals
**Solution**:
- Enable volume filter
- Increase RSI thresholds (e.g., 35/65 instead of 40/60)
- Increase SuperTrend multiplier (e.g., 3.5 instead of 3.0)

#### Issue: Too few signals
**Solution**:
- Disable volume filter
- Decrease RSI thresholds (e.g., 45/55)
- Decrease SuperTrend multiplier (e.g., 2.5)

### 5.2 Python Backtest Issues

#### Issue: "Module not found" error
**Solution**:
```bash
# Ensure you're using Python 3.8+
python --version

# Run from the correct directory
cd trading-indicator/python
python backtest.py
```

#### Issue: No trades in backtest
**Solution**:
- Check date range has enough data
- Verify the symbol is supported
- Check console output for errors

### 5.3 Performance Issues

#### Issue: Indicator causing chart lag
**Solution**:
- Reduce the chart history
- Hide unnecessary elements (background, SL/TP lines)
- Use a faster timeframe

---

## Support

For issues or questions:
1. Check this troubleshooting guide
2. Review the Research Report for methodology details
3. Test on paper trading before live trading

---

*Guide Version: 1.0*  
*Last Updated: January 2026*
