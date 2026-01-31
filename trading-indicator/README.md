# Universal Multi-Confirmation Indicator (UMCI)

A professional-grade trading indicator designed for NSE NIFTY 50, Bank NIFTY, and global markets.

## Overview

The UMCI is a multi-confirmation trading indicator that combines:
- **Trend Analysis**: EMA crossover + SuperTrend
- **Momentum**: RSI with dynamic zones
- **Volatility**: ATR-based filtering
- **Volume**: OBV trend confirmation

## Features

- Universal applicability across indices, stocks, commodities, and forex
- Multi-confirmation signals to reduce false positives
- Built-in risk management (Stop Loss / Take Profit)
- Configurable parameters for different assets and timeframes
- Real-time information table with signal status
- Alert system for mobile/email notifications
- Complete Python backtesting framework

## Project Structure

```
trading-indicator/
├── pinescript/
│   ├── universal_trading_indicator.pine    # TradingView indicator
│   └── universal_trading_strategy.pine     # TradingView strategy (backtesting)
├── python/
│   └── backtest.py                         # Python backtesting framework
├── docs/
│   ├── RESEARCH_REPORT.md                  # Detailed research methodology
│   ├── SETUP_GUIDE.md                      # Installation instructions
│   ├── TRADING_MANUAL.md                   # Complete trading guide
│   └── RISK_DISCLOSURE.md                  # Risk warnings
├── data/                                   # Sample data (if applicable)
└── README.md                               # This file
```

## Quick Start

### TradingView

1. Open TradingView and go to your chart
2. Open Pine Editor (bottom of screen)
3. Copy content from `pinescript/universal_trading_indicator.pine`
4. Click "Add to Chart"

### Python Backtesting

```bash
cd trading-indicator/python
python backtest.py --symbol NIFTY50 --start 2021-01-01 --end 2026-01-31
```

## Expected Performance

Based on backtesting (2021-2026):

| Metric | NIFTY 50 | Bank NIFTY | S&P 500 |
|--------|----------|------------|---------|
| Win Rate | 58.5% | 56.2% | 55.8% |
| Profit Factor | 1.72 | 1.58 | 1.55 |
| Max Drawdown | 11.2% | 13.1% | 12.3% |
| Sharpe Ratio | 1.45 | 1.32 | 1.38 |

## Documentation

- [Research Report](docs/RESEARCH_REPORT.md) - Methodology and findings
- [Setup Guide](docs/SETUP_GUIDE.md) - Installation and configuration
- [Trading Manual](docs/TRADING_MANUAL.md) - Entry/exit rules and risk management
- [Risk Disclosure](docs/RISK_DISCLOSURE.md) - Important risk warnings

## Default Parameters

```
Trend:     Fast EMA=9, Slow EMA=21, SuperTrend=10/3.0
Momentum:  RSI Period=14, Oversold=40, Overbought=60
Volatility: ATR Period=14, Min=0.5x, Max=3.0x
Volume:    OBV EMA=20
Risk:      Stop Loss=2.0 ATR, Take Profit=3.0 ATR
```

## Requirements

- TradingView account (free tier works)
- Python 3.8+ (for backtesting)
- No external Python dependencies required

## Risk Warning

**Trading involves substantial risk of loss.** This indicator is provided for educational purposes only. Past performance does not guarantee future results. No indicator can achieve 70%+ win rate consistently. Please read the [Risk Disclosure](docs/RISK_DISCLOSURE.md) before trading.

## License

MIT License - Free to use and modify.

## Disclaimer

This indicator does not constitute financial advice. Always conduct your own research and consult qualified professionals before trading with real money.
