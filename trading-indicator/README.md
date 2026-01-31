# Universal Multi-Confirmation Indicator (UMCI)

A professional-grade trading indicator designed for NSE NIFTY 50, Bank NIFTY, and global markets.

> **Research-Backed**: Based on 500+ source analysis from [Ultimate Trading Indicator Research.pdf](docs/Ultimate%20Trading%20Indicator%20Research.pdf)

## Overview

The UMCI is a multi-confirmation trading indicator that combines:
- **Trend Analysis**: Triple SuperTrend (10/1.0, 11/2.0, 12/3.0) + EMA(200)
- **Momentum**: Stochastic RSI for entry timing
- **Volatility**: ATR-based stop loss and take profit
- **Volume**: 20-period MA confirmation

## Features

- Research-validated strategy with realistic expectations (42-48% win rate, 1.8-2.2 profit factor)
- Multi-confirmation signals (EMA + 2/3 SuperTrends + Stochastic RSI)
- Built-in risk management (1.5x ATR stop loss, 2:1 reward-risk)
- Monte Carlo simulation for robustness testing
- Walk-forward analysis for overfitting detection
- **Automated prediction validation system**
- GitHub Actions workflow for daily signal generation

## Project Structure

```
trading-indicator/
├── pinescript/
│   ├── universal_trading_indicator.pine    # TradingView indicator
│   └── universal_trading_strategy.pine     # TradingView strategy (backtesting)
├── python/
│   ├── backtest.py                         # Original backtesting framework
│   ├── improved_strategy.py                # Research-based improved strategy
│   └── prediction_validator.py             # Prediction tracking & validation
├── docs/
│   ├── RESEARCH_REPORT.md                  # Detailed research methodology
│   ├── SETUP_GUIDE.md                      # Installation instructions
│   ├── TRADING_MANUAL.md                   # Complete trading guide
│   ├── RISK_DISCLOSURE.md                  # Risk warnings
│   └── Ultimate Trading Indicator Research.pdf  # Full research document
└── predictions/                            # Tracked predictions & reports
```

## Quick Start

### TradingView

1. Open TradingView and go to your chart
2. Open Pine Editor (bottom of screen)
3. Copy content from `pinescript/universal_trading_indicator.pine`
4. Click "Add to Chart"

### Python Backtesting (Improved Strategy)

```bash
cd trading-indicator/python

# Run improved strategy with Monte Carlo and walk-forward analysis
python improved_strategy.py --symbol NIFTY50 --start 2022-01-01 --end 2025-12-31 \
    --monte-carlo --walk-forward --output results.json

# Generate trading prediction
python prediction_validator.py --mode generate --symbol NIFTY50 --storage predictions.json

# Validate predictions
python prediction_validator.py --mode validate --symbol NIFTY50 --storage predictions.json

# Generate performance report
python prediction_validator.py --mode report --symbol NIFTY50 --storage predictions.json
```

### Automated Daily Signals (GitHub Actions)

The repository includes a GitHub Actions workflow that:
- Generates trading signals daily at 9:00 AM IST (before market opens)
- Validates previous predictions against actual outcomes
- Generates performance reports
- Commits results to the `predictions/` folder

To enable: Go to repository Settings → Actions → Allow all actions

## Expected Performance (Research-Based)

Based on academic research and backtesting (2022-2025):

| Metric | Research Target | Our Results |
|--------|-----------------|-------------|
| Win Rate | 42-48% | 45-50% |
| Profit Factor | 1.8-2.2 | 1.7-2.0 |
| Sharpe Ratio | 0.7-1.0 | 0.8-1.2 |
| Max Drawdown | 25-35% | 15-25% |

**Important**: These are realistic expectations, not marketing hype. No indicator achieves 70%+ win rate consistently.

## Documentation

- [Research Report](docs/RESEARCH_REPORT.md) - Methodology and findings
- [Setup Guide](docs/SETUP_GUIDE.md) - Installation and configuration
- [Trading Manual](docs/TRADING_MANUAL.md) - Entry/exit rules and risk management
- [Risk Disclosure](docs/RISK_DISCLOSURE.md) - Important risk warnings

## Default Parameters (Research-Optimized)

```
Trend:      EMA(200), SuperTrend(10/1.0, 11/2.0, 12/3.0)
Entry:      Stochastic RSI (14, 3, 3), Oversold=28, Overbought=78
Risk:       Stop Loss=1.5 ATR, Take Profit=3.0 ATR (2:1 reward-risk)
Position:   2% risk per trade
Costs:      0.1% round-trip (brokerage + STT + slippage)
```

## Key Research Findings

From the [Ultimate Trading Indicator Research.pdf](docs/Ultimate%20Trading%20Indicator%20Research.pdf):

1. **Win Rate vs Profitability**: 70%+ win rate is achievable with tight targets but often leads to negative expectancy. Optimal: 42-48% win rate with 1.5:1+ reward-risk.

2. **EMA(200) as Trend Filter**: Improves Sharpe ratio by ~2x (from 0.43 to 0.91)

3. **Multi-Confirmation Reduces False Signals**: Triple SuperTrend + Stochastic RSI reduces false signals by 60-70%

4. **Transaction Costs Are Critical**: 0.1% minimum round-trip costs can eliminate edge entirely for frequent traders

5. **Market Regime Matters**: Strategy works in trending markets (50% of time), breakeven in sideways markets

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
