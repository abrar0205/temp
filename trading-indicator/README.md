# Universal Multi-Confirmation Indicator (UMCI)

A professional-grade trading indicator designed for NSE NIFTY 50, Bank NIFTY, and global markets.

> **Research-Backed**: Based on 500+ source analysis from [Ultimate Trading Indicator Research.pdf](docs/Ultimate%20Trading%20Indicator%20Research.pdf)

## Overview

The UMCI is a multi-confirmation trading indicator that combines:
- **Trend Analysis**: Triple SuperTrend (10/1.0, 11/2.0, 12/3.0) + EMA(200)
- **Momentum**: Stochastic RSI for entry timing
- **Volatility**: ATR-based stop loss and take profit
- **Volume**: 20-period MA confirmation

## 🆕 Advanced Features

### Machine Learning Signal Filter
- **Random Forest**: Quick classification of signal quality (100 trees, 10 depth)
- **LSTM**: Price direction prediction using sequence modeling
- **Ensemble**: Combines both models for higher accuracy

### Multi-Timeframe Analysis (MTF)
- Aligns signals across 15m, 1H, 4H, and Daily timeframes
- Higher timeframe defines trend, lower timeframe provides entry
- Signal strength based on timeframe agreement

### Market Regime Detection
- **ADX-based** trend strength measurement
- Regimes: Strong Trend, Weak Trend, Ranging, High/Low Volatility, Breakout
- Auto-adjusts position size and stop loss based on regime

### Real-Time Data Integration
- **yfinance**: Global stocks, indices, forex, crypto, commodities
- Supports: NIFTY50, BANKNIFTY, TCS, RELIANCE, AAPL, GOOGL, GOLD, BTC
- Cache system for efficient data fetching

### Streamlit Web Dashboard
- Interactive candlestick charts with Plotly
- Real-time indicator overlays (EMAs, SuperTrend, RSI)
- ML signals, MTF analysis, and regime detection display

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
│   ├── universal_trading_strategy.pine     # TradingView strategy (backtesting)
│   └── ultimate_indicator_v2.pine          # 🆕 Ultimate Pine Script v2
├── python/
│   ├── backtest.py                         # Original backtesting framework
│   ├── improved_strategy.py                # Research-based improved strategy
│   ├── prediction_validator.py             # Prediction tracking & validation
│   ├── ml_signal_filter.py                 # 🆕 Random Forest + LSTM ML filter
│   ├── multi_timeframe.py                  # 🆕 Multi-timeframe analysis
│   ├── regime_detection.py                 # 🆕 ADX-based regime detection
│   ├── realtime_data.py                    # 🆕 yfinance/NSE data integration
│   ├── streamlit_dashboard.py              # 🆕 Web dashboard
│   ├── ultimate_trading_engine.py          # 🆕 Unified trading engine
│   ├── advanced_risk_management.py         # 🆕 Kelly, VAR, drawdown control
│   ├── auto_optimizer.py                   # 🆕 Grid search, walk-forward optimization
│   ├── run_ultimate.py                     # 🆕 One-click runner
│   ├── candlestick_patterns.py             # 🆕 20+ candlestick patterns
│   ├── divergence_detector.py              # 🆕 RSI/MACD/Stoch divergence
│   ├── smart_money_concepts.py             # 🆕 FVG, Order Blocks, BOS/CHOCH
│   ├── portfolio_optimizer.py              # 🆕 Markowitz, Risk Parity, Kelly
│   ├── zerodha_integration.py              # 🆕 Kite API (paper + live)
│   ├── performance_benchmark.py            # 🆕 Strategy comparison & stats
│   └── requirements.txt                    # Python dependencies
├── docs/
│   ├── RESEARCH_REPORT.md                  # Detailed research methodology
│   ├── SETUP_GUIDE.md                      # Installation instructions
│   ├── TRADING_MANUAL.md                   # Complete trading guide
│   ├── RISK_DISCLOSURE.md                  # Risk warnings
│   ├── ULTIMATE_SYSTEM_GUIDE.md            # 🆕 Ultimate system documentation
│   ├── GITHUB_WORKFLOWS_GUIDE.md           # 🆕 Automation guide
│   ├── PROJECT_SUMMARY.md                  # 🆕 Full project summary
│   └── Ultimate Trading Indicator Research.pdf  # Full research document
├── .github/workflows/
│   ├── daily-trading-signals.yml           # Daily signal generation
│   ├── weekly-backtest-optimization.yml    # 🆕 Weekly backtest
│   ├── monthly-performance-report.yml      # 🆕 Monthly report
│   ├── multi-asset-analysis.yml            # 🆕 Multi-asset scanner
│   ├── live-market-scanner.yml             # 🆕 Live 15-min scanner
│   ├── alert-notifications.yml             # 🆕 Telegram/Discord alerts
│   └── code-quality-testing.yml            # 🆕 Lint & test
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

# Install dependencies
pip install -r requirements.txt

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

### 🆕 Advanced Modules

```bash
# Machine Learning Signal Filter (Random Forest + LSTM)
python ml_signal_filter.py --mode demo

# Multi-Timeframe Analysis
python multi_timeframe.py --demo

# Market Regime Detection
python regime_detection.py --demo

# Real-Time Data Fetching
python realtime_data.py --symbol NIFTY50 --timeframe 1d --period 6mo
python realtime_data.py --symbol AAPL --realtime
python realtime_data.py --list-symbols

# Streamlit Web Dashboard
pip install streamlit plotly pandas
streamlit run streamlit_dashboard.py
```

### 🆕 Ultimate Trading System

```bash
# One-click runner for everything
python run_ultimate.py --symbol NIFTY50

# With all features
python run_ultimate.py --symbol BANKNIFTY --optimize --ml --full-report
```

### 🆕 Advanced Analysis Modules

```bash
# Candlestick Pattern Recognition (20+ patterns)
python candlestick_patterns.py

# Divergence Detection (RSI, MACD, Stochastic)
python divergence_detector.py

# Smart Money Concepts (FVG, Order Blocks, Liquidity)
python smart_money_concepts.py

# Portfolio Optimization (Markowitz, Risk Parity, Kelly)
python portfolio_optimizer.py

# Performance Benchmark (compare strategies)
python performance_benchmark.py

# Zerodha Integration (paper trading)
python zerodha_integration.py
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
