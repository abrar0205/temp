# GitHub Workflows Documentation

## Complete Automation Suite for Ultimate Trading System

This project includes a comprehensive set of GitHub Actions workflows that automate all aspects of the trading system.

---

## Workflow Overview

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| Daily Trading Signals | 9:00 AM IST (Mon-Fri) | Generate & validate daily signals |
| Weekly Backtest | Sunday 11:30 AM IST | Full backtest & optimization |
| Monthly Report | 1st of month | Performance analysis |
| Multi-Asset Analysis | Saturday 1:30 PM IST | Cross-asset comparison |
| Live Market Scanner | Every 15 min (market hours) | Real-time signal scanning |
| Alert Notifications | On-demand | Send alerts to Telegram/Discord/Slack |
| Code Quality | On push/PR | Lint, test, validate |

---

## 1. Daily Trading Signals

**File:** `.github/workflows/daily-trading-signals.yml`

### Schedule
- Runs at 9:00 AM IST (3:30 AM UTC) before Indian market opens
- Monday to Friday only

### What it does
1. Generates trading signals for NIFTY50 and Bank NIFTY
2. Validates previous predictions against actual outcomes
3. Creates performance reports
4. Commits results to repository

### Manual Trigger Options
- `full` - Generate + Validate + Report
- `generate-only` - Only generate new signals
- `validate-only` - Only validate past predictions
- `report-only` - Only generate reports

---

## 2. Weekly Backtest & Optimization

**File:** `.github/workflows/weekly-backtest-optimization.yml`

### Schedule
- Runs every Sunday at 11:30 AM IST (6:00 AM UTC)

### What it does
1. Runs comprehensive backtest on multiple symbols
2. Performs strategy optimization (grid search)
3. Runs Monte Carlo simulation (1000 runs by default)
4. Generates weekly report

### Inputs
- `symbols` - Comma-separated list of symbols (default: NIFTY50,BANKNIFTY,RELIANCE.NS,TCS.NS)
- `optimize` - Whether to run optimization (default: true)
- `monte_carlo_runs` - Number of MC runs (default: 1000)

---

## 3. Monthly Performance Report

**File:** `.github/workflows/monthly-performance-report.yml`

### Schedule
- Runs on 1st of each month at 11:30 AM IST (6:00 AM UTC)

### What it does
1. Analyzes full month's performance
2. Compares against targets (win rate, profit factor, drawdown)
3. Generates recommendations
4. Creates detailed markdown report

### Inputs
- `months_back` - Number of months to analyze (default: 3)

---

## 4. Multi-Asset Analysis

**File:** `.github/workflows/multi-asset-analysis.yml`

### Schedule
- Runs every Saturday at 1:30 PM IST (8:00 AM UTC)

### What it does
1. Scans multiple asset categories
2. Ranks assets by strategy performance
3. Identifies top performers
4. Compares performance across markets

### Asset Categories
- `indian-indices` - NIFTY50, Bank NIFTY, NIFTY IT
- `indian-stocks` - Top 10 NSE large caps
- `us-markets` - S&P500, NASDAQ, top US stocks
- `commodities` - Gold, Silver, Crude, Natural Gas
- `forex` - Major currency pairs
- `all` - All categories

---

## 5. Live Market Scanner

**File:** `.github/workflows/live-market-scanner.yml`

### Schedule
- Runs every 15 minutes during market hours
- Indian market: 9:15 AM - 3:30 PM IST
- Automatically detects market status

### What it does
1. Scans specified market for signals
2. Identifies breakouts, reversals, momentum plays
3. Generates real-time alerts
4. Saves scan results

### Scan Types
- `all` - All patterns
- `breakout` - Breakout signals only
- `reversal` - Reversal patterns
- `momentum` - Momentum signals
- `volume-spike` - Volume-based alerts

---

## 6. Alert Notifications

**File:** `.github/workflows/alert-notifications.yml`

### Trigger
- Called by other workflows or manually triggered

### Supported Channels
- **Telegram** - Via bot token
- **Discord** - Via webhook
- **Slack** - Via webhook
- **GitHub Issues** - For critical alerts

### Setup Required
Add these secrets to your repository:
```
TELEGRAM_BOT_TOKEN    - Your Telegram bot token
TELEGRAM_CHAT_ID      - Your Telegram chat ID
DISCORD_WEBHOOK_URL   - Discord webhook URL
SLACK_WEBHOOK_URL     - Slack webhook URL
```

### Alert Types
- `signal` - Trading signal generated
- `drawdown` - Drawdown warning
- `performance` - Performance update
- `error` - System error

---

## 7. Code Quality & Testing

**File:** `.github/workflows/code-quality-testing.yml`

### Trigger
- On push to main or copilot/* branches
- On pull requests
- When Python or Pine Script files change

### What it does
1. **Linting** - flake8, black, isort
2. **Import Tests** - Verify all modules load
3. **Logic Tests** - Test EMA, RSI, ATR calculations
4. **Signal Tests** - Test signal generation
5. **Backtest Smoke Test** - Quick backtest validation
6. **Pine Script Validation** - Basic syntax check

---

## How to Use

### 1. Enable Workflows
Workflows are automatically enabled when pushed to the repository.

### 2. Manual Trigger
1. Go to Actions tab in GitHub
2. Select the workflow
3. Click "Run workflow"
4. Choose options and run

### 3. Set Up Notifications
1. Go to Settings > Secrets and variables > Actions
2. Add your notification credentials:
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`
   - `DISCORD_WEBHOOK_URL`
   - `SLACK_WEBHOOK_URL`

### 4. View Results
- **Actions tab** - See workflow runs and logs
- **Repository** - Check committed reports:
  - `predictions/` - Daily signals
  - `backtest-results/` - Weekly backtests
  - `reports/` - Monthly reports
  - `analysis/` - Multi-asset analysis
  - `scanner-results/` - Live scan results

---

## Workflow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    GitHub Actions Workflows                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │    Daily     │    │   Weekly     │    │   Monthly    │      │
│  │   Signals    │    │  Backtest    │    │   Report     │      │
│  │  (9 AM IST)  │    │  (Sunday)    │    │  (1st)       │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │               │
│         v                   v                   v               │
│  ┌─────────────────────────────────────────────────────┐       │
│  │                  Python Trading Engine               │       │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌────────┐ │       │
│  │  │Ultimate │  │   ML    │  │  Risk   │  │ Auto   │ │       │
│  │  │ Engine  │  │ Filter  │  │  Mgmt   │  │Optimize│ │       │
│  │  └─────────┘  └─────────┘  └─────────┘  └────────┘ │       │
│  └─────────────────────────────────────────────────────┘       │
│         │                   │                   │               │
│         v                   v                   v               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Market     │    │    Alert     │    │    Code      │      │
│  │   Scanner    │────│ Notifications│    │   Quality    │      │
│  │  (15 min)    │    │  (On-demand) │    │   (On PR)    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                              │                                   │
│                              v                                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Notification Channels                 │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │   │
│  │  │ Telegram │  │ Discord  │  │  Slack   │  │ GitHub │  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  │ Issues │  │   │
│  │                                            └────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Customization

### Change Schedule
Edit the `cron` expression in any workflow:
```yaml
schedule:
  - cron: '30 3 * * 1-5'  # Min Hour Day Month DayOfWeek
```

Cron format:
- Minute (0-59)
- Hour (0-23, UTC)
- Day of month (1-31)
- Month (1-12)
- Day of week (0-6, Sunday=0)

### Add New Symbols
Edit the symbol lists in each workflow's Python code.

### Modify Thresholds
Edit the configuration in Python scripts or create a `config.json`.

---

## Troubleshooting

### Workflow not running?
1. Check Actions tab for errors
2. Verify cron schedule is correct
3. Check if workflow is enabled

### Missing dependencies?
```yaml
- name: Install dependencies
  run: pip install numpy pandas yfinance
```

### Permission denied?
```yaml
permissions:
  contents: write
```

### Data not fetching?
The system uses sample data when yfinance is unavailable. For real data:
1. Ensure `yfinance` is installed
2. Check network connectivity
3. Verify symbol format (e.g., `RELIANCE.NS` for NSE)

---

## Best Practices

1. **Monitor Regularly** - Check workflow runs at least weekly
2. **Review Reports** - Read generated reports before trading
3. **Update Secrets** - Rotate API keys periodically
4. **Test Changes** - Use workflow_dispatch to test manually
5. **Keep History** - Don't delete old results for analysis

---

## Support

For issues or enhancements:
1. Check existing issues
2. Review workflow logs
3. Create a detailed issue with logs

---

*This documentation is part of the Ultimate Trading System*
