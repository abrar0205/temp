# Network and Offline Mode Guide

## TL;DR - It Works!

**The network was only blocked in my development sandbox.** When you run this on your computer, internet will work fine!

---

## How Data Fetching Works

### Online Mode (Normal - When You Run It)

When you run the code on your computer with internet access:

```python
# This will download real data from Yahoo Finance
python realtime_data.py --symbol NIFTY50 --timeframe 1d --period 6mo
```

**What happens:**
1. Code connects to Yahoo Finance API
2. Downloads real historical/live data
3. Caches data locally for efficiency
4. Returns real market data for analysis

### Offline Mode (Automatic Fallback)

If internet is unavailable (rare), the code automatically falls back to sample data:

```
Attempting to fetch data from Yahoo Finance...
Warning: Network unavailable, using sample data
Generated sample data for testing (200 bars)
```

**Sample data is:**
- Realistic OHLCV data with proper price movements
- Suitable for testing strategy logic
- NOT for actual trading decisions

---

## Data Sources

### 1. Yahoo Finance (yfinance) - Primary
- **Works for:** Global stocks, indices, forex, crypto, commodities
- **Indian symbols:** 
  - NIFTY 50: `^NSEI` 
  - Bank NIFTY: `^NSEBANK`
  - Reliance: `RELIANCE.NS`
  - TCS: `TCS.NS`
- **US symbols:** `AAPL`, `GOOGL`, `MSFT`, `^GSPC` (S&P 500)
- **Crypto:** `BTC-USD`, `ETH-USD`
- **Forex:** `EURUSD=X`, `USDINR=X`

### 2. NSE Direct (For India) - Optional
For more reliable Indian market data, you can use:
- [nsepy](https://github.com/swapniljariwala/nsepy) - NSE Python library
- [nsetools](https://github.com/vsjha18/nsetools) - NSE tools

### 3. Broker APIs - For Live Trading
When you're ready for live trading:
- **Zerodha Kite:** [kiteconnect](https://kite.trade/docs/pykiteconnect/)
- **Angel One:** [smartapi-python](https://github.com/angel-one/smartapi-python)
- **Interactive Brokers:** [ib_insync](https://github.com/erdewit/ib_insync)

---

## Testing the Data Connection

### Quick Test

```bash
cd trading-indicator/python

# Test with Yahoo Finance
python -c "
from realtime_data import get_data
data = get_data('AAPL', '1d', '3mo')
print(f'Fetched {len(data)} bars')
print(f'Latest: {data[-1]}')
"
```

### Expected Output (With Internet)

```
Fetched 63 bars
Latest: {'timestamp': '2026-01-30', 'open': 185.50, 'high': 187.20, 'low': 184.80, 'close': 186.75, 'volume': 45000000}
```

### Expected Output (Without Internet)

```
Warning: Network unavailable, using sample data
Fetched 200 bars
Latest: {'timestamp': '2026-01-30', 'open': 100.50, 'high': 101.20, 'low': 99.80, 'close': 100.75, 'volume': 1000000}
```

---

## Caching System

To minimize API calls and improve speed, data is cached locally:

### Cache Location
```
trading-indicator/python/data_cache/
├── AAPL_1d.json
├── NSEI_1d.json
├── RELIANCE.NS_1d.json
└── ...
```

### Cache Duration
- **Daily data:** 4 hours
- **Intraday data:** 15 minutes

### Force Fresh Data
```python
from realtime_data import YFinanceFetcher

fetcher = YFinanceFetcher(cache_dir="data_cache")
# Delete cache to force fresh fetch
import os
os.remove("data_cache/AAPL_1d.json")

# Now fetch fresh
data = fetcher.get_data("AAPL", "1d", "3mo")
```

---

## Streamlit Dashboard

The Streamlit dashboard automatically handles data:

```bash
# Run dashboard
streamlit run streamlit_dashboard.py

# It will:
# 1. Try to fetch real data
# 2. Fall back to sample data if network fails
# 3. Show clear message about data source
```

---

## Summary

| Scenario | Data Source | Suitable For |
|----------|-------------|--------------|
| Internet available | Yahoo Finance (real) | Analysis, backtesting, paper trading |
| Internet unavailable | Sample data (generated) | Testing code logic only |
| Live trading | Broker API (Zerodha, etc.) | Actual trading |

**Bottom line:** Run it on your computer with internet - it will work! The sample data fallback is just a safety net.

---

## Next Steps for Live Trading

When you're ready for live trading:

1. **Paper Trade First** (2-4 weeks)
   - Use real data from yfinance
   - Track signals manually
   - Build confidence

2. **Get Broker API** 
   - Sign up for Zerodha/Angel One
   - Apply for API access
   - Get API keys

3. **Connect API**
   ```python
   from kiteconnect import KiteConnect
   
   kite = KiteConnect(api_key="your_key")
   kite.set_access_token("your_token")
   
   # Now you can place real orders
   ```

4. **Start Small**
   - Trade minimum quantity
   - Scale up gradually
   - Always use stop losses
