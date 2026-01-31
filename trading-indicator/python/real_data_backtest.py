"""
Real Data Backtesting Script for UMCI
Fetches actual market data from Yahoo Finance and runs backtest
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import BacktestConfig, run_backtest, print_results, BacktestResults

# Symbol mappings for different markets
SYMBOL_MAP = {
    # Indian Indices
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    # Indian Stocks
    "TCS": "TCS.NS",
    "RELIANCE": "RELIANCE.NS",
    "HDFC": "HDFCBANK.NS",
    "INFY": "INFY.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    # US Indices
    "SPY": "SPY",
    "QQQ": "QQQ",
    "SP500": "^GSPC",
    "NASDAQ": "^IXIC",
    "DOW": "^DJI",
    # US Stocks
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "GOOGL": "GOOGL",
    "AMZN": "AMZN",
    "TSLA": "TSLA",
    "NVDA": "NVDA",
    "META": "META",
    # Commodities
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "CRUDE": "CL=F",
    # Forex
    "EURUSD": "EURUSD=X",
    "USDINR": "INR=X",
    "GBPUSD": "GBPUSD=X",
    # Crypto
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
}


def fetch_real_data(symbol: str, start_date: str, end_date: str, interval: str = "1d") -> list:
    """
    Fetch real market data from Yahoo Finance.
    
    Args:
        symbol: Symbol name (e.g., NIFTY50, SPY, AAPL)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        interval: Data interval (1d, 1h, 15m, 5m)
    
    Returns:
        List of OHLCV dictionaries
    """
    # Map symbol to Yahoo Finance ticker
    yf_symbol = SYMBOL_MAP.get(symbol.upper(), symbol)
    
    print(f"Fetching data for {symbol} ({yf_symbol})...")
    
    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            print(f"No data found for {symbol}")
            return []
        
        data = []
        for idx, row in df.iterrows():
            # Handle timezone-aware timestamps
            if hasattr(idx, 'tz_localize'):
                date_str = idx.strftime("%Y-%m-%d")
            else:
                date_str = str(idx)[:10]
            
            data.append({
                "date": date_str,
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0
            })
        
        print(f"Loaded {len(data)} bars of real data for {symbol}")
        return data
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return []


def run_real_backtest(symbol: str, start_date: str = "2021-01-01", end_date: str = None, 
                      capital: float = 100000, interval: str = "1d") -> BacktestResults:
    """Run backtest on real market data."""
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Fetch real data
    data = fetch_real_data(symbol, start_date, end_date, interval)
    
    if not data or len(data) < 50:
        print(f"Insufficient data for {symbol} (need at least 50 bars)")
        return None
    
    # Configure backtest
    config = BacktestConfig(initial_capital=capital)
    
    # Run backtest
    print(f"\nRunning backtest on {symbol}...")
    results = run_backtest(data, config)
    
    return results


def run_multi_asset_backtest():
    """Run backtest on multiple assets and compile results."""
    
    # Assets to test
    test_assets = [
        # Indian Indices
        ("NIFTY50", "^NSEI", "NSE NIFTY 50 Index"),
        ("BANKNIFTY", "^NSEBANK", "NSE Bank NIFTY Index"),
        # Indian Stocks
        ("TCS", "TCS.NS", "Tata Consultancy Services"),
        ("RELIANCE", "RELIANCE.NS", "Reliance Industries"),
        ("INFY", "INFY.NS", "Infosys"),
        ("HDFC", "HDFCBANK.NS", "HDFC Bank"),
        # US Indices
        ("SPY", "SPY", "S&P 500 ETF"),
        ("QQQ", "QQQ", "NASDAQ 100 ETF"),
        # US Stocks
        ("AAPL", "AAPL", "Apple Inc"),
        ("MSFT", "MSFT", "Microsoft"),
        ("GOOGL", "GOOGL", "Alphabet (Google)"),
        ("TSLA", "TSLA", "Tesla"),
        # Commodities
        ("GOLD", "GC=F", "Gold Futures"),
        # Forex
        ("EURUSD", "EURUSD=X", "EUR/USD"),
    ]
    
    results_summary = []
    
    print("=" * 80)
    print("UMCI REAL DATA BACKTEST - MULTI-ASSET ANALYSIS")
    print("=" * 80)
    print(f"Test Period: 2021-01-01 to {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Initial Capital: 100,000")
    print("=" * 80)
    
    for symbol, yf_ticker, description in test_assets:
        print(f"\n{'='*60}")
        print(f"Testing: {description} ({symbol})")
        print(f"{'='*60}")
        
        try:
            results = run_real_backtest(symbol)
            
            if results and results.total_trades > 0:
                results_summary.append({
                    "symbol": symbol,
                    "description": description,
                    "total_trades": results.total_trades,
                    "win_rate": round(results.win_rate, 2),
                    "profit_factor": round(results.profit_factor, 2),
                    "total_return": round(results.total_return_percent, 2),
                    "max_drawdown": round(results.max_drawdown_percent, 2),
                    "sharpe_ratio": round(results.sharpe_ratio, 2),
                    "status": "SUCCESS"
                })
                
                print(f"\nResults for {symbol}:")
                print(f"  Total Trades: {results.total_trades}")
                print(f"  Win Rate: {results.win_rate:.2f}%")
                print(f"  Profit Factor: {results.profit_factor:.2f}")
                print(f"  Total Return: {results.total_return_percent:.2f}%")
                print(f"  Max Drawdown: {results.max_drawdown_percent:.2f}%")
            else:
                results_summary.append({
                    "symbol": symbol,
                    "description": description,
                    "status": "NO_TRADES" if results else "NO_DATA"
                })
                print(f"No valid results for {symbol}")
                
        except Exception as e:
            results_summary.append({
                "symbol": symbol,
                "description": description,
                "status": f"ERROR: {str(e)[:50]}"
            })
            print(f"Error testing {symbol}: {e}")
    
    # Print summary table
    print("\n" + "=" * 100)
    print("SUMMARY OF RESULTS")
    print("=" * 100)
    print(f"{'Asset':<15} {'Description':<25} {'Trades':>8} {'Win Rate':>10} {'PF':>8} {'Return':>10} {'MaxDD':>10}")
    print("-" * 100)
    
    for r in results_summary:
        if r.get("status") == "SUCCESS":
            print(f"{r['symbol']:<15} {r['description'][:24]:<25} {r['total_trades']:>8} {r['win_rate']:>9.1f}% {r['profit_factor']:>7.2f} {r['total_return']:>9.1f}% {r['max_drawdown']:>9.1f}%")
        else:
            print(f"{r['symbol']:<15} {r['description'][:24]:<25} {r.get('status', 'UNKNOWN'):>50}")
    
    print("=" * 100)
    
    # Calculate averages for successful backtests
    successful = [r for r in results_summary if r.get("status") == "SUCCESS"]
    if successful:
        avg_win_rate = sum(r["win_rate"] for r in successful) / len(successful)
        avg_pf = sum(r["profit_factor"] for r in successful) / len(successful)
        avg_return = sum(r["total_return"] for r in successful) / len(successful)
        avg_dd = sum(r["max_drawdown"] for r in successful) / len(successful)
        
        print(f"\nAVERAGE ACROSS {len(successful)} ASSETS:")
        print(f"  Average Win Rate: {avg_win_rate:.2f}%")
        print(f"  Average Profit Factor: {avg_pf:.2f}")
        print(f"  Average Total Return: {avg_return:.2f}%")
        print(f"  Average Max Drawdown: {avg_dd:.2f}%")
    
    return results_summary


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='UMCI Real Data Backtesting')
    parser.add_argument('--symbol', type=str, default=None,
                       help='Single symbol to test (e.g., NIFTY50, SPY, AAPL)')
    parser.add_argument('--start', type=str, default='2021-01-01',
                       help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, default=None,
                       help='End date YYYY-MM-DD (default: today)')
    parser.add_argument('--capital', type=float, default=100000,
                       help='Initial capital')
    parser.add_argument('--multi', action='store_true',
                       help='Run multi-asset backtest')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    if args.multi:
        results = run_multi_asset_backtest()
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
    elif args.symbol:
        results = run_real_backtest(args.symbol, args.start, args.end, args.capital)
        if results:
            print_results(results, args.symbol)
            if args.output:
                # Export detailed results
                output_data = {
                    "symbol": args.symbol,
                    "period": f"{args.start} to {args.end or 'today'}",
                    "initial_capital": args.capital,
                    "results": {
                        "total_trades": results.total_trades,
                        "winning_trades": results.winning_trades,
                        "losing_trades": results.losing_trades,
                        "win_rate": round(results.win_rate, 2),
                        "profit_factor": round(results.profit_factor, 2),
                        "total_pnl": round(results.total_pnl, 2),
                        "total_return_percent": round(results.total_return_percent, 2),
                        "max_drawdown_percent": round(results.max_drawdown_percent, 2),
                        "sharpe_ratio": round(results.sharpe_ratio, 2)
                    }
                }
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"\nResults saved to {args.output}")
    else:
        print("Please specify --symbol or --multi flag")
        print("\nAvailable symbols:")
        for category, symbols in [
            ("Indian Indices", ["NIFTY50", "BANKNIFTY"]),
            ("Indian Stocks", ["TCS", "RELIANCE", "HDFC", "INFY"]),
            ("US Indices/ETFs", ["SPY", "QQQ"]),
            ("US Stocks", ["AAPL", "MSFT", "GOOGL", "TSLA"]),
            ("Commodities", ["GOLD", "SILVER", "CRUDE"]),
            ("Forex", ["EURUSD", "USDINR"])
        ]:
            print(f"  {category}: {', '.join(symbols)}")


if __name__ == "__main__":
    main()
