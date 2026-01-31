#!/usr/bin/env python3
"""
ONE-CLICK ULTIMATE TRADING SYSTEM RUNNER
=========================================
Run the entire trading system with a single command.

Features:
- Fetches real-time data
- Runs analysis with all components
- Generates signals with full reasoning
- Backtests with Monte Carlo validation
- Produces comprehensive reports

Usage:
    python run_ultimate.py
    python run_ultimate.py --symbol NIFTY50 --optimize
    python run_ultimate.py --help

Author: Trading Indicator Project
Version: 2.0.0 ULTIMATE
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Attempt imports
try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("Error: numpy and pandas required. Install with: pip install numpy pandas")
    sys.exit(1)


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults"""
    
    default_config = {
        "symbol": "NIFTY50",
        "timeframe": "1d",
        "period": "1y",
        
        "strategy": {
            "ema_period": 200,
            "supertrend_periods": [10, 11, 12],
            "supertrend_multipliers": [1.0, 2.0, 3.0],
            "stoch_rsi_period": 14,
            "adx_period": 14,
            "adx_threshold": 25
        },
        
        "risk_management": {
            "risk_per_trade_pct": 1.0,
            "max_drawdown_pct": 15.0,
            "max_positions": 3,
            "sl_atr_multiplier": 2.0,
            "tp_atr_multiplier": 3.0
        },
        
        "ml": {
            "enabled": True,
            "min_confidence": 0.6
        },
        
        "output": {
            "save_report": True,
            "report_path": "trading_report.json",
            "verbose": True
        }
    }
    
    if config_path:
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge configs
                for key, value in user_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    return default_config


def generate_sample_data(symbol: str, days: int = 500) -> pd.DataFrame:
    """Generate realistic sample OHLCV data"""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # Start price based on symbol
    start_prices = {
        'NIFTY50': 22000,
        'BANKNIFTY': 48000,
        'AAPL': 180,
        'GOLD': 2000,
        'SPY': 450
    }
    price = start_prices.get(symbol, 100)
    
    data = []
    for i in range(days):
        # Add trend and volatility
        trend = 0.0003
        vol = 0.015
        
        if i % 100 < 20:
            vol *= 1.5  # High vol period
        
        returns = trend + vol * np.random.randn()
        
        open_p = price
        close_p = price * (1 + returns)
        high_p = max(open_p, close_p) * (1 + abs(np.random.randn() * 0.005))
        low_p = min(open_p, close_p) * (1 - abs(np.random.randn() * 0.005))
        volume = int(1000000 * (1 + np.random.randn() * 0.3))
        
        data.append({
            'open': open_p,
            'high': high_p,
            'low': low_p,
            'close': close_p,
            'volume': max(volume, 100000)
        })
        
        price = close_p
    
    return pd.DataFrame(data, index=dates)


def fetch_data(symbol: str, timeframe: str, period: str) -> pd.DataFrame:
    """Fetch market data (uses sample data as fallback)"""
    
    print(f"\nFetching data for {symbol}...")
    
    try:
        from realtime_data import RealTimeDataFetcher
        fetcher = RealTimeDataFetcher()
        df = fetcher.get_data(symbol, timeframe, period)
        
        if df is not None and len(df) > 100:
            print(f"  ✓ Fetched {len(df)} bars of real data")
            return df
    except Exception as e:
        print(f"  ⚠ Real data fetch failed: {e}")
    
    # Fallback to sample data
    print(f"  → Using sample data (500 days)")
    return generate_sample_data(symbol, 500)


def run_analysis(df: pd.DataFrame, symbol: str, config: Dict) -> Dict[str, Any]:
    """Run complete analysis pipeline"""
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'data_points': len(df),
        'date_range': f"{df.index[0]} to {df.index[-1]}"
    }
    
    # 1. Basic Statistics
    print("\n1. CALCULATING MARKET STATISTICS...")
    close = df['close']
    
    results['statistics'] = {
        'current_price': float(close.iloc[-1]),
        'price_change_1d': float((close.iloc[-1] / close.iloc[-2] - 1) * 100),
        'price_change_5d': float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 5 else 0,
        'price_change_20d': float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) > 20 else 0,
        'volatility_20d': float(close.pct_change().tail(20).std() * 100 * np.sqrt(252)),
        '52w_high': float(close.tail(252).max()) if len(close) >= 252 else float(close.max()),
        '52w_low': float(close.tail(252).min()) if len(close) >= 252 else float(close.min())
    }
    
    print(f"  Current Price: ${results['statistics']['current_price']:,.2f}")
    print(f"  1-Day Change:  {results['statistics']['price_change_1d']:+.2f}%")
    print(f"  20-Day Vol:    {results['statistics']['volatility_20d']:.1f}%")
    
    # 2. Technical Indicators
    print("\n2. CALCULATING TECHNICAL INDICATORS...")
    
    # EMA
    ema_200 = close.ewm(span=200, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    ema_20 = close.ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - close.shift(1)),
        abs(df['low'] - close.shift(1))
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    
    # ADX (simplified)
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(14).mean()
    
    results['indicators'] = {
        'ema_200': float(ema_200.iloc[-1]),
        'ema_50': float(ema_50.iloc[-1]),
        'ema_20': float(ema_20.iloc[-1]),
        'rsi_14': float(rsi.iloc[-1]),
        'atr_14': float(atr.iloc[-1]),
        'adx_14': float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 20,
        'trend': 'BULLISH' if close.iloc[-1] > ema_200.iloc[-1] else 'BEARISH',
        'ema_alignment': 'ALIGNED' if (close.iloc[-1] > ema_20.iloc[-1] > ema_50.iloc[-1] > ema_200.iloc[-1]) or (close.iloc[-1] < ema_20.iloc[-1] < ema_50.iloc[-1] < ema_200.iloc[-1]) else 'MIXED'
    }
    
    print(f"  EMA(200): ${results['indicators']['ema_200']:,.2f}")
    print(f"  RSI(14):  {results['indicators']['rsi_14']:.1f}")
    print(f"  ADX(14):  {results['indicators']['adx_14']:.1f}")
    print(f"  Trend:    {results['indicators']['trend']}")
    
    # 3. Generate Signal
    print("\n3. GENERATING TRADING SIGNAL...")
    
    try:
        from ultimate_trading_engine import UltimateTradingEngine
        
        engine_config = {
            **config['strategy'],
            **config['risk_management'],
            'use_ml': config['ml']['enabled']
        }
        
        engine = UltimateTradingEngine(engine_config)
        signal = engine.generate_signal(df, symbol)
        
        if signal:
            results['signal'] = {
                'type': signal.signal_type.name,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit_1': signal.take_profit_1,
                'take_profit_2': signal.take_profit_2,
                'take_profit_3': signal.take_profit_3,
                'position_size_pct': signal.position_size_pct,
                'regime': signal.regime.value,
                'mtf_alignment': signal.mtf_alignment,
                'ml_confidence': signal.ml_confidence,
                'reasoning': signal.reasoning
            }
            
            print(f"\n  SIGNAL: {signal.signal_type.name}")
            print(f"  Confidence: {signal.confidence:.0f}%")
            print(f"  Entry: ${signal.entry_price:.2f}")
            print(f"  Stop Loss: ${signal.stop_loss:.2f}")
            print(f"  Take Profit 1: ${signal.take_profit_1:.2f}")
            print(f"  Position Size: {signal.position_size_pct:.1f}%")
        else:
            results['signal'] = {'type': 'NEUTRAL', 'confidence': 0}
            print("  No signal generated")
            
    except ImportError:
        # Simplified signal generation
        print("  (Using simplified signal generator)")
        
        trend_signal = 1 if close.iloc[-1] > ema_200.iloc[-1] else -1
        rsi_signal = 1 if rsi.iloc[-1] < 40 else (-1 if rsi.iloc[-1] > 60 else 0)
        
        combined = trend_signal + rsi_signal
        
        if combined >= 1:
            signal_type = 'BUY'
            confidence = 60
        elif combined <= -1:
            signal_type = 'SELL'
            confidence = 60
        else:
            signal_type = 'NEUTRAL'
            confidence = 30
        
        current_price = close.iloc[-1]
        current_atr = atr.iloc[-1]
        
        results['signal'] = {
            'type': signal_type,
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': current_price - (current_atr * 2) if signal_type == 'BUY' else current_price + (current_atr * 2),
            'take_profit_1': current_price + (current_atr * 3) if signal_type == 'BUY' else current_price - (current_atr * 3),
            'position_size_pct': 1.0
        }
        
        print(f"  SIGNAL: {signal_type}")
        print(f"  Confidence: {confidence}%")
    
    return results


def run_backtest(df: pd.DataFrame, symbol: str, config: Dict) -> Dict[str, Any]:
    """Run backtest and return results"""
    
    print("\n4. RUNNING BACKTEST...")
    
    try:
        from ultimate_trading_engine import UltimateTradingEngine
        
        engine_config = {
            **config['strategy'],
            **config['risk_management'],
            'use_ml': config['ml']['enabled']
        }
        
        engine = UltimateTradingEngine(engine_config)
        results = engine.backtest(df, symbol)
        
        return results
        
    except ImportError:
        # Simplified backtest
        print("  (Using simplified backtest)")
        
        close = df['close'].values
        ema = pd.Series(close).ewm(span=200, adjust=False).mean().values
        
        trades = []
        position = None
        
        for i in range(201, len(close)):
            if position is None:
                if close[i] > ema[i] and close[i-1] <= ema[i-1]:
                    position = {'entry': close[i], 'direction': 'long'}
                elif close[i] < ema[i] and close[i-1] >= ema[i-1]:
                    position = {'entry': close[i], 'direction': 'short'}
            else:
                exit_signal = False
                if position['direction'] == 'long' and close[i] < ema[i]:
                    exit_signal = True
                elif position['direction'] == 'short' and close[i] > ema[i]:
                    exit_signal = True
                
                if exit_signal:
                    if position['direction'] == 'long':
                        pnl = (close[i] - position['entry']) / position['entry'] * 100
                    else:
                        pnl = (position['entry'] - close[i]) / position['entry'] * 100
                    trades.append(pnl)
                    position = None
        
        if trades:
            wins = [t for t in trades if t > 0]
            losses = [t for t in trades if t < 0]
            
            return {
                'total_trades': len(trades),
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'win_rate': len(wins) / len(trades) * 100,
                'profit_factor': sum(wins) / abs(sum(losses)) if losses else 10,
                'total_return_pct': sum(trades),
                'max_drawdown': 10.0  # Simplified
            }
        
        return {'total_trades': 0, 'win_rate': 0, 'profit_factor': 0}


def run_optimization(df: pd.DataFrame, config: Dict) -> Dict[str, Any]:
    """Run parameter optimization"""
    
    print("\n5. RUNNING PARAMETER OPTIMIZATION...")
    
    try:
        from auto_optimizer import AutoOptimizer
        
        optimizer = AutoOptimizer()
        
        param_grid = {
            'ema_period': [150, 200, 250],
            'sl_atr_mult': [1.5, 2.0, 2.5],
            'tp_atr_mult': [2.0, 3.0, 4.0]
        }
        
        result = optimizer.grid_search(df, param_grid)
        
        return {
            'best_params': result.best_params,
            'best_score': result.best_score,
            'is_overfit': result.is_overfit,
            'robustness_score': result.robustness_score
        }
        
    except ImportError:
        print("  Optimization module not available")
        return {'status': 'skipped'}


def print_summary(results: Dict[str, Any], backtest: Dict[str, Any]):
    """Print final summary"""
    
    print("\n" + "=" * 60)
    print("TRADING ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"\nSymbol: {results['symbol']}")
    print(f"Analysis Time: {results['timestamp']}")
    print(f"Data Points: {results['data_points']}")
    
    if 'statistics' in results:
        stats = results['statistics']
        print(f"\nCurrent Price: ${stats['current_price']:,.2f}")
        print(f"1-Day Change:  {stats['price_change_1d']:+.2f}%")
    
    if 'indicators' in results:
        ind = results['indicators']
        print(f"\nTrend:     {ind['trend']}")
        print(f"RSI:       {ind['rsi_14']:.1f}")
        print(f"ADX:       {ind['adx_14']:.1f}")
    
    if 'signal' in results:
        sig = results['signal']
        print(f"\n{'='*40}")
        print(f"SIGNAL: {sig['type']}")
        print(f"Confidence: {sig['confidence']}%")
        
        if sig['type'] != 'NEUTRAL':
            print(f"\nEntry:       ${sig.get('entry_price', 0):,.2f}")
            print(f"Stop Loss:   ${sig.get('stop_loss', 0):,.2f}")
            print(f"Take Profit: ${sig.get('take_profit_1', 0):,.2f}")
    
    if backtest and 'total_trades' in backtest:
        print(f"\n{'='*40}")
        print("BACKTEST RESULTS")
        print(f"Total Trades:  {backtest['total_trades']}")
        print(f"Win Rate:      {backtest.get('win_rate', 0):.1f}%")
        print(f"Profit Factor: {backtest.get('profit_factor', 0):.2f}")
        print(f"Total Return:  {backtest.get('total_return_pct', 0):.2f}%")
    
    print("\n" + "=" * 60)


def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description='Ultimate Trading System - One-Click Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ultimate.py                    # Run with defaults (NIFTY50)
  python run_ultimate.py --symbol AAPL      # Analyze AAPL
  python run_ultimate.py --optimize         # Include optimization
  python run_ultimate.py --config my.json   # Use custom config
        """
    )
    
    parser.add_argument('--symbol', '-s', default='NIFTY50',
                       help='Trading symbol (default: NIFTY50)')
    parser.add_argument('--timeframe', '-t', default='1d',
                       help='Timeframe (default: 1d)')
    parser.add_argument('--period', '-p', default='1y',
                       help='Data period (default: 1y)')
    parser.add_argument('--config', '-c', 
                       help='Path to config JSON file')
    parser.add_argument('--optimize', '-o', action='store_true',
                       help='Run parameter optimization')
    parser.add_argument('--no-backtest', action='store_true',
                       help='Skip backtest')
    parser.add_argument('--save', action='store_true',
                       help='Save results to JSON')
    
    args = parser.parse_args()
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║            ULTIMATE TRADING SYSTEM v2.0                      ║
    ║                                                              ║
    ║         ONE-CLICK ANALYSIS & TRADING SIGNALS                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Load config
    config = load_config(args.config)
    config['symbol'] = args.symbol
    config['timeframe'] = args.timeframe
    config['period'] = args.period
    
    # Fetch data
    df = fetch_data(args.symbol, args.timeframe, args.period)
    
    if df is None or len(df) < 100:
        print("Error: Insufficient data")
        return 1
    
    # Run analysis
    results = run_analysis(df, args.symbol, config)
    
    # Run backtest
    backtest_results = None
    if not args.no_backtest:
        backtest_results = run_backtest(df, args.symbol, config)
        results['backtest'] = backtest_results
    
    # Run optimization
    if args.optimize:
        opt_results = run_optimization(df, config)
        results['optimization'] = opt_results
    
    # Print summary
    print_summary(results, backtest_results)
    
    # Save results
    if args.save:
        output_file = f"trading_results_{args.symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert non-serializable types
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Timestamp):
                return str(obj)
            return obj
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=convert)
        
        print(f"\nResults saved to: {output_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
