#!/usr/bin/env python3
"""
PERFORMANCE BENCHMARK MODULE
============================
Compare different trading strategies with statistical tests:

1. Backtest multiple strategies on same data
2. Calculate comprehensive metrics
3. Statistical significance tests
4. Risk-adjusted comparison
5. Generate benchmark report
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class StrategyMetrics:
    """Comprehensive strategy metrics."""
    name: str
    total_trades: int
    win_rate: float
    profit_factor: float
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    consecutive_wins: int
    consecutive_losses: int
    recovery_factor: float
    expectancy: float


@dataclass
class BenchmarkResult:
    """Result of benchmark comparison."""
    strategies: Dict[str, StrategyMetrics]
    best_by_sharpe: str
    best_by_profit_factor: str
    best_by_win_rate: str
    best_by_return: str
    statistical_tests: Dict[str, Dict]


class PerformanceBenchmark:
    """
    Benchmark and compare trading strategies.
    
    Usage:
        benchmark = PerformanceBenchmark()
        benchmark.add_strategy('RSI', rsi_signals)
        benchmark.add_strategy('MACD', macd_signals)
        result = benchmark.run_benchmark(opens, highs, lows, closes)
    """
    
    def __init__(self, risk_free_rate: float = 0.05,
                 transaction_cost: float = 0.001):
        """
        Initialize benchmark.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
            transaction_cost: Round-trip transaction cost (default 0.1%)
        """
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.strategies: Dict[str, np.ndarray] = {}
    
    def add_strategy(self, name: str, signals: np.ndarray):
        """
        Add a strategy to benchmark.
        
        Args:
            name: Strategy name
            signals: Array of signals (1=buy, -1=sell, 0=hold)
        """
        self.strategies[name] = signals
    
    def add_strategy_function(self, name: str, 
                              signal_func: Callable[[np.ndarray, np.ndarray, 
                                                     np.ndarray, np.ndarray], np.ndarray]):
        """
        Add strategy as a function.
        
        Args:
            name: Strategy name
            signal_func: Function that takes (opens, highs, lows, closes) and returns signals
        """
        self.strategy_functions = getattr(self, 'strategy_functions', {})
        self.strategy_functions[name] = signal_func
    
    def _calculate_returns(self, closes: np.ndarray, 
                           signals: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Calculate returns and trades from signals.
        
        Returns:
            Tuple of (daily returns array, list of trades)
        """
        returns = np.zeros(len(closes))
        trades = []
        
        position = 0  # 0=flat, 1=long
        entry_price = 0
        entry_idx = 0
        
        for i in range(1, len(closes)):
            # Check for signal
            if signals[i] == 1 and position == 0:  # Buy signal, not in position
                position = 1
                entry_price = closes[i]
                entry_idx = i
            
            elif signals[i] == -1 and position == 1:  # Sell signal, in position
                # Close position
                exit_price = closes[i]
                trade_return = (exit_price - entry_price) / entry_price
                trade_return -= self.transaction_cost  # Apply costs
                
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_return,
                    'duration': i - entry_idx
                })
                
                position = 0
                entry_price = 0
            
            # Calculate daily return if in position
            if position == 1:
                returns[i] = (closes[i] - closes[i-1]) / closes[i-1]
        
        return returns, trades
    
    def _calculate_metrics(self, name: str, returns: np.ndarray, 
                          trades: List[Dict], closes: np.ndarray) -> StrategyMetrics:
        """Calculate comprehensive metrics for a strategy."""
        
        # Basic trade stats
        total_trades = len(trades)
        
        if total_trades == 0:
            return StrategyMetrics(
                name=name, total_trades=0, win_rate=0, profit_factor=0,
                total_return=0, cagr=0, max_drawdown=0, sharpe_ratio=0,
                sortino_ratio=0, calmar_ratio=0, avg_win=0, avg_loss=0,
                largest_win=0, largest_loss=0, avg_trade_duration=0,
                consecutive_wins=0, consecutive_losses=0, recovery_factor=0,
                expectancy=0
            )
        
        trade_returns = [t['return'] for t in trades]
        winning_trades = [t for t in trade_returns if t > 0]
        losing_trades = [t for t in trade_returns if t < 0]
        
        # Win rate
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(winning_trades) if winning_trades else 0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Returns
        equity_curve = (1 + returns).cumprod()
        total_return = equity_curve[-1] - 1 if len(equity_curve) > 0 else 0
        
        # CAGR
        n_years = len(closes) / 252
        cagr = (equity_curve[-1] ** (1/n_years) - 1) if n_years > 0 and equity_curve[-1] > 0 else 0
        
        # Max drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (equity_curve - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        # Sharpe ratio (annualized)
        daily_rf = self.risk_free_rate / 252
        excess_returns = returns - daily_rf
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        
        # Sortino ratio (only downside volatility)
        downside_returns = returns[returns < daily_rf]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0.0001
        sortino_ratio = np.sqrt(252) * np.mean(excess_returns) / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = cagr / max_drawdown if max_drawdown > 0 else 0
        
        # Average win/loss
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        # Largest win/loss
        largest_win = max(winning_trades) if winning_trades else 0
        largest_loss = min(losing_trades) if losing_trades else 0
        
        # Average trade duration
        avg_trade_duration = np.mean([t['duration'] for t in trades]) if trades else 0
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        current_streak = 0
        current_type = None
        
        for tr in trade_returns:
            if tr > 0:
                if current_type == 'win':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = 'win'
                consecutive_wins = max(consecutive_wins, current_streak)
            else:
                if current_type == 'loss':
                    current_streak += 1
                else:
                    current_streak = 1
                    current_type = 'loss'
                consecutive_losses = max(consecutive_losses, current_streak)
        
        # Recovery factor
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Expectancy (average trade return)
        expectancy = np.mean(trade_returns) if trade_returns else 0
        
        return StrategyMetrics(
            name=name,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_return=total_return,
            cagr=cagr,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_trade_duration,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            recovery_factor=recovery_factor,
            expectancy=expectancy
        )
    
    def _t_test(self, returns1: np.ndarray, returns2: np.ndarray) -> Dict:
        """Perform two-sample t-test."""
        n1, n2 = len(returns1), len(returns2)
        mean1, mean2 = np.mean(returns1), np.mean(returns2)
        var1, var2 = np.var(returns1, ddof=1), np.var(returns2, ddof=1)
        
        # Pooled standard error
        se = np.sqrt(var1/n1 + var2/n2)
        
        if se == 0:
            return {'t_statistic': 0, 'p_value': 1.0, 'significant': False}
        
        t_stat = (mean1 - mean2) / se
        
        # Approximate p-value using normal distribution (for large samples)
        # For more accurate p-value, use scipy.stats.t
        from math import erf, sqrt
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / sqrt(2))))
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def run_benchmark(self, opens: np.ndarray, highs: np.ndarray,
                      lows: np.ndarray, closes: np.ndarray) -> BenchmarkResult:
        """
        Run benchmark on all strategies.
        
        Returns:
            BenchmarkResult with all metrics and comparisons
        """
        # Generate signals from functions if any
        if hasattr(self, 'strategy_functions'):
            for name, func in self.strategy_functions.items():
                self.strategies[name] = func(opens, highs, lows, closes)
        
        # Calculate metrics for each strategy
        all_metrics: Dict[str, StrategyMetrics] = {}
        all_returns: Dict[str, np.ndarray] = {}
        
        for name, signals in self.strategies.items():
            returns, trades = self._calculate_returns(closes, signals)
            metrics = self._calculate_metrics(name, returns, trades, closes)
            all_metrics[name] = metrics
            all_returns[name] = returns
        
        # Find best strategies
        best_sharpe = max(all_metrics.keys(), 
                         key=lambda k: all_metrics[k].sharpe_ratio)
        best_pf = max(all_metrics.keys(), 
                     key=lambda k: all_metrics[k].profit_factor)
        best_wr = max(all_metrics.keys(), 
                     key=lambda k: all_metrics[k].win_rate)
        best_return = max(all_metrics.keys(), 
                         key=lambda k: all_metrics[k].total_return)
        
        # Statistical tests (compare each to best)
        stat_tests = {}
        best_returns = all_returns.get(best_sharpe, np.zeros(len(closes)))
        
        for name, returns in all_returns.items():
            if name != best_sharpe:
                stat_tests[f"{name}_vs_{best_sharpe}"] = self._t_test(returns, best_returns)
        
        return BenchmarkResult(
            strategies=all_metrics,
            best_by_sharpe=best_sharpe,
            best_by_profit_factor=best_pf,
            best_by_win_rate=best_wr,
            best_by_return=best_return,
            statistical_tests=stat_tests
        )
    
    def print_benchmark_report(self, result: BenchmarkResult):
        """Print formatted benchmark report."""
        print("\n" + "=" * 80)
        print("STRATEGY BENCHMARK REPORT")
        print("=" * 80)
        
        # Header
        print(f"\n{'Strategy':<20} {'Trades':>8} {'Win Rate':>10} {'PF':>8} "
              f"{'Return':>10} {'MaxDD':>10} {'Sharpe':>8}")
        print("-" * 80)
        
        # Metrics for each strategy
        for name, metrics in result.strategies.items():
            marker = ""
            if name == result.best_by_sharpe:
                marker = " *"
            
            print(f"{name + marker:<20} {metrics.total_trades:>8} "
                  f"{metrics.win_rate:>9.1%} {metrics.profit_factor:>8.2f} "
                  f"{metrics.total_return:>9.1%} {metrics.max_drawdown:>9.1%} "
                  f"{metrics.sharpe_ratio:>8.2f}")
        
        # Best strategies
        print("\n" + "-" * 80)
        print("BEST STRATEGIES")
        print("-" * 80)
        print(f"Best by Sharpe Ratio:  {result.best_by_sharpe}")
        print(f"Best by Profit Factor: {result.best_by_profit_factor}")
        print(f"Best by Win Rate:      {result.best_by_win_rate}")
        print(f"Best by Total Return:  {result.best_by_return}")
        
        # Statistical tests
        if result.statistical_tests:
            print("\n" + "-" * 80)
            print("STATISTICAL SIGNIFICANCE TESTS")
            print("-" * 80)
            
            for comparison, test in result.statistical_tests.items():
                sig = "YES" if test['significant'] else "NO"
                print(f"{comparison}: t={test['t_statistic']:.3f}, "
                      f"p={test['p_value']:.4f}, Significant: {sig}")
        
        # Detailed metrics for best strategy
        best = result.strategies[result.best_by_sharpe]
        print("\n" + "-" * 80)
        print(f"DETAILED METRICS: {result.best_by_sharpe}")
        print("-" * 80)
        print(f"Total Trades:        {best.total_trades}")
        print(f"Win Rate:            {best.win_rate:.2%}")
        print(f"Profit Factor:       {best.profit_factor:.2f}")
        print(f"Total Return:        {best.total_return:.2%}")
        print(f"CAGR:                {best.cagr:.2%}")
        print(f"Max Drawdown:        {best.max_drawdown:.2%}")
        print(f"Sharpe Ratio:        {best.sharpe_ratio:.2f}")
        print(f"Sortino Ratio:       {best.sortino_ratio:.2f}")
        print(f"Calmar Ratio:        {best.calmar_ratio:.2f}")
        print(f"Avg Win:             {best.avg_win:.2%}")
        print(f"Avg Loss:            {best.avg_loss:.2%}")
        print(f"Largest Win:         {best.largest_win:.2%}")
        print(f"Largest Loss:        {best.largest_loss:.2%}")
        print(f"Avg Trade Duration:  {best.avg_trade_duration:.1f} bars")
        print(f"Max Consecutive Wins:  {best.consecutive_wins}")
        print(f"Max Consecutive Losses: {best.consecutive_losses}")
        print(f"Recovery Factor:     {best.recovery_factor:.2f}")
        print(f"Expectancy:          {best.expectancy:.4f}")


# Built-in strategy generators for comparison
def generate_rsi_signals(opens: np.ndarray, highs: np.ndarray,
                         lows: np.ndarray, closes: np.ndarray,
                         period: int = 14, oversold: int = 30,
                         overbought: int = 70) -> np.ndarray:
    """Generate RSI strategy signals."""
    # Calculate RSI
    deltas = np.diff(closes)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.zeros(len(closes))
    avg_loss = np.zeros(len(closes))
    
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Generate signals
    signals = np.zeros(len(closes))
    
    for i in range(1, len(closes)):
        if rsi[i-1] < oversold and rsi[i] >= oversold:  # Cross above oversold
            signals[i] = 1
        elif rsi[i-1] > overbought and rsi[i] <= overbought:  # Cross below overbought
            signals[i] = -1
    
    return signals


def generate_macd_signals(opens: np.ndarray, highs: np.ndarray,
                          lows: np.ndarray, closes: np.ndarray,
                          fast: int = 12, slow: int = 26,
                          signal_period: int = 9) -> np.ndarray:
    """Generate MACD crossover signals."""
    def ema(data, period):
        result = np.zeros_like(data)
        multiplier = 2 / (period + 1)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = (data[i] * multiplier) + (result[i-1] * (1 - multiplier))
        return result
    
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    
    signals = np.zeros(len(closes))
    
    for i in range(1, len(closes)):
        if macd_line[i-1] < signal_line[i-1] and macd_line[i] > signal_line[i]:
            signals[i] = 1
        elif macd_line[i-1] > signal_line[i-1] and macd_line[i] < signal_line[i]:
            signals[i] = -1
    
    return signals


def generate_ema_crossover_signals(opens: np.ndarray, highs: np.ndarray,
                                   lows: np.ndarray, closes: np.ndarray,
                                   fast: int = 9, slow: int = 21) -> np.ndarray:
    """Generate EMA crossover signals."""
    def ema(data, period):
        result = np.zeros_like(data)
        multiplier = 2 / (period + 1)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = (data[i] * multiplier) + (result[i-1] * (1 - multiplier))
        return result
    
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    
    signals = np.zeros(len(closes))
    
    for i in range(1, len(closes)):
        if ema_fast[i-1] < ema_slow[i-1] and ema_fast[i] > ema_slow[i]:
            signals[i] = 1
        elif ema_fast[i-1] > ema_slow[i-1] and ema_fast[i] < ema_slow[i]:
            signals[i] = -1
    
    return signals


def generate_buy_hold_signals(opens: np.ndarray, highs: np.ndarray,
                              lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    """Generate buy and hold signals (buy on day 1, never sell)."""
    signals = np.zeros(len(closes))
    signals[0] = 1  # Buy on first day
    return signals


# Demo
if __name__ == "__main__":
    print("=" * 80)
    print("PERFORMANCE BENCHMARK - DEMO")
    print("=" * 80)
    
    # Generate sample data
    np.random.seed(42)
    n = 500  # ~2 years
    
    # Trending market with noise
    trend = np.linspace(0, 50, n)
    noise = np.cumsum(np.random.randn(n) * 1.5)
    closes = 100 + trend + noise
    
    opens = np.roll(closes, 1) + np.random.randn(n) * 0.5
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n) * 0.8)
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n) * 0.8)
    
    # Create benchmark
    benchmark = PerformanceBenchmark(risk_free_rate=0.05, transaction_cost=0.001)
    
    # Add strategies
    benchmark.add_strategy('RSI', generate_rsi_signals(opens, highs, lows, closes))
    benchmark.add_strategy('MACD', generate_macd_signals(opens, highs, lows, closes))
    benchmark.add_strategy('EMA_Cross', generate_ema_crossover_signals(opens, highs, lows, closes))
    benchmark.add_strategy('Buy_Hold', generate_buy_hold_signals(opens, highs, lows, closes))
    
    # Run benchmark
    print("\nRunning benchmark...")
    result = benchmark.run_benchmark(opens, highs, lows, closes)
    
    # Print report
    benchmark.print_benchmark_report(result)
    
    print("\n[OK] Performance Benchmark Module Working!")
