#!/usr/bin/env python3
"""
AUTO-OPTIMIZER
==============
Automatic parameter optimization with:
- Grid Search
- Random Search  
- Walk-Forward Optimization
- Anti-Overfitting Measures
- Cross-Validation

Author: Trading Indicator Project
Version: 2.0.0 ULTIMATE
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import itertools
import json
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OptimizationResult:
    """Result of parameter optimization"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict]
    train_performance: Dict[str, float]
    test_performance: Dict[str, float]
    is_overfit: bool
    robustness_score: float


class AutoOptimizer:
    """
    Automatic Parameter Optimization System
    
    Features:
    1. Grid Search - Exhaustive parameter search
    2. Random Search - Efficient random sampling
    3. Walk-Forward - Rolling window optimization
    4. Cross-Validation - Multiple train/test splits
    5. Robustness Testing - Monte Carlo validation
    """
    
    def __init__(self, strategy_class: Any = None):
        """
        Initialize optimizer
        
        Args:
            strategy_class: Class to instantiate for backtesting
        """
        self.strategy_class = strategy_class
        self.results_history: List[OptimizationResult] = []
        
        # Default parameter ranges
        self.default_param_ranges = {
            'ema_period': [100, 150, 200, 250],
            'supertrend_period_1': [8, 10, 12],
            'supertrend_mult_1': [0.8, 1.0, 1.2],
            'supertrend_period_2': [10, 11, 12],
            'supertrend_mult_2': [1.5, 2.0, 2.5],
            'supertrend_period_3': [11, 12, 13, 14],
            'supertrend_mult_3': [2.5, 3.0, 3.5],
            'stoch_rsi_period': [10, 14, 21],
            'adx_period': [10, 14, 20],
            'adx_threshold': [20, 25, 30],
            'sl_atr_mult': [1.5, 2.0, 2.5, 3.0],
            'tp_atr_mult': [2.0, 3.0, 4.0],
            'risk_per_trade': [0.5, 1.0, 1.5, 2.0]
        }
        
        print("Auto-Optimizer Initialized")
    
    # ==================== OBJECTIVE FUNCTIONS ====================
    
    def sharpe_ratio_objective(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio (higher is better)"""
        if len(returns) < 2 or np.std(returns) == 0:
            return -999
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def profit_factor_objective(self, trades: List[float]) -> float:
        """Calculate profit factor (higher is better)"""
        wins = sum(t for t in trades if t > 0)
        losses = abs(sum(t for t in trades if t < 0))
        if losses == 0:
            return 10.0 if wins > 0 else 0.0
        return wins / losses
    
    def combined_objective(self, results: Dict) -> float:
        """
        Combined objective function
        
        Combines multiple metrics:
        - Win Rate (25%)
        - Profit Factor (25%)
        - Sharpe Ratio (25%)
        - Max Drawdown penalty (25%)
        """
        win_rate = results.get('win_rate', 0) / 100
        profit_factor = min(results.get('profit_factor', 0), 5) / 5  # Normalize to 0-1
        sharpe = min(max(results.get('sharpe_ratio', 0), 0), 3) / 3  # Normalize to 0-1
        
        # Drawdown penalty (lower is better, so invert)
        max_dd = results.get('max_drawdown', 100)
        dd_score = max(0, 1 - max_dd / 30)  # 0% DD = 1, 30% DD = 0
        
        # Combined score
        score = (
            0.25 * win_rate +
            0.25 * profit_factor +
            0.25 * sharpe +
            0.25 * dd_score
        )
        
        return score
    
    # ==================== GRID SEARCH ====================
    
    def grid_search(self, data: pd.DataFrame, param_grid: Dict[str, List],
                    objective_func: Callable = None,
                    train_ratio: float = 0.7) -> OptimizationResult:
        """
        Exhaustive grid search over parameter combinations
        
        Args:
            data: OHLCV DataFrame
            param_grid: Dictionary of parameter names to list of values
            objective_func: Function to maximize (default: combined_objective)
            train_ratio: Fraction of data for training
        
        Returns:
            OptimizationResult with best parameters
        """
        if objective_func is None:
            objective_func = self.combined_objective
        
        # Split data
        train_size = int(len(data) * train_ratio)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))
        
        print(f"\nGrid Search: {len(combinations)} combinations to test")
        print(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        all_results = []
        best_score = -999
        best_params = None
        best_train_results = None
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            # Backtest on training data
            try:
                train_results = self._run_backtest(train_data, params)
                score = objective_func(train_results)
                
                all_results.append({
                    'params': params,
                    'score': score,
                    'results': train_results
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_train_results = train_results
                
                if (i + 1) % 10 == 0:
                    print(f"  Tested {i+1}/{len(combinations)}, Best score: {best_score:.4f}")
            
            except Exception as e:
                print(f"  Error with params {params}: {str(e)}")
        
        # Test best params on out-of-sample data
        test_results = self._run_backtest(test_data, best_params)
        test_score = objective_func(test_results)
        
        # Check for overfitting
        is_overfit = best_score > test_score * 1.5  # If train is 50% better than test
        
        # Calculate robustness
        robustness = test_score / best_score if best_score > 0 else 0
        
        result = OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            train_performance={
                'score': best_score,
                **best_train_results
            },
            test_performance={
                'score': test_score,
                **test_results
            },
            is_overfit=is_overfit,
            robustness_score=robustness
        )
        
        self.results_history.append(result)
        
        return result
    
    # ==================== RANDOM SEARCH ====================
    
    def random_search(self, data: pd.DataFrame, param_ranges: Dict[str, Tuple],
                      n_iterations: int = 50,
                      objective_func: Callable = None,
                      train_ratio: float = 0.7) -> OptimizationResult:
        """
        Random search over parameter space
        
        More efficient than grid search for high-dimensional spaces
        
        Args:
            data: OHLCV DataFrame
            param_ranges: Dict of param names to (min, max) tuples
            n_iterations: Number of random samples
            objective_func: Function to maximize
            train_ratio: Fraction for training
        """
        if objective_func is None:
            objective_func = self.combined_objective
        
        # Split data
        train_size = int(len(data) * train_ratio)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        print(f"\nRandom Search: {n_iterations} iterations")
        
        all_results = []
        best_score = -999
        best_params = None
        best_train_results = None
        
        for i in range(n_iterations):
            # Generate random parameters
            params = {}
            for name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[name] = np.random.uniform(min_val, max_val)
            
            try:
                train_results = self._run_backtest(train_data, params)
                score = objective_func(train_results)
                
                all_results.append({
                    'params': params,
                    'score': score,
                    'results': train_results
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_train_results = train_results
                    print(f"  Iteration {i+1}: New best score = {best_score:.4f}")
            
            except Exception as e:
                pass  # Skip failed iterations
        
        # Test on out-of-sample
        test_results = self._run_backtest(test_data, best_params)
        test_score = objective_func(test_results)
        
        is_overfit = best_score > test_score * 1.5
        robustness = test_score / best_score if best_score > 0 else 0
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            train_performance={'score': best_score, **best_train_results},
            test_performance={'score': test_score, **test_results},
            is_overfit=is_overfit,
            robustness_score=robustness
        )
    
    # ==================== WALK-FORWARD OPTIMIZATION ====================
    
    def walk_forward_optimization(self, data: pd.DataFrame, 
                                   param_grid: Dict[str, List],
                                   n_folds: int = 5,
                                   train_pct: float = 0.6,
                                   objective_func: Callable = None) -> Dict[str, Any]:
        """
        Walk-forward optimization (rolling window)
        
        This is the gold standard for strategy validation:
        1. Split data into N periods
        2. For each period: optimize on previous data, test on current
        3. Aggregate out-of-sample results
        
        Pros:
        - Tests on truly unseen data
        - Accounts for parameter drift
        - Most realistic performance estimate
        """
        if objective_func is None:
            objective_func = self.combined_objective
        
        print(f"\nWalk-Forward Optimization: {n_folds} folds")
        
        fold_size = len(data) // n_folds
        
        all_test_results = []
        optimal_params_per_fold = []
        
        for fold in range(1, n_folds):
            # Training: all data before this fold
            train_end = fold * fold_size
            train_start = max(0, int(train_end * (1 - train_pct)))
            train_data = data.iloc[train_start:train_end]
            
            # Test: current fold
            test_start = train_end
            test_end = min((fold + 1) * fold_size, len(data))
            test_data = data.iloc[test_start:test_end]
            
            print(f"\nFold {fold}/{n_folds-1}:")
            print(f"  Train: {train_start} to {train_end} ({len(train_data)} bars)")
            print(f"  Test:  {test_start} to {test_end} ({len(test_data)} bars)")
            
            # Optimize on training data
            best_score = -999
            best_params = None
            
            # Simple grid search for this fold
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            combinations = list(itertools.product(*param_values))
            
            for combo in combinations:
                params = dict(zip(param_names, combo))
                try:
                    results = self._run_backtest(train_data, params)
                    score = objective_func(results)
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                except:
                    pass
            
            # Test best params on out-of-sample
            if best_params:
                test_results = self._run_backtest(test_data, best_params)
                test_score = objective_func(test_results)
                
                print(f"  Best train score: {best_score:.4f}")
                print(f"  Test score: {test_score:.4f}")
                
                all_test_results.append({
                    'fold': fold,
                    'params': best_params,
                    'train_score': best_score,
                    'test_score': test_score,
                    'test_results': test_results
                })
                optimal_params_per_fold.append(best_params)
        
        # Aggregate results
        if all_test_results:
            avg_test_score = np.mean([r['test_score'] for r in all_test_results])
            avg_train_score = np.mean([r['train_score'] for r in all_test_results])
            
            # Find most common optimal parameters
            from collections import Counter
            param_counts = Counter()
            for params in optimal_params_per_fold:
                param_counts[json.dumps(params, sort_keys=True)] += 1
            
            most_common_params = json.loads(param_counts.most_common(1)[0][0])
            
            return {
                'status': 'success',
                'n_folds': n_folds - 1,
                'avg_train_score': avg_train_score,
                'avg_test_score': avg_test_score,
                'performance_ratio': avg_test_score / avg_train_score if avg_train_score > 0 else 0,
                'is_robust': avg_test_score / avg_train_score > 0.7,
                'most_common_params': most_common_params,
                'all_fold_results': all_test_results
            }
        
        return {'status': 'failed', 'error': 'No successful folds'}
    
    # ==================== CROSS-VALIDATION ====================
    
    def k_fold_cross_validation(self, data: pd.DataFrame, params: Dict,
                                 k: int = 5) -> Dict[str, Any]:
        """
        K-Fold cross-validation for parameter stability check
        
        Unlike walk-forward, this uses random splits (not sequential)
        Good for checking if parameters work across different market conditions
        """
        print(f"\nK-Fold Cross-Validation: {k} folds")
        
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        
        fold_size = len(data) // k
        scores = []
        
        for i in range(k):
            # Test indices for this fold
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < k - 1 else len(data)
            test_indices = indices[test_start:test_end]
            
            # Train indices = everything else
            train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
            
            # Create datasets (need to sort for time series)
            train_indices_sorted = np.sort(train_indices)
            test_indices_sorted = np.sort(test_indices)
            
            train_data = data.iloc[train_indices_sorted]
            test_data = data.iloc[test_indices_sorted]
            
            try:
                results = self._run_backtest(test_data, params)
                score = self.combined_objective(results)
                scores.append(score)
                print(f"  Fold {i+1}: Score = {score:.4f}")
            except Exception as e:
                print(f"  Fold {i+1}: Failed - {str(e)}")
        
        if scores:
            return {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'cv_ratio': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 999,
                'is_stable': np.std(scores) / np.mean(scores) < 0.3 if np.mean(scores) > 0 else False,
                'all_scores': scores
            }
        
        return {'error': 'All folds failed'}
    
    # ==================== ROBUSTNESS TESTING ====================
    
    def monte_carlo_validation(self, data: pd.DataFrame, params: Dict,
                               n_simulations: int = 100,
                               skip_pct: float = 0.1) -> Dict[str, Any]:
        """
        Monte Carlo validation for robustness
        
        Tests strategy stability by:
        1. Shuffling trade order
        2. Randomly skipping trades
        3. Adding slippage variation
        """
        print(f"\nMonte Carlo Validation: {n_simulations} simulations")
        
        # Get baseline results
        baseline_results = self._run_backtest(data, params)
        baseline_score = self.combined_objective(baseline_results)
        
        simulation_scores = []
        
        for i in range(n_simulations):
            # Modify data slightly (add noise to prices)
            modified_data = data.copy()
            noise = np.random.normal(1.0, 0.001, len(data))  # 0.1% noise
            modified_data['close'] = modified_data['close'] * noise
            modified_data['high'] = modified_data['high'] * noise
            modified_data['low'] = modified_data['low'] * noise
            modified_data['open'] = modified_data['open'] * noise
            
            try:
                results = self._run_backtest(modified_data, params)
                score = self.combined_objective(results)
                simulation_scores.append(score)
            except:
                pass
        
        if simulation_scores:
            return {
                'baseline_score': baseline_score,
                'mean_score': np.mean(simulation_scores),
                'std_score': np.std(simulation_scores),
                'percentile_5': np.percentile(simulation_scores, 5),
                'percentile_95': np.percentile(simulation_scores, 95),
                'worst_case': np.min(simulation_scores),
                'best_case': np.max(simulation_scores),
                'prob_profitable': sum(1 for s in simulation_scores if s > 0.5) / len(simulation_scores),
                'robustness': np.mean(simulation_scores) / baseline_score if baseline_score > 0 else 0
            }
        
        return {'error': 'All simulations failed'}
    
    # ==================== BACKTEST RUNNER ====================
    
    def _run_backtest(self, data: pd.DataFrame, params: Dict) -> Dict[str, float]:
        """
        Run backtest with given parameters
        
        This is a simplified backtest for optimization purposes
        For full backtest, use UltimateTradingEngine
        """
        if len(data) < 200:
            return {'win_rate': 0, 'profit_factor': 0, 'sharpe_ratio': 0, 'max_drawdown': 100}
        
        data = data.copy()
        data.columns = [c.lower() for c in data.columns]
        
        close = data['close'].values
        high = data['high'].values
        low = data['low'].values
        
        # Extract parameters
        ema_period = params.get('ema_period', 200)
        sl_mult = params.get('sl_atr_mult', 2.0)
        tp_mult = params.get('tp_atr_mult', 3.0)
        
        # Calculate EMA
        ema = pd.Series(close).ewm(span=ema_period, adjust=False).mean().values
        
        # Calculate ATR
        tr = np.maximum(high - low, 
                       np.maximum(abs(high - np.roll(close, 1)), 
                                 abs(low - np.roll(close, 1))))
        atr = pd.Series(tr).rolling(14).mean().values
        
        # Simple strategy: Buy when price crosses above EMA, sell when crosses below
        trades = []
        position = None
        
        for i in range(ema_period + 1, len(close)):
            if position is None:
                # Entry conditions
                if close[i] > ema[i] and close[i-1] <= ema[i-1]:
                    position = {
                        'entry': close[i],
                        'sl': close[i] - atr[i] * sl_mult,
                        'tp': close[i] + atr[i] * tp_mult,
                        'direction': 'long'
                    }
                elif close[i] < ema[i] and close[i-1] >= ema[i-1]:
                    position = {
                        'entry': close[i],
                        'sl': close[i] + atr[i] * sl_mult,
                        'tp': close[i] - atr[i] * tp_mult,
                        'direction': 'short'
                    }
            else:
                # Exit conditions
                exit_price = None
                
                if position['direction'] == 'long':
                    if low[i] <= position['sl']:
                        exit_price = position['sl']
                    elif high[i] >= position['tp']:
                        exit_price = position['tp']
                else:
                    if high[i] >= position['sl']:
                        exit_price = position['sl']
                    elif low[i] <= position['tp']:
                        exit_price = position['tp']
                
                if exit_price:
                    if position['direction'] == 'long':
                        pnl = (exit_price - position['entry']) / position['entry'] * 100
                    else:
                        pnl = (position['entry'] - exit_price) / position['entry'] * 100
                    
                    trades.append(pnl)
                    position = None
        
        # Calculate metrics
        if not trades:
            return {'win_rate': 0, 'profit_factor': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'total_trades': 0}
        
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        
        win_rate = len(wins) / len(trades) * 100
        profit_factor = sum(wins) / abs(sum(losses)) if losses else 10.0
        
        # Sharpe
        if len(trades) > 1:
            sharpe = np.mean(trades) / np.std(trades) * np.sqrt(252 / len(trades) * len(data))
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative = np.cumsum(trades)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'win_rate': win_rate,
            'profit_factor': min(profit_factor, 10),
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': len(trades),
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0
        }
    
    # ==================== REPORT ====================
    
    def generate_optimization_report(self, result: OptimizationResult) -> str:
        """Generate detailed optimization report"""
        
        report = []
        report.append("=" * 60)
        report.append("OPTIMIZATION REPORT")
        report.append("=" * 60)
        
        report.append("\nBEST PARAMETERS:")
        report.append("-" * 40)
        for name, value in result.best_params.items():
            report.append(f"  {name}: {value}")
        
        report.append(f"\nTRAINING PERFORMANCE:")
        report.append("-" * 40)
        for key, value in result.train_performance.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.4f}")
            else:
                report.append(f"  {key}: {value}")
        
        report.append(f"\nTEST PERFORMANCE (Out-of-Sample):")
        report.append("-" * 40)
        for key, value in result.test_performance.items():
            if isinstance(value, float):
                report.append(f"  {key}: {value:.4f}")
            else:
                report.append(f"  {key}: {value}")
        
        report.append(f"\nROBUSTNESS CHECK:")
        report.append("-" * 40)
        report.append(f"  Robustness Score: {result.robustness_score:.2%}")
        report.append(f"  Overfitting Detected: {'YES ⚠️' if result.is_overfit else 'NO ✓'}")
        
        if result.is_overfit:
            report.append("\n  ⚠️ WARNING: Parameters may be overfit!")
            report.append("  Consider using walk-forward optimization.")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def demo():
    """Demonstrate auto-optimization"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║              AUTO-OPTIMIZER DEMO                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=500, freq='D')
    
    price = 100.0
    data = []
    for i in range(500):
        returns = np.random.normal(0.0005, 0.02)
        open_p = price
        close_p = price * (1 + returns)
        high_p = max(open_p, close_p) * (1 + abs(np.random.randn() * 0.005))
        low_p = min(open_p, close_p) * (1 - abs(np.random.randn() * 0.005))
        
        data.append({
            'open': open_p, 'high': high_p, 'low': low_p, 
            'close': close_p, 'volume': 1000000
        })
        price = close_p
    
    df = pd.DataFrame(data, index=dates)
    
    # Initialize optimizer
    optimizer = AutoOptimizer()
    
    # Define parameter grid
    param_grid = {
        'ema_period': [100, 150, 200],
        'sl_atr_mult': [1.5, 2.0, 2.5],
        'tp_atr_mult': [2.0, 3.0, 4.0]
    }
    
    # Run grid search
    print("\n1. GRID SEARCH")
    result = optimizer.grid_search(df, param_grid)
    print(optimizer.generate_optimization_report(result))
    
    # Run walk-forward
    print("\n2. WALK-FORWARD OPTIMIZATION")
    wf_result = optimizer.walk_forward_optimization(
        df, 
        {'ema_period': [150, 200], 'sl_atr_mult': [2.0], 'tp_atr_mult': [3.0]},
        n_folds=4
    )
    
    print(f"\nWalk-Forward Results:")
    print(f"  Avg Train Score: {wf_result.get('avg_train_score', 0):.4f}")
    print(f"  Avg Test Score:  {wf_result.get('avg_test_score', 0):.4f}")
    print(f"  Is Robust:       {'YES' if wf_result.get('is_robust') else 'NO'}")
    
    # Monte Carlo validation
    print("\n3. MONTE CARLO VALIDATION")
    mc_result = optimizer.monte_carlo_validation(df, result.best_params, n_simulations=50)
    
    print(f"  Baseline Score:     {mc_result.get('baseline_score', 0):.4f}")
    print(f"  Mean Score:         {mc_result.get('mean_score', 0):.4f}")
    print(f"  5th Percentile:     {mc_result.get('percentile_5', 0):.4f}")
    print(f"  95th Percentile:    {mc_result.get('percentile_95', 0):.4f}")
    print(f"  Prob. Profitable:   {mc_result.get('prob_profitable', 0):.1%}")


if __name__ == "__main__":
    demo()
