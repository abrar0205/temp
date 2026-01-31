#!/usr/bin/env python3
"""
PORTFOLIO OPTIMIZATION MODULE
=============================
Advanced portfolio optimization techniques:

1. Mean-Variance (Markowitz) - Classic optimization
2. Risk Parity - Equal risk contribution
3. Maximum Sharpe Ratio - Optimal risk-adjusted returns
4. Minimum Volatility - Lowest risk portfolio
5. Black-Litterman - Incorporating views
6. Kelly Criterion (Multi-Asset) - Optimal betting
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PortfolioResult:
    """Result of portfolio optimization."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    method: str
    asset_names: List[str]


class PortfolioOptimizer:
    """
    Advanced portfolio optimization.
    
    Usage:
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize_markowitz(returns, target_return=0.1)
    """
    
    def __init__(self, risk_free_rate: float = 0.05):
        """
        Initialize optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 5% for India)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_portfolio_metrics(self, weights: np.ndarray, 
                                    mean_returns: np.ndarray,
                                    cov_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio return, volatility, and Sharpe ratio.
        """
        port_return = np.dot(weights, mean_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (port_return - self.risk_free_rate) / port_vol if port_vol > 0 else 0
        
        return port_return, port_vol, sharpe
    
    # ==========================================================================
    # MEAN-VARIANCE (MARKOWITZ) OPTIMIZATION
    # ==========================================================================
    
    def optimize_markowitz(self, returns: np.ndarray, 
                          target_return: Optional[float] = None,
                          asset_names: List[str] = None,
                          max_weight: float = 0.4,
                          min_weight: float = 0.0) -> PortfolioResult:
        """
        Classic Markowitz mean-variance optimization.
        
        Args:
            returns: Array of shape (n_periods, n_assets)
            target_return: Target annual return (None = max Sharpe)
            asset_names: Names of assets
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            
        Returns:
            PortfolioResult with optimal weights
        """
        n_assets = returns.shape[1]
        mean_returns = np.mean(returns, axis=0) * 252  # Annualize
        cov_matrix = np.cov(returns.T) * 252
        
        if asset_names is None:
            asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
        
        # Simple gradient descent optimization
        best_weights = np.ones(n_assets) / n_assets
        best_sharpe = -np.inf
        
        # Random search for good starting point
        for _ in range(1000):
            # Generate random weights
            weights = np.random.random(n_assets)
            weights = weights / weights.sum()
            
            # Apply constraints
            weights = np.clip(weights, min_weight, max_weight)
            weights = weights / weights.sum()
            
            ret, vol, sharpe = self.calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
            
            if target_return is not None:
                # Penalize if not meeting target
                penalty = abs(ret - target_return) * 10
                sharpe -= penalty
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = weights.copy()
        
        # Fine-tune with gradient descent
        learning_rate = 0.01
        for _ in range(100):
            gradients = np.zeros(n_assets)
            
            for i in range(n_assets):
                weights_plus = best_weights.copy()
                weights_plus[i] += 0.01
                weights_plus = weights_plus / weights_plus.sum()
                
                _, _, sharpe_plus = self.calculate_portfolio_metrics(
                    weights_plus, mean_returns, cov_matrix)
                
                gradients[i] = sharpe_plus - best_sharpe
            
            best_weights += learning_rate * gradients
            best_weights = np.clip(best_weights, min_weight, max_weight)
            best_weights = best_weights / best_weights.sum()
            
            _, _, best_sharpe = self.calculate_portfolio_metrics(
                best_weights, mean_returns, cov_matrix)
        
        ret, vol, sharpe = self.calculate_portfolio_metrics(best_weights, mean_returns, cov_matrix)
        
        return PortfolioResult(
            weights=best_weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            method='markowitz',
            asset_names=asset_names
        )
    
    # ==========================================================================
    # RISK PARITY
    # ==========================================================================
    
    def optimize_risk_parity(self, returns: np.ndarray,
                             asset_names: List[str] = None) -> PortfolioResult:
        """
        Risk Parity optimization - equal risk contribution from each asset.
        
        Each asset contributes equally to portfolio volatility.
        """
        n_assets = returns.shape[1]
        mean_returns = np.mean(returns, axis=0) * 252
        cov_matrix = np.cov(returns.T) * 252
        
        if asset_names is None:
            asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
        
        # Target risk contribution per asset
        target_risk = 1.0 / n_assets
        
        # Initialize with equal weights
        weights = np.ones(n_assets) / n_assets
        
        # Iterative optimization
        for _ in range(100):
            # Calculate marginal risk contributions
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            if port_vol == 0:
                break
            
            marginal_risk = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_risk / port_vol
            
            # Total risk contribution
            total_risk_contrib = np.sum(risk_contrib)
            if total_risk_contrib == 0:
                break
            
            risk_contrib_pct = risk_contrib / total_risk_contrib
            
            # Adjust weights to equalize risk contribution
            adjustment = target_risk - risk_contrib_pct
            weights = weights * (1 + 0.5 * adjustment)
            weights = np.maximum(weights, 0.01)  # Minimum weight
            weights = weights / weights.sum()
        
        ret, vol, sharpe = self.calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
        
        return PortfolioResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            method='risk_parity',
            asset_names=asset_names
        )
    
    # ==========================================================================
    # MAXIMUM SHARPE RATIO
    # ==========================================================================
    
    def optimize_max_sharpe(self, returns: np.ndarray,
                            asset_names: List[str] = None) -> PortfolioResult:
        """
        Find portfolio with maximum Sharpe ratio.
        """
        return self.optimize_markowitz(returns, target_return=None, 
                                       asset_names=asset_names)
    
    # ==========================================================================
    # MINIMUM VOLATILITY
    # ==========================================================================
    
    def optimize_min_volatility(self, returns: np.ndarray,
                                asset_names: List[str] = None) -> PortfolioResult:
        """
        Find minimum volatility portfolio.
        """
        n_assets = returns.shape[1]
        mean_returns = np.mean(returns, axis=0) * 252
        cov_matrix = np.cov(returns.T) * 252
        
        if asset_names is None:
            asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
        
        best_weights = np.ones(n_assets) / n_assets
        best_vol = np.inf
        
        # Random search
        for _ in range(1000):
            weights = np.random.random(n_assets)
            weights = weights / weights.sum()
            
            _, vol, _ = self.calculate_portfolio_metrics(weights, mean_returns, cov_matrix)
            
            if vol < best_vol:
                best_vol = vol
                best_weights = weights.copy()
        
        # Fine-tune
        learning_rate = 0.01
        for _ in range(100):
            gradients = np.zeros(n_assets)
            
            for i in range(n_assets):
                weights_plus = best_weights.copy()
                weights_plus[i] += 0.01
                weights_plus = weights_plus / weights_plus.sum()
                
                _, vol_plus, _ = self.calculate_portfolio_metrics(
                    weights_plus, mean_returns, cov_matrix)
                
                gradients[i] = best_vol - vol_plus  # Negative gradient (minimize)
            
            best_weights += learning_rate * gradients
            best_weights = np.maximum(best_weights, 0.01)
            best_weights = best_weights / best_weights.sum()
            
            _, best_vol, _ = self.calculate_portfolio_metrics(
                best_weights, mean_returns, cov_matrix)
        
        ret, vol, sharpe = self.calculate_portfolio_metrics(best_weights, mean_returns, cov_matrix)
        
        return PortfolioResult(
            weights=best_weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            method='min_volatility',
            asset_names=asset_names
        )
    
    # ==========================================================================
    # BLACK-LITTERMAN
    # ==========================================================================
    
    def optimize_black_litterman(self, returns: np.ndarray,
                                  views: Dict[int, float],
                                  view_confidence: float = 0.5,
                                  asset_names: List[str] = None) -> PortfolioResult:
        """
        Black-Litterman optimization with investor views.
        
        Args:
            returns: Historical returns
            views: Dict mapping asset index to expected return view
            view_confidence: Confidence in views (0-1)
            
        Example views:
            {0: 0.15, 2: -0.05}  # Asset 0 will return 15%, Asset 2 will return -5%
        """
        n_assets = returns.shape[1]
        
        # Prior: Equilibrium returns from market cap weights (approximated)
        cov_matrix = np.cov(returns.T) * 252
        
        # Risk aversion parameter
        delta = 2.5
        
        # Market equilibrium weights (equal weight approximation)
        w_market = np.ones(n_assets) / n_assets
        
        # Implied equilibrium returns
        pi = delta * np.dot(cov_matrix, w_market)
        
        if views:
            # Build view matrices
            n_views = len(views)
            P = np.zeros((n_views, n_assets))  # Pick matrix
            Q = np.zeros(n_views)  # View returns
            
            for i, (asset_idx, view_return) in enumerate(views.items()):
                P[i, asset_idx] = 1
                Q[i] = view_return
            
            # Uncertainty in views
            tau = 0.05
            omega = np.diag(np.diag(np.dot(np.dot(P, cov_matrix * tau), P.T))) * (1 - view_confidence)
            
            # Black-Litterman posterior
            # E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 * [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]
            
            tau_sigma_inv = np.linalg.inv(tau * cov_matrix)
            omega_inv = np.linalg.inv(omega + 1e-10 * np.eye(n_views))
            
            # Posterior precision
            posterior_precision = tau_sigma_inv + np.dot(np.dot(P.T, omega_inv), P)
            posterior_cov = np.linalg.inv(posterior_precision)
            
            # Posterior mean
            term1 = np.dot(tau_sigma_inv, pi)
            term2 = np.dot(np.dot(P.T, omega_inv), Q)
            posterior_mean = np.dot(posterior_cov, term1 + term2)
        else:
            posterior_mean = pi
        
        # Optimize using posterior
        if asset_names is None:
            asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
        
        # Use posterior as mean returns
        best_weights = np.ones(n_assets) / n_assets
        best_sharpe = -np.inf
        
        for _ in range(1000):
            weights = np.random.random(n_assets)
            weights = weights / weights.sum()
            
            ret = np.dot(weights, posterior_mean)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = weights.copy()
        
        ret = np.dot(best_weights, posterior_mean)
        vol = np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix, best_weights)))
        
        return PortfolioResult(
            weights=best_weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=best_sharpe,
            method='black_litterman',
            asset_names=asset_names
        )
    
    # ==========================================================================
    # KELLY CRITERION (MULTI-ASSET)
    # ==========================================================================
    
    def optimize_kelly(self, returns: np.ndarray,
                       asset_names: List[str] = None,
                       kelly_fraction: float = 0.5) -> PortfolioResult:
        """
        Multi-asset Kelly Criterion optimization.
        
        f* = Sigma^-1 * (mu - r) / (1 + f* Sigma^-1 * (mu - r))
        
        Using fractional Kelly (default 50%) for reduced volatility.
        """
        n_assets = returns.shape[1]
        mean_returns = np.mean(returns, axis=0) * 252
        cov_matrix = np.cov(returns.T) * 252
        
        if asset_names is None:
            asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
        
        # Excess returns over risk-free
        excess_returns = mean_returns - self.risk_free_rate
        
        # Inverse covariance matrix
        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_matrix)
        
        # Full Kelly weights
        kelly_weights = np.dot(cov_inv, excess_returns)
        
        # Apply fractional Kelly
        kelly_weights = kelly_weights * kelly_fraction
        
        # Normalize to sum to 1 (long only)
        kelly_weights = np.maximum(kelly_weights, 0)  # No shorting
        
        if kelly_weights.sum() > 0:
            kelly_weights = kelly_weights / kelly_weights.sum()
        else:
            kelly_weights = np.ones(n_assets) / n_assets
        
        ret, vol, sharpe = self.calculate_portfolio_metrics(kelly_weights, mean_returns, cov_matrix)
        
        return PortfolioResult(
            weights=kelly_weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            method=f'kelly_{kelly_fraction:.0%}',
            asset_names=asset_names
        )
    
    # ==========================================================================
    # EFFICIENT FRONTIER
    # ==========================================================================
    
    def generate_efficient_frontier(self, returns: np.ndarray,
                                    n_points: int = 50) -> List[Tuple[float, float]]:
        """
        Generate points on the efficient frontier.
        
        Returns:
            List of (volatility, return) tuples
        """
        mean_returns = np.mean(returns, axis=0) * 252
        
        min_ret = np.min(mean_returns)
        max_ret = np.max(mean_returns)
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier = []
        for target in target_returns:
            try:
                result = self.optimize_markowitz(returns, target_return=target)
                frontier.append((result.volatility, result.expected_return))
            except Exception:
                continue
        
        return frontier
    
    # ==========================================================================
    # PORTFOLIO COMPARISON
    # ==========================================================================
    
    def compare_all_methods(self, returns: np.ndarray,
                            asset_names: List[str] = None) -> Dict[str, PortfolioResult]:
        """
        Run all optimization methods and compare.
        """
        if asset_names is None:
            asset_names = [f"Asset_{i+1}" for i in range(returns.shape[1])]
        
        results = {}
        
        # Equal weight benchmark
        n_assets = returns.shape[1]
        equal_weights = np.ones(n_assets) / n_assets
        mean_returns = np.mean(returns, axis=0) * 252
        cov_matrix = np.cov(returns.T) * 252
        ret, vol, sharpe = self.calculate_portfolio_metrics(equal_weights, mean_returns, cov_matrix)
        results['equal_weight'] = PortfolioResult(
            weights=equal_weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            method='equal_weight',
            asset_names=asset_names
        )
        
        # All optimization methods
        results['markowitz'] = self.optimize_markowitz(returns, asset_names=asset_names)
        results['risk_parity'] = self.optimize_risk_parity(returns, asset_names=asset_names)
        results['min_volatility'] = self.optimize_min_volatility(returns, asset_names=asset_names)
        results['kelly_50'] = self.optimize_kelly(returns, asset_names=asset_names, kelly_fraction=0.5)
        results['kelly_25'] = self.optimize_kelly(returns, asset_names=asset_names, kelly_fraction=0.25)
        
        return results


def print_portfolio_result(result: PortfolioResult):
    """Pretty print portfolio result."""
    print(f"\n{'='*50}")
    print(f"PORTFOLIO: {result.method.upper()}")
    print(f"{'='*50}")
    print(f"Expected Annual Return: {result.expected_return:.2%}")
    print(f"Annual Volatility:      {result.volatility:.2%}")
    print(f"Sharpe Ratio:           {result.sharpe_ratio:.2f}")
    print(f"\nWeights:")
    for name, weight in zip(result.asset_names, result.weights):
        print(f"  {name:15s}: {weight:6.2%}")


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION - DEMO")
    print("=" * 60)
    
    # Generate sample returns data
    np.random.seed(42)
    n_days = 252 * 3  # 3 years
    n_assets = 5
    
    # Different assets with varying characteristics
    asset_names = ['NIFTY50', 'BANKNIFTY', 'RELIANCE', 'TCS', 'GOLD']
    
    # Simulate returns (annualized ~10-20% with different vols)
    returns = np.zeros((n_days, n_assets))
    returns[:, 0] = np.random.randn(n_days) * 0.01 + 0.0004  # NIFTY: 10% annual
    returns[:, 1] = np.random.randn(n_days) * 0.015 + 0.0005  # BankNifty: 12.5% annual, higher vol
    returns[:, 2] = np.random.randn(n_days) * 0.018 + 0.0006  # Reliance: 15% annual
    returns[:, 3] = np.random.randn(n_days) * 0.012 + 0.0005  # TCS: 12.5% annual, lower vol
    returns[:, 4] = np.random.randn(n_days) * 0.008 + 0.0003  # Gold: 7.5% annual, low vol
    
    # Add correlations
    correlation = 0.3
    for i in range(1, n_days):
        returns[i, 1] += correlation * returns[i, 0]
        returns[i, 2] += correlation * returns[i, 0]
        returns[i, 3] += correlation * returns[i, 2]
    
    # Run optimization
    optimizer = PortfolioOptimizer(risk_free_rate=0.06)  # 6% risk-free (India)
    
    # Compare all methods
    print("\nComparing all optimization methods...")
    results = optimizer.compare_all_methods(returns, asset_names)
    
    # Print results
    for method, result in results.items():
        print_portfolio_result(result)
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"{'Method':<20} {'Return':>10} {'Volatility':>12} {'Sharpe':>10}")
    print("-" * 55)
    for method, result in results.items():
        print(f"{method:<20} {result.expected_return:>10.2%} "
              f"{result.volatility:>12.2%} {result.sharpe_ratio:>10.2f}")
    
    # Best portfolio
    best = max(results.values(), key=lambda x: x.sharpe_ratio)
    print(f"\nBest Portfolio: {best.method} (Sharpe: {best.sharpe_ratio:.2f})")
    
    print("\n[OK] Portfolio Optimization Module Working!")
