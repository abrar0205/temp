#!/usr/bin/env python3
"""
ADVANCED RISK MANAGEMENT MODULE
===============================
Professional-grade risk management with:
- Kelly Criterion position sizing
- Value at Risk (VAR)
- Maximum Drawdown Control
- Correlation-based portfolio risk
- Dynamic position sizing
- Volatility-adjusted stops

Author: Trading Indicator Project
Version: 2.0.0 ULTIMATE
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskParameters:
    """Risk parameters for a trade"""
    position_size_pct: float  # % of capital
    position_size_shares: int  # Number of shares
    stop_loss: float  # Stop loss price
    take_profit: float  # Take profit price
    risk_reward_ratio: float
    max_loss_amount: float  # $ at risk
    expected_value: float  # Expected profit
    kelly_fraction: float
    var_95: float  # 95% Value at Risk
    var_99: float  # 99% Value at Risk


class AdvancedRiskManager:
    """
    Advanced Risk Management System
    
    Features:
    1. Kelly Criterion - Optimal position sizing
    2. Value at Risk (VAR) - Statistical risk measure
    3. Maximum Drawdown Control - Circuit breaker
    4. Correlation Risk - Portfolio diversification
    5. Volatility Adjustment - ATR-based sizing
    6. Dynamic Stops - Trailing and volatility stops
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize Risk Manager"""
        
        self.config = {
            # Capital management
            'initial_capital': 100000.0,
            'max_risk_per_trade_pct': 2.0,  # Max 2% risk per trade
            'max_portfolio_risk_pct': 10.0,  # Max 10% total portfolio risk
            'max_position_size_pct': 25.0,  # Max 25% in single position
            'max_correlated_exposure_pct': 40.0,  # Max 40% in correlated assets
            
            # Drawdown control
            'max_drawdown_pct': 15.0,  # Stop trading at 15% drawdown
            'drawdown_reduction_start': 10.0,  # Start reducing size at 10%
            'drawdown_reduction_rate': 0.5,  # Reduce by 50% for each 5% DD
            
            # Kelly criterion
            'kelly_fraction': 0.25,  # Use 25% of Kelly (quarter Kelly)
            'min_kelly': 0.01,  # Minimum 1% position
            'max_kelly': 0.20,  # Maximum 20% position
            
            # VAR parameters
            'var_confidence_95': 0.95,
            'var_confidence_99': 0.99,
            'var_lookback': 100,  # Days for VAR calculation
            
            # Stop loss
            'atr_period': 14,
            'sl_atr_multiplier': 2.0,
            'trailing_atr_multiplier': 1.5,
            'use_volatility_stops': True,
            
            # Transaction costs
            'commission_pct': 0.1,
            'slippage_pct': 0.05,
        }
        
        if config:
            self.config.update(config)
        
        # State
        self.current_capital = self.config['initial_capital']
        self.peak_capital = self.config['initial_capital']
        self.current_drawdown = 0.0
        self.positions: List[Dict] = []
        self.trade_history: List[Dict] = []
        
        print("Advanced Risk Manager Initialized")
        print(f"  Max risk per trade: {self.config['max_risk_per_trade_pct']}%")
        print(f"  Max drawdown limit: {self.config['max_drawdown_pct']}%")
        print(f"  Kelly fraction: {self.config['kelly_fraction']*100}%")
    
    # ==================== KELLY CRITERION ====================
    
    def calculate_kelly_criterion(self, win_rate: float, avg_win: float, 
                                   avg_loss: float) -> float:
        """
        Calculate Kelly Criterion for optimal position sizing
        
        Formula: f* = (p * b - q) / b
        where:
            f* = optimal fraction of capital
            p = probability of winning
            q = probability of losing (1 - p)
            b = ratio of average win to average loss
        
        Returns:
            Optimal fraction of capital (0-1)
        """
        if avg_loss == 0 or avg_win == 0:
            return self.config['min_kelly']
        
        p = win_rate  # Win probability
        q = 1 - p  # Loss probability
        b = avg_win / avg_loss  # Win/loss ratio
        
        # Kelly formula
        kelly = (p * b - q) / b
        
        # Apply fraction (quarter Kelly is common for safety)
        kelly *= self.config['kelly_fraction']
        
        # Clamp to bounds
        kelly = max(self.config['min_kelly'], min(self.config['max_kelly'], kelly))
        
        return kelly
    
    def estimate_kelly_from_history(self) -> float:
        """Estimate Kelly from trade history"""
        if len(self.trade_history) < 20:
            return self.config['min_kelly']
        
        recent_trades = self.trade_history[-100:]  # Last 100 trades
        
        wins = [t['pnl'] for t in recent_trades if t['pnl'] > 0]
        losses = [t['pnl'] for t in recent_trades if t['pnl'] < 0]
        
        if not wins or not losses:
            return self.config['min_kelly']
        
        win_rate = len(wins) / len(recent_trades)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        return self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
    
    # ==================== VALUE AT RISK (VAR) ====================
    
    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VAR)
        
        VAR answers: "What is the maximum loss at X% confidence?"
        
        Historical method: Use actual return distribution
        
        Returns:
            VAR as a positive percentage (potential loss)
        """
        if len(returns) < 10:
            return 0.05  # Default 5% VAR
        
        # Historical VAR: percentile of returns
        var = -np.percentile(returns, (1 - confidence) * 100)
        
        return max(0, var)
    
    def calculate_parametric_var(self, returns: np.ndarray, confidence: float = 0.95,
                                  holding_period: int = 1) -> float:
        """
        Calculate Parametric VAR (assumes normal distribution)
        
        Formula: VAR = μ - Z * σ * √t
        where:
            μ = mean return
            Z = Z-score for confidence level
            σ = standard deviation
            t = holding period
        """
        if len(returns) < 10:
            return 0.05
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Z-scores for common confidence levels
        z_scores = {0.90: 1.282, 0.95: 1.645, 0.99: 2.326}
        z = z_scores.get(confidence, 1.645)
        
        var = -(mean_return - z * std_return * np.sqrt(holding_period))
        
        return max(0, var)
    
    def calculate_conditional_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional VAR (Expected Shortfall)
        
        CVaR = Average of losses beyond VAR
        This is a more conservative risk measure
        """
        if len(returns) < 10:
            return 0.08
        
        var = self.calculate_var(returns, confidence)
        
        # Get returns worse than VAR
        tail_losses = returns[returns < -var]
        
        if len(tail_losses) == 0:
            return var
        
        cvar = -np.mean(tail_losses)
        
        return max(var, cvar)
    
    # ==================== DRAWDOWN MANAGEMENT ====================
    
    def update_drawdown(self, current_equity: float):
        """Update drawdown tracking"""
        if current_equity > self.peak_capital:
            self.peak_capital = current_equity
        
        self.current_capital = current_equity
        self.current_drawdown = (self.peak_capital - current_equity) / self.peak_capital * 100
    
    def get_drawdown_adjusted_size(self, base_size: float) -> float:
        """
        Reduce position size based on current drawdown
        
        Logic:
        - Below 10% DD: Full size
        - 10-15% DD: Reduce by 50%
        - 15%+ DD: Stop trading
        """
        if self.current_drawdown >= self.config['max_drawdown_pct']:
            return 0.0  # Stop trading
        
        if self.current_drawdown <= self.config['drawdown_reduction_start']:
            return base_size  # Full size
        
        # Linear reduction
        excess_dd = self.current_drawdown - self.config['drawdown_reduction_start']
        dd_range = self.config['max_drawdown_pct'] - self.config['drawdown_reduction_start']
        reduction = excess_dd / dd_range * self.config['drawdown_reduction_rate']
        
        adjusted_size = base_size * (1 - reduction)
        
        return max(0, adjusted_size)
    
    def is_trading_allowed(self) -> Tuple[bool, str]:
        """Check if trading is allowed based on risk limits"""
        
        # Check drawdown
        if self.current_drawdown >= self.config['max_drawdown_pct']:
            return False, f"Max drawdown reached: {self.current_drawdown:.1f}%"
        
        # Check portfolio risk
        total_risk = sum(p.get('risk_pct', 0) for p in self.positions)
        if total_risk >= self.config['max_portfolio_risk_pct']:
            return False, f"Max portfolio risk reached: {total_risk:.1f}%"
        
        return True, "Trading allowed"
    
    # ==================== POSITION SIZING ====================
    
    def calculate_position_size(self, entry_price: float, stop_loss: float,
                                 win_rate: float = 0.55, 
                                 avg_win_pct: float = 2.0,
                                 avg_loss_pct: float = 1.0,
                                 volatility: float = None) -> RiskParameters:
        """
        Calculate optimal position size using multiple methods
        
        Methods combined:
        1. Fixed fractional (risk per trade)
        2. Kelly criterion
        3. Volatility adjustment
        4. Drawdown adjustment
        
        Returns:
            RiskParameters with all sizing details
        """
        
        # 1. Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        risk_pct = risk_per_share / entry_price * 100
        
        # 2. Fixed fractional sizing
        max_risk_amount = self.current_capital * (self.config['max_risk_per_trade_pct'] / 100)
        fixed_fractional_shares = int(max_risk_amount / risk_per_share) if risk_per_share > 0 else 0
        
        # 3. Kelly criterion sizing
        kelly = self.calculate_kelly_criterion(win_rate, avg_win_pct, avg_loss_pct)
        kelly_amount = self.current_capital * kelly
        kelly_shares = int(kelly_amount / entry_price)
        
        # 4. Volatility adjustment (if provided)
        if volatility and volatility > 0:
            # Higher volatility = smaller position
            vol_adjustment = 0.02 / volatility  # Target 2% daily volatility
            vol_adjustment = min(2.0, max(0.5, vol_adjustment))  # Clamp 0.5x - 2x
        else:
            vol_adjustment = 1.0
        
        # 5. Combine methods (take minimum for safety)
        base_shares = min(fixed_fractional_shares, kelly_shares)
        
        # 6. Apply volatility adjustment
        adjusted_shares = int(base_shares * vol_adjustment)
        
        # 7. Apply drawdown adjustment
        dd_multiplier = self.get_drawdown_adjusted_size(1.0)
        final_shares = int(adjusted_shares * dd_multiplier)
        
        # 8. Apply maximum position limit
        max_position_amount = self.current_capital * (self.config['max_position_size_pct'] / 100)
        max_shares = int(max_position_amount / entry_price)
        final_shares = min(final_shares, max_shares)
        
        # Calculate position details
        position_amount = final_shares * entry_price
        position_pct = position_amount / self.current_capital * 100
        max_loss = final_shares * risk_per_share
        
        # Calculate risk/reward
        rr_ratio = avg_win_pct / avg_loss_pct if avg_loss_pct > 0 else 2.0
        take_profit = entry_price + (risk_per_share * rr_ratio) if entry_price > stop_loss else entry_price - (risk_per_share * rr_ratio)
        
        # Calculate expected value
        expected_value = (win_rate * avg_win_pct - (1 - win_rate) * avg_loss_pct) / 100 * position_amount
        
        # Calculate VAR for this position
        daily_vol = volatility if volatility else 0.02
        var_95 = position_amount * daily_vol * 1.645
        var_99 = position_amount * daily_vol * 2.326
        
        return RiskParameters(
            position_size_pct=position_pct,
            position_size_shares=final_shares,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=rr_ratio,
            max_loss_amount=max_loss,
            expected_value=expected_value,
            kelly_fraction=kelly,
            var_95=var_95,
            var_99=var_99
        )
    
    # ==================== STOP LOSS MANAGEMENT ====================
    
    def calculate_atr_stop(self, current_price: float, atr: float, 
                           direction: str = 'long') -> float:
        """Calculate ATR-based stop loss"""
        sl_distance = atr * self.config['sl_atr_multiplier']
        
        if direction == 'long':
            return current_price - sl_distance
        else:
            return current_price + sl_distance
    
    def calculate_volatility_stop(self, prices: pd.Series, current_price: float,
                                   direction: str = 'long', lookback: int = 20) -> float:
        """
        Calculate volatility-based stop loss
        
        Uses recent price range to set stop
        """
        if len(prices) < lookback:
            lookback = len(prices)
        
        recent_prices = prices.tail(lookback)
        volatility = recent_prices.std()
        
        multiplier = 2.5  # 2.5 standard deviations
        
        if direction == 'long':
            return current_price - (volatility * multiplier)
        else:
            return current_price + (volatility * multiplier)
    
    def calculate_trailing_stop(self, entry_price: float, current_price: float,
                                 highest_price: float, atr: float,
                                 direction: str = 'long') -> float:
        """
        Calculate trailing stop loss
        
        Moves stop in direction of profit, never against
        """
        trail_distance = atr * self.config['trailing_atr_multiplier']
        
        if direction == 'long':
            # Stop trails below highest price
            trailing_stop = highest_price - trail_distance
            # Never move stop below initial stop
            initial_stop = entry_price - (atr * self.config['sl_atr_multiplier'])
            return max(trailing_stop, initial_stop)
        else:
            # Stop trails above lowest price
            lowest_price = min(entry_price, current_price)
            trailing_stop = lowest_price + trail_distance
            initial_stop = entry_price + (atr * self.config['sl_atr_multiplier'])
            return min(trailing_stop, initial_stop)
    
    # ==================== PORTFOLIO RISK ====================
    
    def calculate_portfolio_var(self, positions: List[Dict], 
                                 correlation_matrix: np.ndarray = None) -> float:
        """
        Calculate portfolio VAR considering correlations
        
        If no correlation matrix provided, assumes 0.5 correlation
        """
        if not positions:
            return 0.0
        
        n = len(positions)
        
        # Get position values and individual VARs
        values = np.array([p.get('value', 0) for p in positions])
        individual_vars = np.array([p.get('var', 0.02) for p in positions])
        
        if correlation_matrix is None:
            # Default correlation matrix (0.5 between all assets)
            correlation_matrix = np.ones((n, n)) * 0.5
            np.fill_diagonal(correlation_matrix, 1.0)
        
        # Calculate portfolio VAR using variance-covariance method
        weights = values / np.sum(values) if np.sum(values) > 0 else np.ones(n) / n
        
        # Covariance matrix from correlations and individual VARs
        vol = individual_vars
        cov_matrix = np.outer(vol, vol) * correlation_matrix
        
        # Portfolio variance
        portfolio_variance = weights @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # 95% VAR
        portfolio_var = portfolio_vol * 1.645
        
        return portfolio_var
    
    def check_correlation_limit(self, new_symbol: str, 
                                 correlations: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if adding a position would exceed correlation limits
        """
        if not self.positions:
            return True, "First position"
        
        # Calculate exposure to correlated assets
        correlated_exposure = 0.0
        
        for pos in self.positions:
            symbol = pos.get('symbol', '')
            value = pos.get('value', 0)
            
            # Get correlation with new symbol
            corr = correlations.get(f"{new_symbol}_{symbol}", 0.5)
            
            if corr > 0.7:  # High correlation
                correlated_exposure += value
        
        exposure_pct = correlated_exposure / self.current_capital * 100
        
        if exposure_pct >= self.config['max_correlated_exposure_pct']:
            return False, f"Correlated exposure too high: {exposure_pct:.1f}%"
        
        return True, "Correlation check passed"
    
    # ==================== RISK REPORT ====================
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            
            'capital': {
                'current': self.current_capital,
                'peak': self.peak_capital,
                'drawdown_pct': self.current_drawdown,
                'drawdown_amount': self.peak_capital - self.current_capital
            },
            
            'positions': {
                'count': len(self.positions),
                'total_value': sum(p.get('value', 0) for p in self.positions),
                'total_risk': sum(p.get('risk_amount', 0) for p in self.positions)
            },
            
            'limits': {
                'max_drawdown': self.config['max_drawdown_pct'],
                'max_risk_per_trade': self.config['max_risk_per_trade_pct'],
                'max_portfolio_risk': self.config['max_portfolio_risk_pct'],
                'trading_allowed': self.is_trading_allowed()[0]
            },
            
            'kelly': {
                'estimated': self.estimate_kelly_from_history(),
                'fraction_used': self.config['kelly_fraction']
            }
        }
        
        # Add trade history stats if available
        if self.trade_history:
            pnls = [t['pnl'] for t in self.trade_history]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            report['performance'] = {
                'total_trades': len(self.trade_history),
                'win_rate': len(wins) / len(pnls) * 100 if pnls else 0,
                'profit_factor': sum(wins) / abs(sum(losses)) if losses else 0,
                'avg_win': np.mean(wins) if wins else 0,
                'avg_loss': abs(np.mean(losses)) if losses else 0,
                'largest_win': max(wins) if wins else 0,
                'largest_loss': min(losses) if losses else 0
            }
        
        return report
    
    def print_risk_report(self):
        """Print formatted risk report"""
        report = self.generate_risk_report()
        
        print("\n" + "=" * 60)
        print("RISK MANAGEMENT REPORT")
        print("=" * 60)
        
        print(f"\n{'CAPITAL STATUS':^60}")
        print("-" * 60)
        print(f"Current Capital:  ${report['capital']['current']:,.2f}")
        print(f"Peak Capital:     ${report['capital']['peak']:,.2f}")
        print(f"Current Drawdown: {report['capital']['drawdown_pct']:.2f}%")
        
        print(f"\n{'POSITION RISK':^60}")
        print("-" * 60)
        print(f"Open Positions:   {report['positions']['count']}")
        print(f"Total Exposure:   ${report['positions']['total_value']:,.2f}")
        print(f"Total Risk:       ${report['positions']['total_risk']:,.2f}")
        
        print(f"\n{'RISK LIMITS':^60}")
        print("-" * 60)
        print(f"Max Drawdown:     {report['limits']['max_drawdown']}%")
        print(f"Max Risk/Trade:   {report['limits']['max_risk_per_trade']}%")
        print(f"Trading Allowed:  {'YES' if report['limits']['trading_allowed'] else 'NO'}")
        
        if 'performance' in report:
            print(f"\n{'PERFORMANCE':^60}")
            print("-" * 60)
            print(f"Total Trades:     {report['performance']['total_trades']}")
            print(f"Win Rate:         {report['performance']['win_rate']:.1f}%")
            print(f"Profit Factor:    {report['performance']['profit_factor']:.2f}")
        
        print("\n" + "=" * 60)


def demo():
    """Demonstrate advanced risk management"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         ADVANCED RISK MANAGEMENT DEMO                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize
    rm = AdvancedRiskManager({
        'initial_capital': 100000,
        'max_risk_per_trade_pct': 2.0,
        'max_drawdown_pct': 15.0
    })
    
    # Example 1: Calculate position size
    print("\n1. POSITION SIZING EXAMPLE")
    print("-" * 40)
    
    params = rm.calculate_position_size(
        entry_price=100.0,
        stop_loss=95.0,
        win_rate=0.55,
        avg_win_pct=3.0,
        avg_loss_pct=2.0,
        volatility=0.02
    )
    
    print(f"Entry Price:        $100.00")
    print(f"Stop Loss:          ${params.stop_loss:.2f}")
    print(f"Take Profit:        ${params.take_profit:.2f}")
    print(f"Position Size:      {params.position_size_shares} shares")
    print(f"Position %:         {params.position_size_pct:.1f}%")
    print(f"Max Loss:           ${params.max_loss_amount:.2f}")
    print(f"Risk/Reward:        {params.risk_reward_ratio:.1f}")
    print(f"Kelly Fraction:     {params.kelly_fraction*100:.1f}%")
    print(f"VAR (95%):          ${params.var_95:.2f}")
    
    # Example 2: Kelly Criterion
    print("\n2. KELLY CRITERION")
    print("-" * 40)
    
    for win_rate in [0.50, 0.55, 0.60]:
        kelly = rm.calculate_kelly_criterion(win_rate, avg_win=2.0, avg_loss=1.0)
        print(f"Win Rate {win_rate*100:.0f}%: Kelly = {kelly*100:.1f}%")
    
    # Example 3: VAR Calculation
    print("\n3. VALUE AT RISK")
    print("-" * 40)
    
    # Generate sample returns
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # 1 year daily returns
    
    var_95 = rm.calculate_var(returns, 0.95)
    var_99 = rm.calculate_var(returns, 0.99)
    cvar = rm.calculate_conditional_var(returns, 0.95)
    
    print(f"95% VAR:    {var_95*100:.2f}% (1-day)")
    print(f"99% VAR:    {var_99*100:.2f}% (1-day)")
    print(f"95% CVaR:   {cvar*100:.2f}% (Expected Shortfall)")
    
    # Example 4: Drawdown Management
    print("\n4. DRAWDOWN MANAGEMENT")
    print("-" * 40)
    
    for dd in [0, 5, 10, 12, 15]:
        rm.current_drawdown = dd
        multiplier = rm.get_drawdown_adjusted_size(1.0)
        print(f"Drawdown {dd:2d}%: Size multiplier = {multiplier:.2f}x")
    
    # Example 5: Risk Report
    print("\n5. RISK REPORT")
    rm.current_drawdown = 8.0
    rm.trade_history = [
        {'pnl': 500}, {'pnl': -200}, {'pnl': 300},
        {'pnl': -150}, {'pnl': 400}, {'pnl': 200},
        {'pnl': -100}, {'pnl': 350}, {'pnl': -250}, {'pnl': 450}
    ]
    rm.print_risk_report()


if __name__ == "__main__":
    demo()
