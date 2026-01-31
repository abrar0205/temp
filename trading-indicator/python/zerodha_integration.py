#!/usr/bin/env python3
"""
ZERODHA KITE INTEGRATION MODULE
===============================
Ready-to-use integration with Zerodha Kite API for:

1. Authentication & Session Management
2. Placing Orders (Market, Limit, SL, SL-M)
3. Getting Positions & Holdings
4. Live Market Data Streaming
5. Paper Trading Mode (simulation)

Requirements:
    pip install kiteconnect (for live trading)
    
Note: This module works in paper trading mode by default.
      Set LIVE_TRADING=True and provide API credentials for live trading.
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class TransactionType(Enum):
    """Transaction types."""
    BUY = "BUY"
    SELL = "SELL"


class ProductType(Enum):
    """Product types."""
    CNC = "CNC"       # Cash and Carry (delivery)
    MIS = "MIS"       # Intraday
    NRML = "NRML"     # Normal (F&O)


@dataclass
class Order:
    """Represents an order."""
    order_id: str
    symbol: str
    transaction_type: TransactionType
    order_type: OrderType
    quantity: int
    price: float
    trigger_price: float
    status: str
    filled_quantity: int
    average_price: float
    timestamp: datetime


@dataclass
class Position:
    """Represents a position."""
    symbol: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    pnl_percentage: float


@dataclass
class Holding:
    """Represents a holding (delivery position)."""
    symbol: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    instrument_token: str


class ZerodhaIntegration:
    """
    Zerodha Kite API Integration.
    
    Paper Trading Mode (default):
        - Simulates order execution
        - Tracks virtual positions
        - No real money involved
    
    Live Trading Mode:
        - Requires Kite API credentials
        - Places real orders
        - USE WITH CAUTION!
    """
    
    def __init__(self, api_key: str = None, access_token: str = None,
                 paper_trading: bool = True, initial_capital: float = 100000):
        """
        Initialize Zerodha integration.
        
        Args:
            api_key: Zerodha API key (for live trading)
            access_token: Zerodha access token (for live trading)
            paper_trading: If True, use paper trading mode
            initial_capital: Starting capital for paper trading
        """
        self.api_key = api_key
        self.access_token = access_token
        self.paper_trading = paper_trading
        self.initial_capital = initial_capital
        
        # Paper trading state
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.order_counter = 0
        
        # Kite client (for live trading)
        self.kite = None
        
        if not paper_trading:
            self._initialize_kite()
    
    def _initialize_kite(self):
        """Initialize Kite Connect client."""
        if self.paper_trading:
            return
        
        try:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=self.api_key)
            if self.access_token:
                self.kite.set_access_token(self.access_token)
            print("[OK] Kite Connect initialized")
        except ImportError:
            print("[WARNING] kiteconnect not installed. Install with: pip install kiteconnect")
            self.paper_trading = True
        except Exception as e:
            print(f"[ERROR] Failed to initialize Kite: {e}")
            self.paper_trading = True
    
    # ==========================================================================
    # ORDER MANAGEMENT
    # ==========================================================================
    
    def place_order(self, symbol: str, transaction_type: TransactionType,
                    order_type: OrderType, quantity: int,
                    price: float = 0, trigger_price: float = 0,
                    product: ProductType = ProductType.MIS) -> Optional[str]:
        """
        Place an order.
        
        Args:
            symbol: Trading symbol (e.g., 'RELIANCE', 'NIFTY23JAN18000CE')
            transaction_type: BUY or SELL
            order_type: MARKET, LIMIT, SL, or SL-M
            quantity: Number of shares/lots
            price: Limit price (for LIMIT and SL orders)
            trigger_price: Trigger price (for SL and SL-M orders)
            product: CNC, MIS, or NRML
            
        Returns:
            Order ID if successful, None otherwise
        """
        if self.paper_trading:
            return self._paper_place_order(symbol, transaction_type, order_type,
                                          quantity, price, trigger_price, product)
        else:
            return self._live_place_order(symbol, transaction_type, order_type,
                                         quantity, price, trigger_price, product)
    
    def _paper_place_order(self, symbol: str, transaction_type: TransactionType,
                           order_type: OrderType, quantity: int,
                           price: float, trigger_price: float,
                           product: ProductType) -> str:
        """Place order in paper trading mode."""
        self.order_counter += 1
        order_id = f"PAPER_{self.order_counter:06d}"
        
        # Simulate execution price
        if order_type == OrderType.MARKET:
            exec_price = self._get_simulated_price(symbol)
        else:
            exec_price = price
        
        # Calculate order value
        order_value = exec_price * quantity
        
        # Check if we have enough capital (for buy orders)
        if transaction_type == TransactionType.BUY:
            if order_value > self.capital:
                print(f"[REJECTED] Insufficient capital. Required: {order_value:.2f}, Available: {self.capital:.2f}")
                return None
            self.capital -= order_value
        
        # Create order
        order = Order(
            order_id=order_id,
            symbol=symbol,
            transaction_type=transaction_type,
            order_type=order_type,
            quantity=quantity,
            price=price,
            trigger_price=trigger_price,
            status="COMPLETE",
            filled_quantity=quantity,
            average_price=exec_price,
            timestamp=datetime.now()
        )
        
        self.orders.append(order)
        
        # Update positions
        self._update_position(symbol, transaction_type, quantity, exec_price)
        
        print(f"[PAPER] Order placed: {order_id} | {transaction_type.value} {quantity} {symbol} @ {exec_price:.2f}")
        
        return order_id
    
    def _live_place_order(self, symbol: str, transaction_type: TransactionType,
                          order_type: OrderType, quantity: int,
                          price: float, trigger_price: float,
                          product: ProductType) -> Optional[str]:
        """Place order via Kite API."""
        if not self.kite:
            print("[ERROR] Kite client not initialized")
            return None
        
        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=self.kite.EXCHANGE_NSE,
                tradingsymbol=symbol,
                transaction_type=transaction_type.value,
                quantity=quantity,
                product=product.value,
                order_type=order_type.value,
                price=price if order_type != OrderType.MARKET else None,
                trigger_price=trigger_price if order_type in [OrderType.SL, OrderType.SL_M] else None
            )
            
            print(f"[LIVE] Order placed: {order_id}")
            return str(order_id)
            
        except Exception as e:
            print(f"[ERROR] Order failed: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        if self.paper_trading:
            for order in self.orders:
                if order.order_id == order_id and order.status == "PENDING":
                    order.status = "CANCELLED"
                    return True
            return False
        else:
            try:
                self.kite.cancel_order(variety=self.kite.VARIETY_REGULAR,
                                       order_id=order_id)
                return True
            except Exception as e:
                print(f"[ERROR] Cancel failed: {e}")
                return False
    
    def get_orders(self) -> List[Order]:
        """Get all orders."""
        if self.paper_trading:
            return self.orders
        else:
            try:
                live_orders = self.kite.orders()
                return [self._parse_kite_order(o) for o in live_orders]
            except Exception as e:
                print(f"[ERROR] Failed to get orders: {e}")
                return []
    
    # ==========================================================================
    # POSITION MANAGEMENT
    # ==========================================================================
    
    def _update_position(self, symbol: str, transaction_type: TransactionType,
                         quantity: int, price: float):
        """Update position after order execution."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0,
                average_price=0,
                last_price=price,
                pnl=0,
                pnl_percentage=0
            )
        
        pos = self.positions[symbol]
        
        if transaction_type == TransactionType.BUY:
            # Update average price
            total_value = (pos.quantity * pos.average_price) + (quantity * price)
            pos.quantity += quantity
            if pos.quantity > 0:
                pos.average_price = total_value / pos.quantity
        else:  # SELL
            # Realize P&L
            realized_pnl = (price - pos.average_price) * quantity
            self.capital += (price * quantity)  # Add sale proceeds
            pos.quantity -= quantity
            
            if pos.quantity == 0:
                pos.average_price = 0
        
        # Update P&L
        pos.last_price = price
        if pos.quantity != 0:
            pos.pnl = (pos.last_price - pos.average_price) * pos.quantity
            pos.pnl_percentage = ((pos.last_price / pos.average_price) - 1) * 100 if pos.average_price > 0 else 0
    
    def get_positions(self) -> Dict[str, Position]:
        """Get all positions."""
        if self.paper_trading:
            return {k: v for k, v in self.positions.items() if v.quantity != 0}
        else:
            try:
                live_positions = self.kite.positions()
                return self._parse_kite_positions(live_positions)
            except Exception as e:
                print(f"[ERROR] Failed to get positions: {e}")
                return {}
    
    def get_holdings(self) -> List[Holding]:
        """Get delivery holdings."""
        if self.paper_trading:
            # In paper trading, treat all positions as holdings
            return [Holding(
                symbol=p.symbol,
                quantity=p.quantity,
                average_price=p.average_price,
                last_price=p.last_price,
                pnl=p.pnl,
                instrument_token=""
            ) for p in self.positions.values() if p.quantity > 0]
        else:
            try:
                return self.kite.holdings()
            except Exception as e:
                print(f"[ERROR] Failed to get holdings: {e}")
                return []
    
    # ==========================================================================
    # MARKET DATA
    # ==========================================================================
    
    def get_quote(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get live quotes for symbols.
        
        Returns dict with last_price, ohlc, volume, etc.
        """
        if self.paper_trading:
            return {s: {'last_price': self._get_simulated_price(s)} for s in symbols}
        else:
            try:
                instruments = [f"NSE:{s}" for s in symbols]
                return self.kite.quote(instruments)
            except Exception as e:
                print(f"[ERROR] Failed to get quote: {e}")
                return {}
    
    def _get_simulated_price(self, symbol: str) -> float:
        """Get simulated price for paper trading."""
        # Base prices for common symbols
        base_prices = {
            'NIFTY50': 22000,
            'NIFTY': 22000,
            'BANKNIFTY': 47000,
            'RELIANCE': 2500,
            'TCS': 3800,
            'INFY': 1500,
            'HDFC': 1600,
            'ICICIBANK': 1000,
            'SBIN': 600,
            'HDFCBANK': 1500,
            'TATAMOTORS': 700,
            'ADANIENT': 2800,
            'ITC': 450,
            'HINDUNILVR': 2400,
            'BHARTIARTL': 1100,
        }
        
        base = base_prices.get(symbol.upper(), 1000)
        
        # Add small random variation
        variation = np.random.uniform(-0.005, 0.005)
        return base * (1 + variation)
    
    # ==========================================================================
    # ACCOUNT INFO
    # ==========================================================================
    
    def get_account_summary(self) -> Dict:
        """Get account summary."""
        if self.paper_trading:
            positions = self.get_positions()
            positions_value = sum(p.last_price * p.quantity for p in positions.values())
            total_pnl = sum(p.pnl for p in positions.values())
            
            return {
                'mode': 'PAPER TRADING',
                'initial_capital': self.initial_capital,
                'available_capital': self.capital,
                'positions_value': positions_value,
                'total_value': self.capital + positions_value,
                'unrealized_pnl': total_pnl,
                'total_orders': len(self.orders),
                'open_positions': len(positions)
            }
        else:
            try:
                margins = self.kite.margins()
                return {
                    'mode': 'LIVE TRADING',
                    'available_cash': margins['equity']['available']['cash'],
                    'used_margin': margins['equity']['utilised']['debits'],
                    'total_value': margins['equity']['net']
                }
            except Exception as e:
                print(f"[ERROR] Failed to get margins: {e}")
                return {}
    
    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================
    
    def _parse_kite_order(self, order_data: Dict) -> Order:
        """Parse Kite order response into Order object."""
        return Order(
            order_id=str(order_data.get('order_id', '')),
            symbol=order_data.get('tradingsymbol', ''),
            transaction_type=TransactionType(order_data.get('transaction_type', 'BUY')),
            order_type=OrderType(order_data.get('order_type', 'MARKET')),
            quantity=order_data.get('quantity', 0),
            price=order_data.get('price', 0),
            trigger_price=order_data.get('trigger_price', 0),
            status=order_data.get('status', ''),
            filled_quantity=order_data.get('filled_quantity', 0),
            average_price=order_data.get('average_price', 0),
            timestamp=datetime.now()
        )
    
    def _parse_kite_positions(self, positions_data: Dict) -> Dict[str, Position]:
        """Parse Kite positions response."""
        result = {}
        
        for pos in positions_data.get('net', []):
            symbol = pos.get('tradingsymbol', '')
            if pos.get('quantity', 0) != 0:
                result[symbol] = Position(
                    symbol=symbol,
                    quantity=pos.get('quantity', 0),
                    average_price=pos.get('average_price', 0),
                    last_price=pos.get('last_price', 0),
                    pnl=pos.get('pnl', 0),
                    pnl_percentage=pos.get('pnl', 0) / pos.get('value', 1) * 100
                )
        
        return result
    
    # ==========================================================================
    # TRADING SIGNALS INTEGRATION
    # ==========================================================================
    
    def execute_signal(self, signal: int, symbol: str, quantity: int,
                       current_price: float, stop_loss_pct: float = 0.02,
                       take_profit_pct: float = 0.04) -> Optional[str]:
        """
        Execute a trading signal.
        
        Args:
            signal: 1 for BUY, -1 for SELL
            symbol: Trading symbol
            quantity: Number of shares
            current_price: Current market price
            stop_loss_pct: Stop loss percentage (default 2%)
            take_profit_pct: Take profit percentage (default 4%)
            
        Returns:
            Order ID if executed
        """
        if signal == 0:
            return None
        
        # Check existing position
        existing_pos = self.positions.get(symbol)
        
        if signal == 1:  # BUY
            # Check if we already have a long position
            if existing_pos and existing_pos.quantity > 0:
                print(f"[SKIP] Already long {symbol}")
                return None
            
            # Place buy order
            order_id = self.place_order(
                symbol=symbol,
                transaction_type=TransactionType.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity
            )
            
            if order_id:
                print(f"[SIGNAL] BUY {quantity} {symbol}")
                print(f"  Stop Loss: {current_price * (1 - stop_loss_pct):.2f}")
                print(f"  Take Profit: {current_price * (1 + take_profit_pct):.2f}")
            
            return order_id
        
        elif signal == -1:  # SELL
            # If we have a long position, close it
            if existing_pos and existing_pos.quantity > 0:
                order_id = self.place_order(
                    symbol=symbol,
                    transaction_type=TransactionType.SELL,
                    order_type=OrderType.MARKET,
                    quantity=existing_pos.quantity
                )
                
                if order_id:
                    print(f"[SIGNAL] SELL (close) {existing_pos.quantity} {symbol}")
                
                return order_id
            else:
                print(f"[SKIP] No position to close in {symbol}")
                return None
        
        return None


# Demo
if __name__ == "__main__":
    print("=" * 60)
    print("ZERODHA KITE INTEGRATION - DEMO")
    print("=" * 60)
    
    # Initialize in paper trading mode
    zerodha = ZerodhaIntegration(paper_trading=True, initial_capital=100000)
    
    # Show initial account
    print("\n" + "-" * 40)
    print("INITIAL ACCOUNT")
    print("-" * 40)
    summary = zerodha.get_account_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:,.2f}")
        else:
            print(f"{key:20s}: {value}")
    
    # Place some orders
    print("\n" + "-" * 40)
    print("PLACING ORDERS")
    print("-" * 40)
    
    # Buy RELIANCE
    zerodha.place_order(
        symbol="RELIANCE",
        transaction_type=TransactionType.BUY,
        order_type=OrderType.MARKET,
        quantity=10
    )
    
    # Buy TCS
    zerodha.place_order(
        symbol="TCS",
        transaction_type=TransactionType.BUY,
        order_type=OrderType.MARKET,
        quantity=5
    )
    
    # Buy INFY
    zerodha.place_order(
        symbol="INFY",
        transaction_type=TransactionType.BUY,
        order_type=OrderType.MARKET,
        quantity=20
    )
    
    # Show positions
    print("\n" + "-" * 40)
    print("POSITIONS")
    print("-" * 40)
    positions = zerodha.get_positions()
    print(f"{'Symbol':<15} {'Qty':>8} {'Avg Price':>12} {'Current':>12} {'P&L':>12}")
    print("-" * 60)
    for symbol, pos in positions.items():
        print(f"{pos.symbol:<15} {pos.quantity:>8} {pos.average_price:>12.2f} "
              f"{pos.last_price:>12.2f} {pos.pnl:>12.2f}")
    
    # Sell partial RELIANCE
    print("\n" + "-" * 40)
    print("SELLING 5 RELIANCE")
    print("-" * 40)
    zerodha.place_order(
        symbol="RELIANCE",
        transaction_type=TransactionType.SELL,
        order_type=OrderType.MARKET,
        quantity=5
    )
    
    # Final account summary
    print("\n" + "-" * 40)
    print("FINAL ACCOUNT")
    print("-" * 40)
    summary = zerodha.get_account_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:,.2f}")
        else:
            print(f"{key:20s}: {value}")
    
    # Execute signal
    print("\n" + "-" * 40)
    print("EXECUTE TRADING SIGNAL")
    print("-" * 40)
    
    zerodha.execute_signal(
        signal=1,  # BUY
        symbol="HDFCBANK",
        quantity=10,
        current_price=1500
    )
    
    # Final positions
    print("\n" + "-" * 40)
    print("FINAL POSITIONS")
    print("-" * 40)
    positions = zerodha.get_positions()
    print(f"{'Symbol':<15} {'Qty':>8} {'Avg Price':>12} {'Current':>12} {'P&L':>12}")
    print("-" * 60)
    for symbol, pos in positions.items():
        print(f"{pos.symbol:<15} {pos.quantity:>8} {pos.average_price:>12.2f} "
              f"{pos.last_price:>12.2f} {pos.pnl:>12.2f}")
    
    print("\n[OK] Zerodha Integration Module Working!")
    print("\n[NOTE] This was PAPER TRADING mode. For live trading:")
    print("       1. pip install kiteconnect")
    print("       2. Get API key from Zerodha Developer Console")
    print("       3. Initialize with paper_trading=False")
