"""
Streamlit Web Dashboard for Trading Indicator
Interactive visualization and analysis tool

Features:
- Real-time price charts
- Technical indicator overlays
- ML signal visualization
- Multi-timeframe analysis display
- Regime detection status
- Performance tracking
- Alert configuration

Run with: streamlit run streamlit_dashboard.py
"""

import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Check for required packages
STREAMLIT_AVAILABLE = False
PLOTLY_AVAILABLE = False
PANDAS_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    pass

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    pass

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pass

# Local imports (will work once Streamlit runs)
try:
    from realtime_data import DataManager
    from regime_detection import RegimeDetector, MarketRegime
    from multi_timeframe import MultiTimeframeAnalyzer, Timeframe
    from ml_signal_filter import MLSignalFilter
except ImportError:
    DataManager = None
    RegimeDetector = None
    MultiTimeframeAnalyzer = None
    MLSignalFilter = None


def check_dependencies():
    """Check if all required packages are installed."""
    missing = []
    
    if not STREAMLIT_AVAILABLE:
        missing.append('streamlit')
    if not PLOTLY_AVAILABLE:
        missing.append('plotly')
    if not PANDAS_AVAILABLE:
        missing.append('pandas')
    
    return missing


def calculate_indicators(data: List[Dict]) -> Dict:
    """Calculate technical indicators for visualization."""
    closes = [d['close'] for d in data]
    highs = [d['high'] for d in data]
    lows = [d['low'] for d in data]
    
    # EMA calculations
    def calc_ema(prices, period):
        if len(prices) < period:
            return [None] * len(prices)
        
        ema = [sum(prices[:period]) / period]
        mult = 2 / (period + 1)
        
        for price in prices[period:]:
            ema.append((price * mult) + (ema[-1] * (1 - mult)))
        
        return [None] * (period - 1) + ema
    
    # RSI calculation
    def calc_rsi(prices, period=14):
        if len(prices) < period + 1:
            return [50] * len(prices)
        
        rsi = [None] * period
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(c, 0) for c in changes]
        losses = [abs(min(c, 0)) for c in changes]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        if avg_loss == 0:
            rsi.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi.append(100 - (100 / (1 + rs)))
        
        for i in range(period, len(changes)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))
        
        return rsi
    
    # SuperTrend calculation
    def calc_supertrend(highs, lows, closes, period=10, mult=3.0):
        if len(closes) < period + 1:
            return [None] * len(closes), [0] * len(closes)
        
        # ATR
        tr = [highs[0] - lows[0]]
        for i in range(1, len(closes)):
            tr.append(max(highs[i] - lows[i], 
                         abs(highs[i] - closes[i-1]), 
                         abs(lows[i] - closes[i-1])))
        
        atr = []
        for i in range(len(tr)):
            if i < period - 1:
                atr.append(None)
            else:
                atr.append(sum(tr[i-period+1:i+1]) / period)
        
        supertrend = [None] * len(closes)
        direction = [0] * len(closes)
        
        for i in range(period, len(closes)):
            if atr[i] is None:
                continue
            
            hl2 = (highs[i] + lows[i]) / 2
            upper = hl2 + (mult * atr[i])
            lower = hl2 - (mult * atr[i])
            
            if i == period:
                supertrend[i] = lower
                direction[i] = 1
            else:
                if closes[i-1] > supertrend[i-1]:
                    direction[i] = 1
                    supertrend[i] = max(lower, supertrend[i-1]) if direction[i-1] == 1 else lower
                else:
                    direction[i] = -1
                    supertrend[i] = min(upper, supertrend[i-1]) if direction[i-1] == -1 else upper
        
        return supertrend, direction
    
    ema9 = calc_ema(closes, 9)
    ema21 = calc_ema(closes, 21)
    ema50 = calc_ema(closes, 50)
    ema200 = calc_ema(closes, 200)
    rsi = calc_rsi(closes, 14)
    supertrend, st_direction = calc_supertrend(highs, lows, closes)
    
    return {
        'ema9': ema9,
        'ema21': ema21,
        'ema50': ema50,
        'ema200': ema200,
        'rsi': rsi,
        'supertrend': supertrend,
        'st_direction': st_direction
    }


def create_candlestick_chart(data: List[Dict], indicators: Dict, symbol: str) -> go.Figure:
    """Create interactive candlestick chart with indicators."""
    if not PLOTLY_AVAILABLE:
        return None
    
    dates = [d['date'] for d in data]
    opens = [d['open'] for d in data]
    highs = [d['high'] for d in data]
    lows = [d['low'] for d in data]
    closes = [d['close'] for d in data]
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[f'{symbol} Price', 'RSI', 'Volume']
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name='Price',
            increasing_line_color='green',
            decreasing_line_color='red'
        ),
        row=1, col=1
    )
    
    # EMAs
    colors = {'ema9': 'blue', 'ema21': 'orange', 'ema50': 'purple', 'ema200': 'red'}
    for ema_name, color in colors.items():
        if ema_name in indicators:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=indicators[ema_name],
                    name=ema_name.upper(),
                    line=dict(color=color, width=1),
                    opacity=0.7
                ),
                row=1, col=1
            )
    
    # SuperTrend
    if 'supertrend' in indicators:
        st_colors = ['green' if d == 1 else 'red' for d in indicators['st_direction']]
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=indicators['supertrend'],
                name='SuperTrend',
                mode='lines',
                line=dict(width=2),
                marker=dict(color=st_colors)
            ),
            row=1, col=1
        )
    
    # RSI
    if 'rsi' in indicators:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=indicators['rsi'],
                name='RSI',
                line=dict(color='purple', width=1)
            ),
            row=2, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # Volume
    volumes = [d['volume'] for d in data]
    colors = ['green' if closes[i] >= opens[i] else 'red' for i in range(len(data))]
    
    fig.add_trace(
        go.Bar(
            x=dates,
            y=volumes,
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ),
        row=3, col=1
    )
    
    # Layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        template='plotly_dark'
    )
    
    return fig


def main_dashboard():
    """Main Streamlit dashboard."""
    if not STREAMLIT_AVAILABLE:
        print("Streamlit not installed. Install with: pip install streamlit")
        return
    
    st.set_page_config(
        page_title="Trading Indicator Dashboard",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📈 Universal Trading Indicator Dashboard")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Symbol selection
    symbol_categories = {
        'Indian Indices': ['NIFTY50', 'BANKNIFTY', 'SENSEX'],
        'Indian Stocks': ['TCS', 'RELIANCE', 'INFY', 'HDFC'],
        'US Stocks': ['AAPL', 'GOOGL', 'MSFT', 'TSLA'],
        'Global Indices': ['SPX', 'NASDAQ', 'DAX'],
        'Forex': ['USDINR', 'EURUSD'],
        'Crypto': ['BTC', 'ETH'],
        'Commodities': ['GOLD', 'SILVER', 'CRUDE']
    }
    
    category = st.sidebar.selectbox("Category", list(symbol_categories.keys()))
    symbol = st.sidebar.selectbox("Symbol", symbol_categories[category])
    
    timeframe = st.sidebar.selectbox(
        "Timeframe",
        ['1d', '1h', '15m', '5m'],
        index=0
    )
    
    period = st.sidebar.selectbox(
        "Historical Period",
        ['1mo', '3mo', '6mo', '1y', '2y'],
        index=2
    )
    
    # Analysis options
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Analysis Options")
    
    show_emas = st.sidebar.checkbox("Show EMAs", value=True)
    show_supertrend = st.sidebar.checkbox("Show SuperTrend", value=True)
    show_rsi = st.sidebar.checkbox("Show RSI", value=True)
    run_ml = st.sidebar.checkbox("ML Signal Filter", value=False)
    run_mtf = st.sidebar.checkbox("Multi-Timeframe Analysis", value=False)
    run_regime = st.sidebar.checkbox("Regime Detection", value=True)
    
    # Fetch data button
    if st.sidebar.button("🔄 Fetch Data", type="primary"):
        st.session_state['fetch_data'] = True
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header(f"📊 {symbol} Analysis")
        
        # Fetch and display data
        if DataManager is not None:
            with st.spinner(f"Fetching data for {symbol}..."):
                try:
                    manager = DataManager()
                    data = manager.get_data(symbol, timeframe, period)
                    
                    if data and len(data) > 0:
                        st.success(f"Loaded {len(data)} bars of data")
                        
                        # Calculate indicators
                        indicators = calculate_indicators(data)
                        
                        # Create chart
                        if PLOTLY_AVAILABLE:
                            fig = create_candlestick_chart(data, indicators, symbol)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Plotly not installed. Install with: pip install plotly")
                            
                            # Show basic data table
                            if PANDAS_AVAILABLE:
                                df = pd.DataFrame(data[-20:])
                                st.dataframe(df)
                        
                        # Store data in session state for other analyses
                        st.session_state['data'] = data
                        st.session_state['indicators'] = indicators
                    else:
                        st.error("Could not fetch data. Check your internet connection.")
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")
        else:
            st.warning("Data module not available. Please check imports.")
            
            # Generate sample data for demo
            import numpy as np
            np.random.seed(42)
            
            data = []
            price = 100
            
            for i in range(200):
                change = np.random.randn() * 0.5
                open_price = price
                close_price = price * (1 + change / 100)
                
                data.append({
                    'date': (datetime.now() - timedelta(days=200-i)).strftime('%Y-%m-%d'),
                    'open': open_price,
                    'high': max(open_price, close_price) * 1.002,
                    'low': min(open_price, close_price) * 0.998,
                    'close': close_price,
                    'volume': int(np.random.uniform(100000, 500000))
                })
                
                price = close_price
            
            indicators = calculate_indicators(data)
            
            if PLOTLY_AVAILABLE:
                fig = create_candlestick_chart(data, indicators, symbol)
                st.plotly_chart(fig, use_container_width=True)
            
            st.session_state['data'] = data
            st.session_state['indicators'] = indicators
    
    with col2:
        # Current price info
        st.header("📌 Current Status")
        
        if 'data' in st.session_state and len(st.session_state['data']) > 0:
            data = st.session_state['data']
            latest = data[-1]
            prev = data[-2] if len(data) > 1 else latest
            
            change = latest['close'] - prev['close']
            change_pct = (change / prev['close']) * 100
            
            st.metric(
                label="Price",
                value=f"${latest['close']:.2f}",
                delta=f"{change_pct:.2f}%"
            )
            
            st.metric(label="High", value=f"${latest['high']:.2f}")
            st.metric(label="Low", value=f"${latest['low']:.2f}")
            st.metric(label="Volume", value=f"{latest['volume']:,}")
            
            # RSI Status
            if 'indicators' in st.session_state:
                rsi = st.session_state['indicators']['rsi'][-1]
                if rsi:
                    rsi_status = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                    st.metric(label="RSI", value=f"{rsi:.1f}", delta=rsi_status)
    
    # Additional analysis sections
    st.markdown("---")
    
    # Regime Detection
    if run_regime:
        st.header("🎯 Market Regime Detection")
        
        if 'data' in st.session_state and RegimeDetector is not None:
            data = st.session_state['data']
            
            detector = RegimeDetector()
            analysis = detector.analyze(data)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                regime_colors = {
                    'strong_uptrend': '🟢',
                    'weak_uptrend': '🟡',
                    'strong_downtrend': '🔴',
                    'weak_downtrend': '🟠',
                    'ranging': '⚪',
                    'high_volatility': '⚡',
                    'low_volatility': '💤',
                    'breakout_potential': '💥'
                }
                emoji = regime_colors.get(analysis.regime.value, '❓')
                st.metric("Regime", f"{emoji} {analysis.regime.value.replace('_', ' ').title()}")
            
            with col2:
                st.metric("ADX", f"{analysis.adx:.1f}")
                st.metric("Trend", f"{analysis.trend_direction.title()} ({analysis.trend_strength})")
            
            with col3:
                st.metric("Confidence", f"{analysis.confidence:.0f}%")
                st.metric("Position Size", f"{analysis.position_size_multiplier:.1f}x")
            
            st.info(f"📋 **Recommendation:** {analysis.strategy_recommendation}")
        else:
            st.warning("Regime detection requires data and RegimeDetector module.")
    
    # Multi-Timeframe Analysis
    if run_mtf:
        st.header("📊 Multi-Timeframe Analysis")
        
        if 'data' in st.session_state and MultiTimeframeAnalyzer is not None:
            data = st.session_state['data']
            
            analyzer = MultiTimeframeAnalyzer()
            result = analyzer.analyze_from_single_data(data, Timeframe.D1)
            
            st.metric("MTF Signal", result.signal.upper())
            st.metric("Confidence", f"{result.confidence:.1f}%")
            st.metric("Alignment", f"{result.alignment_score:.1f}%")
            
            st.markdown("**Reasoning:**")
            for reason in result.reasoning:
                st.write(f"- {reason}")
        else:
            st.warning("MTF analysis requires data and MultiTimeframeAnalyzer module.")
    
    # ML Signal Filter
    if run_ml:
        st.header("🤖 ML Signal Filter")
        
        if 'data' in st.session_state and MLSignalFilter is not None:
            data = st.session_state['data']
            
            with st.spinner("Running ML analysis..."):
                ml_filter = MLSignalFilter()
                
                if len(data) > 200:
                    # Train on historical data
                    ml_filter.train(data[:-50])
                    
                    # Predict on recent data
                    predictions = ml_filter.predict(data[-50:])
                    
                    if predictions:
                        latest_pred = predictions[-1]
                        
                        st.metric("ML Signal", latest_pred.signal.upper())
                        st.metric("Confidence", f"{latest_pred.confidence:.1f}%")
                        
                        # Feature importance
                        importance = ml_filter.get_feature_importance()
                        if importance:
                            st.markdown("**Top Features:**")
                            for name, imp in list(importance.items())[:5]:
                                st.write(f"- {name}: {imp:.3f}")
                else:
                    st.warning("Need more data for ML analysis (200+ bars)")
        else:
            st.warning("ML analysis requires data and MLSignalFilter module.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Universal Trading Indicator v1.0 | "
        "⚠️ For educational purposes only. Not financial advice."
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    missing = check_dependencies()
    
    if missing:
        print("=" * 60)
        print("MISSING DEPENDENCIES")
        print("=" * 60)
        print(f"\nThe following packages are required but not installed:")
        print(f"  {', '.join(missing)}")
        print(f"\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        print(f"\nThen run the dashboard with:")
        print(f"  streamlit run streamlit_dashboard.py")
    else:
        main_dashboard()
