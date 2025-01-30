import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
import pytz
from scipy.stats import norm
import time

class PaperTrade:
    """Class to represent a paper trade"""
    def __init__(self, entry_price, quantity, trade_type, timestamp):
        self.entry_price = entry_price
        self.quantity = quantity
        self.trade_type = trade_type  # 'BUY' or 'SELL'
        self.entry_time = timestamp
        self.exit_time = None
        self.exit_price = None
        self.pnl = 0
        self.status = "OPEN"
        
    def close_trade(self, exit_price, exit_time):
        """Close the trade and calculate P&L"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = "CLOSED"
        
        # Calculate P&L
        if self.trade_type == "BUY":
            self.pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # SELL
            self.pnl = (self.entry_price - self.exit_price) * self.quantity
            
    def get_unrealized_pnl(self, current_price):
        """Calculate unrealized P&L"""
        if self.trade_type == "BUY":
            return (current_price - self.entry_price) * self.quantity
        else:  # SELL
            return (self.entry_price - current_price) * self.quantity
            
    def to_dict(self):
        """Convert trade to dictionary for display"""
        return {
            'Entry Time': self.entry_time,
            'Exit Time': self.exit_time if self.exit_time else 'Open',
            'Type': self.trade_type,
            'Entry Price': f"₹{self.entry_price:,.2f}",
            'Exit Price': f"₹{self.exit_price:,.2f}" if self.exit_price else '-',
            'Quantity': self.quantity,
            'P&L': f"₹{self.pnl:,.2f}",
            'Status': self.status
        }

class TradeSignal:
    """Class to represent a trading signal"""
    def __init__(self, timestamp, signal_type, price, strategy):
        self.timestamp = timestamp
        self.signal_type = signal_type  # 'BUY', 'SELL', or 'EXIT'
        self.price = price
        self.strategy = strategy
        
    def to_dict(self):
        """Convert signal to dictionary for display"""
        return {
            'Time': self.timestamp,
            'Signal': self.signal_type,
            'Price': f"₹{self.price:,.2f}",
            'Strategy': self.strategy
        }

class Position:
    """Class to represent an options position"""
    def __init__(self, strike, option_type, entry_price, quantity):
        self.strike = strike
        self.option_type = option_type  # 'CALL' or 'PUT'
        self.entry_price = entry_price
        self.quantity = quantity
        self.entry_time = datetime.now(pytz.timezone('Asia/Kolkata'))
        self.status = "OPEN"
        self.pnl = 0
        
    def update_pnl(self, current_price):
        """Update position P&L"""
        self.pnl = (current_price - self.entry_price) * self.quantity
        
    def to_dict(self):
        """Convert position to dictionary for display"""
        return {
            'Strike': self.strike,
            'Type': self.option_type,
            'Entry Price': f"₹{self.entry_price:,.2f}",
            'Quantity': self.quantity,
            'P&L': f"₹{self.pnl:,.2f}",
            'Status': self.status
        }

class NiftyOptionsAI:
    def __init__(self):
        """Initialize the NiftyOptionsAI class"""
        st.set_page_config(layout="wide", page_title="NIFTY Options AI")
        
        # Initialize strategies
        self.strategies = {
            "Triple MA": self.triple_ma_strategy,
            "Volume Breakout": self.volume_breakout_strategy,
            "Bollinger Mean Reversion": self.bollinger_mean_reversion,
            "Options Greeks": self.options_greek_strategy,
            "Support/Resistance": self.support_resistance_strategy,
            "Momentum Risk Parity": self.momentum_risk_parity,
            "EMA-MA Crossover": self.ema_ma_crossover_strategy,
            "RSI Strategy": self.rsi_strategy,
            "MACD Strategy": self.macd_strategy
        }
        
        # Constants
        self.INDICES = {
            "NIFTY 50": "^NSEI",
            "BANK NIFTY": "^NSEBANK",
            "FINNIFTY": "NIFTY_FIN_SERVICE.NS"
        }
        
        self.SYMBOL_NAMES = {
            "^NSEI": {"name": "NIFTY 50", "lot_size": 75},
            "^NSEBANK": {"name": "BANK NIFTY", "lot_size": 30},
            "NIFTY_FIN_SERVICE.NS": {"name": "FINNIFTY", "lot_size": 65}
        }
        
        # Initialize session state
        self.initialize_session_state()
        
        # Set default index
        if 'selected_index' not in st.session_state:
            st.session_state.selected_index = "NIFTY 50"
        
        # Initialize data with default symbol
        default_symbol = self.INDICES[st.session_state.selected_index]
        self.initial_data = self.load_nifty_data(default_symbol)
        
        if self.initial_data is not None:
            st.session_state.current_price = self.initial_data['Close'].iloc[-1]
        else:
            st.warning("Unable to load initial market data")

    def initialize_session_state(self):
        """Initialize all session state variables"""
        if 'strategy' not in st.session_state:
            st.session_state.strategy = "Triple MA"
        if 'positions' not in st.session_state:
            st.session_state.positions = []
        if 'trades' not in st.session_state:
            st.session_state.trades = []
        if 'cash' not in st.session_state:
            st.session_state.cash = 100000
        if 'daily_signals' not in st.session_state:
            st.session_state.daily_signals = []
        if 'current_price' not in st.session_state:
            st.session_state.current_price = 0
        
        # Add testing mode
        if 'testing_mode' not in st.session_state:
            st.session_state.testing_mode = False
        
        # Strategy parameters
        if 'fast_period' not in st.session_state:
            st.session_state.fast_period = 5
        if 'medium_period' not in st.session_state:
            st.session_state.medium_period = 13
        if 'slow_period' not in st.session_state:
            st.session_state.slow_period = 26
        if 'volume_multiplier' not in st.session_state:
            st.session_state.volume_multiplier = 1.5
        if 'price_period' not in st.session_state:
            st.session_state.price_period = 20
        if 'ma_length' not in st.session_state:
            st.session_state.ma_length = 20
        if 'ema_length' not in st.session_state:
            st.session_state.ema_length = 9
        if 'rsi_period' not in st.session_state:
            st.session_state.rsi_period = 14
        if 'macd_fast' not in st.session_state:
            st.session_state.macd_fast = 12
        if 'macd_slow' not in st.session_state:
            st.session_state.macd_slow = 26
        if 'macd_signal' not in st.session_state:
            st.session_state.macd_signal = 9
        
        # Risk management
        if 'max_position_size' not in st.session_state:
            st.session_state.max_position_size = 0.02
        if 'stop_loss_pct' not in st.session_state:
            st.session_state.stop_loss_pct = 0.02
        if 'take_profit_pct' not in st.session_state:
            st.session_state.take_profit_pct = 0.03

        if 'nifty_data' not in st.session_state:
            st.session_state.nifty_data = None
        if 'selected_index' not in st.session_state:
            st.session_state.selected_index = "NIFTY 50"
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 60

    def load_nifty_data(self, symbol):
        """Load and preprocess NIFTY data"""
        try:
            # Get data from yfinance
            nifty = yf.Ticker(symbol)
            df = nifty.history(period='1d', interval='5m')
            
            if df.empty:
                st.error("No data available")
                return None
            
            # Store current price
            st.session_state.current_price = df['Close'].iloc[-1]
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    def get_nearest_strikes(self):
        """Get nearest option strikes"""
        if 'current_price' not in st.session_state:
            return []
        
        current_price = st.session_state.current_price
        base_strike = round(current_price / 50) * 50
        strikes = []
        
        for i in range(-5, 6):  # 5 strikes above and below
            strike = base_strike + (i * 50)
            strikes.append(strike)
            
        return strikes

    def calculate_iv(self, option_type='call'):
        """Calculate implied volatility based on market conditions"""
        try:
            if 'nifty_data' not in st.session_state:
                return 0.15
                
            data = st.session_state.nifty_data
            returns = np.log(data['Close'] / data['Close'].shift(1))
            hist_vol = returns.std() * np.sqrt(252)
            
            # Adjust IV based on market conditions
            if option_type == 'call':
                iv = hist_vol * 1.1 if data['Close'].iloc[-1] > data['Close'].iloc[0] else hist_vol * 0.9
            else:
                iv = hist_vol * 1.1 if data['Close'].iloc[-1] < data['Close'].iloc[0] else hist_vol * 0.9
                
            return max(iv, 0.12)  # Minimum IV of 12%
            
        except Exception:
            return 0.15

    def generate_option_chain(self):
        """Generate realistic option chain"""
        try:
            strikes = self.get_nearest_strikes()
            current_price = st.session_state.current_price
            expiry_days = 30  # Next month expiry
            
            calls = []
            puts = []
            
            for strike in strikes:
                # Calculate IVs
                call_iv = self.calculate_iv('call')
                put_iv = self.calculate_iv('put')
                
                # Calculate option prices
                call_price = self.black_scholes(current_price, strike, expiry_days/365, 0.05, call_iv, 'call')
                put_price = self.black_scholes(current_price, strike, expiry_days/365, 0.05, put_iv, 'put')
                
                # Calculate Greeks
                call_greeks = self.calculate_greeks(current_price, strike, expiry_days/365, 0.05, call_iv, 'call')
                put_greeks = self.calculate_greeks(current_price, strike, expiry_days/365, 0.05, put_iv, 'put')
                
                calls.append({
                    'Strike': strike,
                    'Premium': call_price,
                    'IV': call_iv,
                    'Delta': call_greeks['delta'],
                    'Gamma': call_greeks['gamma'],
                    'Theta': call_greeks['theta'],
                    'Vega': call_greeks['vega']
                })
                
                puts.append({
                    'Strike': strike,
                    'Premium': put_price,
                    'IV': put_iv,
                    'Delta': put_greeks['delta'],
                    'Gamma': put_greeks['gamma'],
                    'Theta': put_greeks['theta'],
                    'Vega': put_greeks['vega']
                })
                
            return pd.DataFrame(calls), pd.DataFrame(puts)
            
        except Exception as e:
            st.error(f"Error generating option chain: {str(e)}")
            return None, None

    def black_scholes(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option price using Black-Scholes"""
        try:
            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type == 'call':
                price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            else:
                price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
                
            return round(max(price, 0.05), 2)
            
        except Exception:
            return 0.05

    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        """Calculate option Greeks"""
        try:
            d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type == 'call':
                delta = norm.cdf(d1)
            else:
                delta = -norm.cdf(-d1)
                
            gamma = norm.pdf(d1)/(S*sigma*np.sqrt(T))
            theta = (-S*sigma*norm.pdf(d1))/(2*np.sqrt(T))
            vega = S*np.sqrt(T)*norm.pdf(d1)
            
            return {
                'delta': round(delta, 3),
                'gamma': round(gamma, 4),
                'theta': round(theta, 2),
                'vega': round(vega, 2)
            }
            
        except Exception:
            return {
                'delta': 0,
                'gamma': 0,
                'theta': 0,
                'vega': 0
            }

    def analyze_market_conditions(self):
        """Analyze market conditions for trading signals"""
        try:
            data = st.session_state.nifty_data
            
            # Calculate indicators
            data['SMA20'] = data['Close'].rolling(window=20).mean()
            data['SMA50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self.calculate_rsi(data['Close'])
            
            current_price = data['Close'].iloc[-1]
            sma20 = data['SMA20'].iloc[-1]
            sma50 = data['SMA50'].iloc[-1]
            rsi = data['RSI'].iloc[-1]
            
            # Market trend analysis
            trend = "Bullish" if sma20 > sma50 else "Bearish"
            strength = "Strong" if abs(rsi - 50) > 20 else "Moderate"
            
            return {
                'trend': trend,
                'strength': strength,
                'rsi': rsi,
                'price': current_price
            }
            
        except Exception as e:
            st.error(f"Error analyzing market: {str(e)}")
            return None

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_candlestick_chart(self, df):
        """Create interactive candlestick chart with indicators"""
        try:
            fig = go.Figure()
            
            # Add candlestick
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='NIFTY'
            ))
            
            # Add MA and EMA
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[f'MA{st.session_state.ma_length}'],
                line=dict(color='blue', width=1),
                name=f'MA{st.session_state.ma_length}'
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['EMA'],  # Using the EMA calculated in calculate_indicators
                line=dict(color='orange', width=1),
                name=f'EMA{st.session_state.ema_length}'
            ))
            
            # Update layout
            fig.update_layout(
                title='NIFTY Price Action',
                yaxis_title='Price',
                xaxis_title='Time',
                height=600,
                template='plotly_dark',
                xaxis_rangeslider_visible=False
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None

    def create_volume_chart(self, data):
        """Create volume chart"""
        fig = go.Figure()
        
        # Add volume bars
        colors = ['red' if close < open else 'green' 
                  for close, open in zip(data['Close'], data['Open'])]
        
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['Volume'],
            marker_color=colors,
            name='Volume'
        ))
        
        # Update layout
        fig.update_layout(
            title='Trading Volume',
            yaxis_title='Volume',
            xaxis_title='Time',
            height=200,
            template='plotly_dark'
        )
        
        return fig

    def display_technical_charts(self, df):
        """Display technical analysis charts"""
        try:
            st.subheader("Technical Analysis")
            
            # Create tabs for different chart types
            chart_tab1, chart_tab2, chart_tab3 = st.tabs([
                "Price Action", "Technical Indicators", "Volume Analysis"
            ])
            
            with chart_tab1:
                # Main price chart
                fig_price = self.create_candlestick_chart(df)
                if fig_price:
                    st.plotly_chart(fig_price, use_container_width=True, key="price_chart")
            
            with chart_tab2:
                # Technical indicators in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI Chart
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(
                        x=df.index,
                        y=df['RSI'],
                        line=dict(color='purple', width=1),
                        name='RSI'
                    ))
                    
                    # Add RSI levels
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
                    
                    fig_rsi.update_layout(
                        title='RSI',
                        yaxis_title='RSI',
                        height=300,
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig_rsi, use_container_width=True, key="rsi_chart")
                
                with col2:
                    # MACD Chart
                    fig_macd = go.Figure()
                    fig_macd.add_trace(go.Scatter(
                        x=df.index,
                        y=df['MACD'],
                        line=dict(color='blue', width=1),
                        name='MACD'
                    ))
                    
                    fig_macd.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Signal_Line'],
                        line=dict(color='orange', width=1),
                        name='Signal'
                    ))
                    
                    # Add MACD histogram
                    fig_macd.add_trace(go.Bar(
                        x=df.index,
                        y=df['MACD'] - df['Signal_Line'],
                        name='MACD Histogram',
                        marker_color=np.where(df['MACD'] >= df['Signal_Line'], 'green', 'red')
                    ))
                    
                    fig_macd.update_layout(
                        title='MACD',
                        yaxis_title='Value',
                        height=300,
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig_macd, use_container_width=True, key="macd_chart")
                
                # Bollinger Bands
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(
                    x=df.index, 
                    y=df['BB_upper'],
                    line=dict(color='gray', width=1),
                    name='Upper Band'
                ))
                
                fig_bb.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_middle'],
                    line=dict(color='blue', width=1),
                    name='Middle Band'
                ))
                
                fig_bb.add_trace(go.Scatter(
                    x=df.index,
                    y=df['BB_lower'],
                    line=dict(color='gray', width=1),
                    name='Lower Band',
                    fill='tonexty'
                ))
                
                fig_bb.update_layout(
                    title='Bollinger Bands',
                    yaxis_title='Price',
                    height=300,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig_bb, use_container_width=True, key="bb_chart")
            
            with chart_tab3:
                # Volume Analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    # Volume Chart
                    fig_volume = go.Figure()
                    colors = ['red' if close < open else 'green' 
                             for close, open in zip(df['Close'], df['Open'])]
                    
                    fig_volume.add_trace(go.Bar(
                        x=df.index,
                        y=df['Volume'],
                        marker_color=colors,
                        name='Volume'
                    ))
                    
                    # Add volume MA
                    fig_volume.add_trace(go.Scatter(
                        x=df.index,
                        y=df['Volume'].rolling(window=20).mean(),
                        line=dict(color='yellow', width=1),
                        name='Volume MA(20)'
                    ))
                    
                    fig_volume.update_layout(
                        title='Volume Analysis',
                        yaxis_title='Volume',
                        height=300,
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig_volume, use_container_width=True, key="volume_chart")
                
                with col2:
                    # Volume Profile
                    volume_profile = df.groupby(pd.qcut(df['Close'], 10))['Volume'].sum()
                    
                    fig_vp = go.Figure()
                    fig_vp.add_trace(go.Bar(
                        y=volume_profile.index.map(lambda x: x.left),
                        x=volume_profile.values,
                        orientation='h',
                        name='Volume Profile'
                    ))
                    
                    fig_vp.update_layout(
                        title='Volume Profile',
                        xaxis_title='Volume',
                        yaxis_title='Price',
                        height=300,
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig_vp, use_container_width=True, key="volume_profile")
        
        except Exception as e:
            st.error(f"Error displaying technical charts: {str(e)}")

    def display_options_chain(self, df):
        """Display options chain data"""
        try:
            st.subheader("Options Chain Analysis")
            
            # Create tabs for different options analysis
            options_tab1, options_tab2, options_tab3 = st.tabs([
                "Options Chain", "Greeks Analysis", "Options Strategy Builder"
            ])
            
            with options_tab1:
                # Get options chain data
                options_df, spot_price = self.get_nse_option_chain(
                    self.INDICES[st.session_state.selected_index]
                )
                
                if options_df is not None:
                    # Calculate time to expiry
                    expiry_date = pd.to_datetime(options_df['expiry'].min())
                    days_to_expiry = (expiry_date - pd.Timestamp.now()).days
                    
                    # Filter strikes around current price
                    nearest_strike = round(spot_price / 50) * 50
                    strike_range = 500  # Range of strikes to display
                    
                    filtered_df = options_df[
                        (options_df['strike'] >= nearest_strike - strike_range) & 
                        (options_df['strike'] <= nearest_strike + strike_range)
                    ].copy()
                    
                    # Split into calls and puts
                    calls = filtered_df[filtered_df['type'] == 'CE'].sort_values('strike')
                    puts = filtered_df[filtered_df['type'] == 'PE'].sort_values('strike')
                    
                    # Display options data
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Calls")
                        if not calls.empty:
                            st.dataframe(
                                calls[['strike', 'ltp', 'change', 'volume', 'oi', 'iv']],
                                hide_index=True
                            )
                    
                    with col2:
                        st.subheader("Puts")
                        if not puts.empty:
                            st.dataframe(
                                puts[['strike', 'ltp', 'change', 'volume', 'oi', 'iv']],
                                hide_index=True
                            )
                    
                    # Display PCR and other metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Days to Expiry", days_to_expiry)
                    
                    with col2:
                        ce_oi = calls['oi'].sum()
                        pe_oi = puts['oi'].sum()
                        pcr = pe_oi / ce_oi if ce_oi > 0 else 0
                        st.metric("Put-Call Ratio", f"{pcr:.2f}")
                    
                    with col3:
                        st.metric("ATM Strike", nearest_strike)
                        
                    with col4:
                        max_pain = self.calculate_max_pain(filtered_df)
                        st.metric("Max Pain", max_pain)
            
            with options_tab2:
                if options_df is not None:
                    st.subheader("Greeks Analysis")
                    
                    # Greeks visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Delta curve
                        fig_delta = go.Figure()
                        
                        fig_delta.add_trace(go.Scatter(
                            x=calls['strike'],
                            y=calls['delta'],
                            name='Call Delta'
                        ))
                        
                        fig_delta.add_trace(go.Scatter(
                            x=puts['strike'],
                            y=puts['delta'],
                            name='Put Delta'
                        ))
                        
                        fig_delta.update_layout(
                            title='Delta Curve',
                            xaxis_title='Strike Price',
                            yaxis_title='Delta',
                            height=300,
                            template='plotly_dark'
                        )
                        
                        st.plotly_chart(fig_delta, use_container_width=True, key="delta_curve")
                    
                    with col2:
                        # Gamma curve
                        fig_gamma = go.Figure()
                        
                        fig_gamma.add_trace(go.Scatter(
                            x=calls['strike'],
                            y=calls['gamma'],
                            name='Gamma'
                        ))
                        
                        fig_gamma.update_layout(
                            title='Gamma Curve',
                            xaxis_title='Strike Price',
                            yaxis_title='Gamma',
                            height=300,
                            template='plotly_dark'
                        )
                        
                        st.plotly_chart(fig_gamma, use_container_width=True, key="gamma_curve")
                    
                    # Theta and Vega
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Theta curve
                        fig_theta = go.Figure()
                        
                        fig_theta.add_trace(go.Scatter(
                            x=calls['strike'],
                            y=calls['theta'],
                            name='Call Theta'
                        ))
                        
                        fig_theta.add_trace(go.Scatter(
                            x=puts['strike'],
                            y=puts['theta'],
                            name='Put Theta'
                        ))
                        
                        fig_theta.update_layout(
                            title='Theta Curve',
                            xaxis_title='Strike Price',
                            yaxis_title='Theta',
                            height=300,
                            template='plotly_dark'
                        )
                        
                        st.plotly_chart(fig_theta, use_container_width=True, key="theta_curve")
                    
                    with col2:
                        # Vega curve
                        fig_vega = go.Figure()
                        
                        fig_vega.add_trace(go.Scatter(
                            x=calls['strike'],
                            y=calls['vega'],
                            name='Vega'
                        ))
                        
                        fig_vega.update_layout(
                            title='Vega Curve',
                            xaxis_title='Strike Price',
                            yaxis_title='Vega',
                            height=300,
                            template='plotly_dark'
                        )
                        
                        st.plotly_chart(fig_vega, use_container_width=True, key="vega_curve")
            
            with options_tab3:
                st.subheader("Options Strategy Builder")
                
                # Strategy selection
                strategy_type = st.selectbox(
                    "Select Strategy Type",
                    ["Single Option", "Spread", "Combination"]
                )
                
                if strategy_type == "Single Option":
                    self.display_single_option_strategy(options_df)
                elif strategy_type == "Spread":
                    self.display_spread_strategy(options_df)
                else:
                    self.display_combination_strategy(options_df)
                
        except Exception as e:
            st.error(f"Error displaying options chain: {str(e)}")

    def calculate_max_pain(self, options_df):
        """Calculate option chain max pain point"""
        try:
            strikes = sorted(options_df['strike'].unique())
            pain = []
            
            for strike in strikes:
                total_pain = 0
                
                # Calculate call options pain
                calls = options_df[options_df['type'] == 'CE']
                for _, call in calls.iterrows():
                    if strike > call['strike']:
                        total_pain += (strike - call['strike']) * call['oi']
                
                # Calculate put options pain
                puts = options_df[options_df['type'] == 'PE']
                for _, put in puts.iterrows():
                    if strike < put['strike']:
                        total_pain += (put['strike'] - strike) * put['oi']
                
                pain.append(total_pain)
            
            # Find strike price with minimum pain
            max_pain_strike = strikes[np.argmin(pain)]
            return max_pain_strike
            
        except Exception as e:
            st.error(f"Error calculating max pain: {str(e)}")
            return None

    def display_market_overview(self, df):
        """Display market overview section"""
        try:
            st.header("Market Overview")
            
            # Market summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                current_price = df['Close'].iloc[-1]
                prev_close = df['Close'].iloc[-2]
                change = ((current_price - prev_close) / prev_close) * 100
                st.metric("Price", f"₹{current_price:,.2f}", f"{change:+.2f}%")
                
            with col2:
                volume = df['Volume'].iloc[-1]
                avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
                vol_change = ((volume - avg_volume) / avg_volume) * 100
                st.metric("Volume", f"{volume:,.0f}", f"{vol_change:+.2f}%")
                
            with col3:
                rsi = df['RSI'].iloc[-1]
                st.metric("RSI", f"{rsi:.2f}")
                
            with col4:
                macd = df['MACD'].iloc[-1]
                signal = df['Signal_Line'].iloc[-1]
                macd_diff = macd - signal
                st.metric("MACD", f"{macd:.2f}", f"{macd_diff:+.2f}")
            
            # Display additional market metrics
            self.display_metrics(df)
            
            # Display trading signals if available
            if 'signals_df' in st.session_state and st.session_state.signals_df is not None:
                self.display_signals(st.session_state.signals_df)
                
        except Exception as e:
            st.error(f"Error displaying market overview: {str(e)}")

    def display_metrics(self, df):
        """Display additional market metrics"""
        try:
            st.subheader("Market Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Volatility
                returns = df['Close'].pct_change()
                volatility = returns.std() * np.sqrt(252) * 100
                st.metric("Volatility (Annual)", f"{volatility:.2f}%")
                
            with col2:
                # Average True Range
                high_low = df['High'] - df['Low']
                high_close = np.abs(df['High'] - df['Close'].shift())
                low_close = np.abs(df['Low'] - df['Close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = np.max(ranges, axis=1)
                atr = true_range.rolling(14).mean().iloc[-1]
                st.metric("ATR", f"{atr:.2f}")
                
            with col3:
                # Trend Strength
                adx = self.calculate_adx(df)
                st.metric("ADX", f"{adx:.2f}")
                
        except Exception as e:
            st.error(f"Error displaying metrics: {str(e)}")

    def display_signals(self, signals_df):
        """Display trading signals"""
        try:
            st.subheader("Trading Signals")
            
            if not signals_df.empty:
                # Get latest non-HOLD signal
                latest_signals = signals_df[signals_df['Signal'] != 'HOLD'].tail()
                
                if not latest_signals.empty:
                    latest_signal = latest_signals.iloc[-1]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Signal", latest_signal['Signal'])
                    with col2:
                        st.metric("Trade", latest_signal['Trade'])
                    
                    # Signal history
                    st.dataframe(
                        latest_signals[['Signal', 'Trade']].tail(),
                        hide_index=True
                    )
                else:
                    st.info("No trading signals generated")
                
        except Exception as e:
            st.error(f"Error displaying signals: {str(e)}")

    def calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        try:
            # Calculate directional movement
            high_diff = df['High'].diff()
            low_diff = df['Low'].diff()
            
            pos_dm = np.where((high_diff > 0) & (high_diff > low_diff), high_diff, 0)
            neg_dm = np.where((low_diff < 0) & (abs(low_diff) > high_diff), abs(low_diff), 0)
            
            # Calculate true range
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = np.max([high_low, high_close, low_close], axis=0)
            
            # Smooth the indicators
            tr_ema = pd.Series(true_range).ewm(span=period, adjust=False).mean()
            pos_di = pd.Series(pos_dm).ewm(span=period, adjust=False).mean() / tr_ema * 100
            neg_di = pd.Series(neg_dm).ewm(span=period, adjust=False).mean() / tr_ema * 100
            
            # Calculate ADX
            dx = np.abs(pos_di - neg_di) / (pos_di + neg_di) * 100
            adx = pd.Series(dx).ewm(span=period, adjust=False).mean()
            
            return adx.iloc[-1]
            
        except Exception:
            return 0

    def display_strategy_analysis(self, df, signals_df):
        """Display strategy analysis and signals"""
        try:
            st.header("Strategy Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Strategy Performance")
                if signals_df is not None and not signals_df.empty:
                    # Strategy metrics
                    total_signals = len(signals_df[signals_df['Signal'] != 'HOLD'])
                    buy_signals = len(signals_df[signals_df['Signal'] == 'BUY'])
                    sell_signals = len(signals_df[signals_df['Signal'] == 'SELL'])
                    
                    st.metric("Total Signals", total_signals)
                    st.metric("Buy Signals", buy_signals)
                    st.metric("Sell Signals", sell_signals)
                    
                    # Signal distribution chart
                    fig = go.Figure()
                    signal_counts = signals_df['Signal'].value_counts()
                    fig.add_trace(go.Bar(
                        x=signal_counts.index,
                        y=signal_counts.values,
                        name='Signal Distribution'
                    ))
                    
                    fig.update_layout(
                        title='Signal Distribution',
                        xaxis_title='Signal Type',
                        yaxis_title='Count',
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, key="signal_dist")
                    
            with col2:
                st.subheader("Strategy Settings")
                # Display current strategy parameters
                st.json({
                    'Strategy': st.session_state.strategy,
                    'Parameters': self.get_strategy_parameters(),
                    'Risk Management': {
                        'Max Position Size': f"{st.session_state.max_position_size*100}%",
                        'Stop Loss': f"{st.session_state.stop_loss_pct*100}%",
                        'Take Profit': f"{st.session_state.take_profit_pct*100}%"
                    }
                })
                
            # Backtest results if available
            if hasattr(st.session_state, 'backtest_results'):
                st.subheader("Backtest Results")
                self.display_backtest_results()
                
        except Exception as e:
            st.error(f"Error displaying strategy analysis: {str(e)}")

    def get_strategy_parameters(self):
        """Get current strategy parameters"""
        params = {}
        
        if st.session_state.strategy == "Triple MA":
            params = {
                'Fast Period': st.session_state.fast_period,
                'Medium Period': st.session_state.medium_period,
                'Slow Period': st.session_state.slow_period
            }
        elif st.session_state.strategy == "RSI Strategy":
            params = {
                'RSI Period': st.session_state.rsi_period,
                'Overbought': st.session_state.rsi_overbought,
                'Oversold': st.session_state.rsi_oversold
            }
            # Add other strategies...
        
        return params

    def display_backtest_results(self):
        """Display backtest results"""
        try:
            results = st.session_state.backtest_results
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Returns", f"{results['total_returns']:.2f}%")
            with col2:
                st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
            with col3:
                st.metric("Max Drawdown", f"{results['max_drawdown']:.2f}%")
            with col4:
                st.metric("Win Rate", f"{results['win_rate']:.2f}%")
            
            # Equity curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=results['equity_curve'].index,
                y=results['equity_curve'].values,
                mode='lines',
                name='Equity Curve'
            ))
            
            fig.update_layout(
                title='Backtest Equity Curve',
                xaxis_title='Date',
                yaxis_title='Portfolio Value',
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True, key="equity_curve")
            
        except Exception as e:
            st.error(f"Error displaying backtest results: {str(e)}")

    def display_portfolio_dashboard(self, df):
        """Display portfolio performance dashboard"""
        try:
            st.header("Portfolio Dashboard")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Account Value", f"₹{st.session_state.cash:,.2f}")
            
            with col2:
                total_pnl = sum(trade.pnl for trade in st.session_state.trades)
                st.metric("Total P&L", f"₹{total_pnl:,.2f}")
                
            with col3:
                win_trades = len([t for t in st.session_state.trades if t.pnl > 0])
                total_trades = len(st.session_state.trades)
                win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
                
            # Display trade history
            if st.session_state.trades:
                st.subheader("Trade History")
                trades_df = pd.DataFrame([vars(t) for t in st.session_state.trades])
                st.dataframe(trades_df)
            
        except Exception as e:
            st.error(f"Error displaying portfolio dashboard: {str(e)}")

    def display_positions(self, df):
        """Display current positions and performance"""
        try:
            st.subheader("Open Positions")
            
            # Get current positions
            open_positions = [p for p in st.session_state.positions if p['status'] == 'OPEN']
            
            if open_positions:
                # Create positions dataframe
                positions_df = pd.DataFrame(open_positions)
                
                # Calculate current values
                current_price = df['Close'].iloc[-1]
                
                # Format for display
                display_df = positions_df.copy()
                display_df['Entry Time'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                display_df['Type'] = display_df['type']
                display_df['Entry Price'] = display_df['entry_price'].map('₹{:,.2f}'.format)
                display_df['Quantity'] = display_df['quantity']
                display_df['P&L'] = display_df['pnl'].map('₹{:,.2f}'.format)
                
                # Display positions table
                st.dataframe(
                    display_df[['Entry Time', 'Type', 'Entry Price', 'Quantity', 'P&L']],
                    hide_index=True
                )
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_value = sum(p['entry_price'] * p['quantity'] for p in open_positions)
                    st.metric("Total Position Value", f"₹{total_value:,.2f}")
                    
                with col2:
                    total_pnl = sum(p['pnl'] for p in open_positions)
                    st.metric("Unrealized P&L", f"₹{total_pnl:,.2f}")
                    
                with col3:
                    roi = (total_pnl / total_value * 100) if total_value > 0 else 0
                    st.metric("ROI", f"{roi:.2f}%")
                    
            else:
                st.info("No open positions")
            
        except Exception as e:
            st.error(f"Error displaying positions: {str(e)}")

    def display_trade_history(self):
        """Display trading history and performance metrics"""
        try:
            st.subheader("Trade History")
            
            if st.session_state.trades:
                # Create trades dataframe
                trades_df = pd.DataFrame(st.session_state.trades)
                
                # Format for display
                display_df = trades_df.copy()
                display_df['Entry Time'] = pd.to_datetime(display_df['entry_date']).dt.strftime('%Y-%m-%d %H:%M')
                display_df['Exit Time'] = pd.to_datetime(display_df['exit_date']).dt.strftime('%Y-%m-%d %H:%M')
                display_df['Type'] = display_df['type']
                display_df['Entry Price'] = display_df['entry_price'].map('₹{:,.2f}'.format)
                display_df['Exit Price'] = display_df['exit_price'].map('₹{:,.2f}'.format)
                display_df['P&L'] = display_df['pnl'].map('₹{:,.2f}'.format)
                
                # Display trades table
                st.dataframe(
                    display_df[['Entry Time', 'Exit Time', 'Type', 'Entry Price', 'Exit Price', 'P&L']],
                    hide_index=True
                )
                
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_trades = len(trades_df)
                    st.metric("Total Trades", total_trades)
                    
                with col2:
                    win_trades = len(trades_df[trades_df['pnl'] > 0])
                    win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
                    st.metric("Win Rate", f"{win_rate:.1f}%")
                    
                with col3:
                    total_pnl = trades_df['pnl'].sum()
                    st.metric("Total P&L", f"₹{total_pnl:,.2f}")
                    
                with col4:
                    avg_pnl = trades_df['pnl'].mean() if not trades_df.empty else 0
                    st.metric("Avg P&L per Trade", f"₹{avg_pnl:,.2f}")
                    
                # P&L Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=pd.to_datetime(trades_df['exit_date']),
                    y=trades_df['pnl'].cumsum(),
                    mode='lines+markers',
                    name='Cumulative P&L'
                ))
                
                fig.update_layout(
                    title='P&L Curve',
                    xaxis_title='Date',
                    yaxis_title='Cumulative P&L (₹)',
                    height=400,
                    template='plotly_dark'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("No trades executed yet")
            
        except Exception as e:
            st.error(f"Error displaying trade history: {str(e)}")

    def exit_positions(self, current_price, signal_type):
        """Exit positions based on signals"""
        try:
            for position in st.session_state.positions:
                if position['status'] == 'OPEN':
                    if ((signal_type == 'SELL' and position['type'] == 'BUY') or
                        (signal_type == 'BUY' and position['type'] == 'SELL')):
                        
                        # Calculate P&L
                        pnl = (current_price - position['entry_price']) * position['quantity']
                        if position['type'] == 'SELL':
                            pnl = -pnl
                            
                        # Record trade
                        trade = {
                            'entry_date': position['timestamp'],
                            'exit_date': datetime.now(),
                            'type': position['type'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'quantity': position['quantity'],
                            'pnl': pnl
                        }
                        
                        st.session_state.trades.append(trade)
                        position['status'] = 'CLOSED'
                        st.session_state.cash += (current_price * position['quantity'])
                        
                        st.success(f"Closed {position['type']} position at ₹{current_price:,.2f} with P&L: ₹{pnl:,.2f}")
                        
        except Exception as e:
            st.error(f"Error exiting positions: {str(e)}")

    def ema_ma_crossover_strategy(self, df):
        """EMA-MA Crossover Strategy from algo_ai.py"""
        try:
            # Calculate indicators
            df['EMA'] = df['Close'].ewm(span=st.session_state.ema_length, adjust=False).mean()
            df['MA'] = df['Close'].rolling(window=st.session_state.ma_length).mean()
            
            # Initialize signals
            df['Signal'] = 'HOLD'
            df['Trade'] = None
            
            # Generate signals
            for i in range(1, len(df)):
                # Bullish crossover
                if (df['EMA'].iloc[i] > df['MA'].iloc[i] and 
                    df['EMA'].iloc[i-1] <= df['MA'].iloc[i-1]):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'BUY'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                # Bearish crossover
                elif (df['EMA'].iloc[i] < df['MA'].iloc[i] and 
                      df['EMA'].iloc[i-1] >= df['MA'].iloc[i-1]):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'SELL'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                # Exit signals
                elif ((df['EMA'].iloc[i] < df['MA'].iloc[i] and df['Signal'].iloc[i-1] == 'BUY') or
                      (df['EMA'].iloc[i] > df['MA'].iloc[i] and df['Signal'].iloc[i-1] == 'SELL')):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'EXIT'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'EXIT'
                
            return df
            
        except Exception as e:
            st.error(f"Error in EMA-MA Crossover strategy: {str(e)}")
            return None

    def rsi_strategy(self, df):
        """RSI Strategy from algo_ai.py"""
        try:
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Initialize signals
            df['Signal'] = 'HOLD'
            df['Trade'] = None
            
            # Generate signals
            for i in range(1, len(df)):
                # Oversold condition
                if df['RSI'].iloc[i] < 30 and df['RSI'].iloc[i-1] >= 30:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'BUY'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                # Overbought condition
                elif df['RSI'].iloc[i] > 70 and df['RSI'].iloc[i-1] <= 70:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'SELL'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                # Exit signals
                elif ((df['RSI'].iloc[i] > 50 and df['Signal'].iloc[i-1] == 'BUY') or
                      (df['RSI'].iloc[i] < 50 and df['Signal'].iloc[i-1] == 'SELL')):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'EXIT'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'EXIT'
                
            return df
            
        except Exception as e:
            st.error(f"Error in RSI strategy: {str(e)}")
            return None

    def macd_strategy(self, df):
        """MACD Strategy from algo_ai.py"""
        try:
            # Calculate MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
            
            # Initialize signals
            df['Signal'] = 'HOLD'
            df['Trade'] = None
            
            # Generate signals
            for i in range(1, len(df)):
                # Bullish crossover
                if (df['MACD'].iloc[i] > df['Signal_Line'].iloc[i] and 
                    df['MACD'].iloc[i-1] <= df['Signal_Line'].iloc[i-1]):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'BUY'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                # Bearish crossover
                elif (df['MACD'].iloc[i] < df['Signal_Line'].iloc[i] and 
                      df['MACD'].iloc[i-1] >= df['Signal_Line'].iloc[i-1]):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'SELL'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                # Exit signals
                elif ((df['MACD_Hist'].iloc[i] < 0 and df['Signal'].iloc[i-1] == 'BUY') or
                      (df['MACD_Hist'].iloc[i] > 0 and df['Signal'].iloc[i-1] == 'SELL')):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'EXIT'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'EXIT'
                
            return df
            
        except Exception as e:
            st.error(f"Error in MACD strategy: {str(e)}")
            return None

    def display_paper_trading(self, df, symbol):
        """Display paper trading interface"""
        st.subheader("Paper Trading")
        
        try:
            # Get latest signal
            signals = df[df['Signal'] != 'HOLD'].copy()
            if not signals.empty:
                latest_signal = signals.iloc[-1]
                
                # Add Trade column if needed
                if 'Trade' not in signals.columns:
                    signals['Trade'] = signals['Signal'].apply(
                        lambda x: 'ENTRY' if x in ['BUY', 'SELL'] else None
                    )
                    latest_signal = signals.iloc[-1]
                
                # Display trading interface
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Latest Signal", latest_signal['Signal'])
                    st.metric("Current Price", f"₹{df['Close'].iloc[-1]:,.2f}")
                
                with col2:
                    quantity = st.number_input("Quantity", min_value=1, value=100)
                    if st.button("Execute Trade"):
                        result = self.process_auto_trade(latest_signal, df, symbol, quantity)
                        if result:
                            st.success(result)
            else:
                st.info("No trading signals available")
            
        except Exception as e:
            st.error(f"Error in paper trading: {str(e)}")

    def triple_ma_strategy(self, df):
        """Triple Moving Average Crossover Strategy"""
        try:
            # Calculate three moving averages
            df['MA_Fast'] = df['Close'].rolling(window=st.session_state.fast_period).mean()
            df['MA_Medium'] = df['Close'].rolling(window=st.session_state.medium_period).mean()
            df['MA_Slow'] = df['Close'].rolling(window=st.session_state.slow_period).mean()
            
            # Initialize signals
            df['Signal'] = 'HOLD'
            df['Trade'] = None
            
            # Generate signals
            for i in range(1, len(df)):
                # Bullish alignment (Fast > Medium > Slow)
                if (df['MA_Fast'].iloc[i] > df['MA_Medium'].iloc[i] > df['MA_Slow'].iloc[i] and
                    not (df['MA_Fast'].iloc[i-1] > df['MA_Medium'].iloc[i-1] > df['MA_Slow'].iloc[i-1])):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'BUY'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                # Bearish alignment (Fast < Medium < Slow)
                elif (df['MA_Fast'].iloc[i] < df['MA_Medium'].iloc[i] < df['MA_Slow'].iloc[i] and
                      not (df['MA_Fast'].iloc[i-1] < df['MA_Medium'].iloc[i-1] < df['MA_Slow'].iloc[i-1])):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'SELL'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                # Exit signals
                elif ((df['MA_Fast'].iloc[i] < df['MA_Medium'].iloc[i] and df['Signal'].iloc[i-1] == 'BUY') or
                      (df['MA_Fast'].iloc[i] > df['MA_Medium'].iloc[i] and df['Signal'].iloc[i-1] == 'SELL')):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'EXIT'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'EXIT'
                
            return df
            
        except Exception as e:
            st.error(f"Error in Triple MA strategy: {str(e)}")
            return None

    def volume_breakout_strategy(self, df):
        """Volume Breakout Strategy"""
        try:
            # Calculate volume and price indicators
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Price_MA'] = df['Close'].rolling(window=st.session_state.price_period).mean()
            
            # Initialize signals
            df['Signal'] = 'HOLD'
            df['Trade'] = None
            
            # Generate signals
            for i in range(1, len(df)):
                # Volume breakout with price confirmation
                if (df['Volume'].iloc[i] > df['Volume_MA'].iloc[i] * st.session_state.volume_multiplier and
                    df['Close'].iloc[i] > df['Price_MA'].iloc[i] and
                    df['Close'].iloc[i] > df['Close'].iloc[i-1]):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'BUY'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                elif (df['Volume'].iloc[i] > df['Volume_MA'].iloc[i] * st.session_state.volume_multiplier and
                      df['Close'].iloc[i] < df['Price_MA'].iloc[i] and
                      df['Close'].iloc[i] < df['Close'].iloc[i-1]):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'SELL'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                # Exit signals
                elif ((df['Close'].iloc[i] < df['Price_MA'].iloc[i] and df['Signal'].iloc[i-1] == 'BUY') or
                      (df['Close'].iloc[i] > df['Price_MA'].iloc[i] and df['Signal'].iloc[i-1] == 'SELL')):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'EXIT'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'EXIT'
                
            return df
            
        except Exception as e:
            st.error(f"Error in Volume Breakout strategy: {str(e)}")
            return None

    def bollinger_mean_reversion(self, df):
        """Bollinger Bands Mean Reversion Strategy"""
        try:
            # Calculate Bollinger Bands
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['STD20'] = df['Close'].rolling(window=20).std()
            df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
            df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)
            
            # Initialize signals
            df['Signal'] = 'HOLD'
            df['Trade'] = None
            
            # Generate signals
            for i in range(1, len(df)):
                # Price below lower band (oversold)
                if df['Close'].iloc[i] < df['Lower_Band'].iloc[i]:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'BUY'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                # Price above upper band (overbought)
                elif df['Close'].iloc[i] > df['Upper_Band'].iloc[i]:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'SELL'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                # Exit signals (mean reversion)
                elif ((df['Close'].iloc[i] > df['MA20'].iloc[i] and df['Signal'].iloc[i-1] == 'BUY') or
                      (df['Close'].iloc[i] < df['MA20'].iloc[i] and df['Signal'].iloc[i-1] == 'SELL')):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'EXIT'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'EXIT'
                
            return df
            
        except Exception as e:
            st.error(f"Error in Bollinger Bands strategy: {str(e)}")
            return None

    def options_greek_strategy(self, df):
        """Options Greek-based Trading Strategy"""
        try:
            # Calculate current price and volatility
            current_price = df['Close'].iloc[-1]
            returns = np.log(df['Close'] / df['Close'].shift(1))
            historical_volatility = returns.std() * np.sqrt(252)
            
            # Initialize signals dataframe
            df['Signal'] = 'HOLD'
            df['Trade'] = None
            
            # Get ATM strike price
            atm_strike = round(current_price / 50) * 50
            
            # Calculate option Greeks for ATM options
            expiry_days = 30  # Next month expiry
            risk_free_rate = 0.05
            
            for i in range(1, len(df)):
                spot_price = df['Close'].iloc[i]
                
                # Calculate Greeks for ATM Call and Put
                call_greeks = self.calculate_greeks(
                    spot_price, atm_strike, expiry_days/365, 
                    risk_free_rate, historical_volatility, 'call'
                )
                
                put_greeks = self.calculate_greeks(
                    spot_price, atm_strike, expiry_days/365, 
                    risk_free_rate, historical_volatility, 'put'
                )
                
                # Trading signals based on Delta and Gamma
                if call_greeks['delta'] > 0.6 and call_greeks['gamma'] > 0.02:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'BUY'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                elif put_greeks['delta'] < -0.6 and put_greeks['gamma'] > 0.02:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'SELL'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                # Exit signals based on Theta decay
                elif ((call_greeks['theta'] < -0.5 and df['Signal'].iloc[i-1] == 'BUY') or
                      (put_greeks['theta'] < -0.5 and df['Signal'].iloc[i-1] == 'SELL')):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'EXIT'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'EXIT'
                
            return df
            
        except Exception as e:
            st.error(f"Error in Options Greek strategy: {str(e)}")
            return None

    def support_resistance_strategy(self, df):
        """Support and Resistance Trading Strategy"""
        try:
            # Calculate pivot points
            df['PP'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['R1'] = 2 * df['PP'] - df['Low']
            df['S1'] = 2 * df['PP'] - df['High']
            df['R2'] = df['PP'] + (df['High'] - df['Low'])
            df['S2'] = df['PP'] - (df['High'] - df['Low'])
            
            # Initialize signals
            df['Signal'] = 'HOLD'
            df['Trade'] = None
            
            # Generate signals
            for i in range(1, len(df)):
                # Breakout above resistance
                if (df['Close'].iloc[i] > df['R1'].iloc[i] and 
                    df['Close'].iloc[i-1] <= df['R1'].iloc[i-1]):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'BUY'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                # Breakdown below support
                elif (df['Close'].iloc[i] < df['S1'].iloc[i] and 
                      df['Close'].iloc[i-1] >= df['S1'].iloc[i-1]):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'SELL'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                # Exit signals
                elif ((df['Close'].iloc[i] < df['PP'].iloc[i] and df['Signal'].iloc[i-1] == 'BUY') or
                      (df['Close'].iloc[i] > df['PP'].iloc[i] and df['Signal'].iloc[i-1] == 'SELL')):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'EXIT'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'EXIT'
                
            return df
            
        except Exception as e:
            st.error(f"Error in Support/Resistance strategy: {str(e)}")
            return None

    def momentum_risk_parity(self, df):
        """Momentum Risk Parity Strategy"""
        try:
            # Calculate momentum indicators
            df['ROC'] = df['Close'].pct_change(periods=10)
            df['RSI'] = self.calculate_rsi(df['Close'])
            df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
            
            # Initialize signals
            df['Signal'] = 'HOLD'
            df['Trade'] = None
            
            # Generate signals
            for i in range(1, len(df)):
                # Risk-adjusted momentum signals
                momentum_score = df['ROC'].iloc[i] / df['Volatility'].iloc[i]
                
                if momentum_score > 1 and df['RSI'].iloc[i] < 70:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'BUY'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                elif momentum_score < -1 and df['RSI'].iloc[i] > 30:
                    df.iloc[i, df.columns.get_loc('Signal')] = 'SELL'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'ENTRY'
                    
                # Exit signals
                elif ((momentum_score < 0 and df['Signal'].iloc[i-1] == 'BUY') or
                      (momentum_score > 0 and df['Signal'].iloc[i-1] == 'SELL')):
                    df.iloc[i, df.columns.get_loc('Signal')] = 'EXIT'
                    df.iloc[i, df.columns.get_loc('Trade')] = 'EXIT'
                
            return df
            
        except Exception as e:
            st.error(f"Error in Momentum Risk Parity strategy: {str(e)}")
            return None

    def display_dashboard(self):
        """Enhanced dashboard with features from algo_ai.py"""
        try:
            # Sidebar controls
            st.sidebar.title("NIFTY Options AI")
            selected_index = st.sidebar.selectbox("Select Index", list(self.INDICES.keys()))
            symbol = self.INDICES[selected_index]
            
            # Strategy selection
            st.session_state.strategy = st.sidebar.selectbox(
                "Select Strategy",
                list(self.strategies.keys())
            )
            
            # Display strategy parameters
            self.display_strategy_parameters(st.session_state.strategy)
            
            # Auto-refresh toggle
            st.sidebar.subheader("Refresh Settings")
            st.session_state.auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
            if st.session_state.auto_refresh:
                st.session_state.refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 5, 300, 60)
            
            # Testing mode
            st.session_state.testing_mode = st.sidebar.checkbox("Testing Mode")
            
            # Load and process data
            df = self.load_nifty_data(symbol)
            if df is not None:
                df = self.calculate_indicators(df)
                strategy = self.strategies.get(st.session_state.strategy)
                signals_df = strategy(df) if strategy else None
                
                # Create tabs
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "Market Overview", 
                    "Technical Analysis",
                    "Options Chain",
                    "Strategy Analysis",
                    "Portfolio"
                ])
                
                with tab1:
                    self.display_market_overview(df)
                    self.display_metrics(df)
                    self.display_signals(signals_df)
                
                with tab2:
                    self.display_technical_charts(df)
                
                with tab3:
                    self.display_options_chain(df)
                
                with tab4:
                    self.display_strategy_analysis(df, signals_df)
                
                with tab5:
                    self.display_portfolio_dashboard(df)
                    self.display_positions(df)
                    self.display_trade_history()
                
                # Auto-refresh logic
                if st.session_state.auto_refresh:
                    time.sleep(st.session_state.refresh_interval)
                    st.experimental_rerun()
                
        except Exception as e:
            st.error(f"Error in dashboard: {str(e)}")
            if st.session_state.testing_mode:
                st.error(f"Detailed error: {str(e)}")

    def load_nifty_data(self, symbol):
        """Load and preprocess NIFTY data"""
        try:
            # Get data from yfinance
            nifty = yf.Ticker(symbol)
            df = nifty.history(period='1d', interval='5m')
            
            if df.empty:
                st.error("No data available")
                return None
            
            # Store current price
            st.session_state.current_price = df['Close'].iloc[-1]
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None

    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            # Moving averages
            df[f'MA{st.session_state.ma_length}'] = df['Close'].rolling(
                window=st.session_state.ma_length).mean()
            df['EMA'] = df['Close'].ewm(
                span=st.session_state.ema_length, adjust=False).mean()
            
            # RSI
            df['RSI'] = self.calculate_rsi(df['Close'])
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            df['BB_upper'] = df['BB_middle'] + 2*df['Close'].rolling(window=20).std()
            df['BB_lower'] = df['BB_middle'] - 2*df['Close'].rolling(window=20).std()
            
            return df
            
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return df

    def display_metrics(self, df):
        """Display additional market metrics"""
        try:
            st.subheader("Market Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Volatility
                returns = df['Close'].pct_change()
                volatility = returns.std() * np.sqrt(252) * 100
                st.metric("Volatility (Annual)", f"{volatility:.2f}%")
                
            with col2:
                # Average True Range
                high_low = df['High'] - df['Low']
                high_close = np.abs(df['High'] - df['Close'].shift())
                low_close = np.abs(df['Low'] - df['Close'].shift())
                ranges = pd.concat([high_low, high_close, low_close], axis=1)
                true_range = np.max(ranges, axis=1)
                atr = true_range.rolling(14).mean().iloc[-1]
                st.metric("ATR", f"{atr:.2f}")
                
            with col3:
                # Trend Strength
                adx = self.calculate_adx(df)
                st.metric("ADX", f"{adx:.2f}")
                
        except Exception as e:
            st.error(f"Error displaying metrics: {str(e)}")

    def display_signals(self, signals_df):
        """Display trading signals"""
        try:
            st.subheader("Trading Signals")
            
            if not signals_df.empty:
                # Get latest non-HOLD signal
                latest_signals = signals_df[signals_df['Signal'] != 'HOLD'].tail()
                
                if not latest_signals.empty:
                    latest_signal = latest_signals.iloc[-1]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Signal", latest_signal['Signal'])
                    with col2:
                        st.metric("Trade", latest_signal['Trade'])
                    
                    # Signal history
                    st.dataframe(
                        latest_signals[['Signal', 'Trade']].tail(),
                        hide_index=True
                    )
                else:
                    st.info("No trading signals generated")
                
        except Exception as e:
            st.error(f"Error displaying signals: {str(e)}")

    def calculate_adx(self, df, period=14):
        """Calculate Average Directional Index"""
        try:
            # Calculate directional movement
            high_diff = df['High'].diff()
            low_diff = df['Low'].diff()
            
            pos_dm = np.where((high_diff > 0) & (high_diff > low_diff), high_diff, 0)
            neg_dm = np.where((low_diff < 0) & (abs(low_diff) > high_diff), abs(low_diff), 0)
            
            # Calculate true range
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            true_range = np.max([high_low, high_close, low_close], axis=0)
            
            # Smooth the indicators
            tr_ema = pd.Series(true_range).ewm(span=period, adjust=False).mean()
            pos_di = pd.Series(pos_dm).ewm(span=period, adjust=False).mean() / tr_ema * 100
            neg_di = pd.Series(neg_dm).ewm(span=period, adjust=False).mean() / tr_ema * 100
            
            # Calculate ADX
            dx = np.abs(pos_di - neg_di) / (pos_di + neg_di) * 100
            adx = pd.Series(dx).ewm(span=period, adjust=False).mean()
            
            return adx.iloc[-1]
            
        except Exception:
            return 0

    def display_strategy_parameters(self, strategy):
        """Display and update strategy parameters"""
        st.sidebar.subheader("Strategy Parameters")
        
        if strategy == "Triple MA":
            st.session_state.fast_period = st.sidebar.slider("Fast MA Period", 3, 15, st.session_state.fast_period)
            st.session_state.medium_period = st.sidebar.slider("Medium MA Period", 10, 30, st.session_state.medium_period)
            st.session_state.slow_period = st.sidebar.slider("Slow MA Period", 20, 50, st.session_state.slow_period)
            
        elif strategy == "Volume Breakout":
            st.session_state.volume_multiplier = st.sidebar.slider("Volume Multiplier", 1.0, 3.0, st.session_state.volume_multiplier)
            st.session_state.price_period = st.sidebar.slider("Price MA Period", 10, 50, st.session_state.price_period)
            
        elif strategy == "Bollinger Mean Reversion":
            st.session_state.bb_period = st.sidebar.slider("BB Period", 10, 30, 20)
            st.session_state.bb_std = st.sidebar.slider("BB Standard Deviation", 1.0, 3.0, 2.0)
            
        elif strategy == "EMA-MA Crossover":
            st.session_state.ma_length = st.sidebar.slider("MA Length", 10, 50, st.session_state.ma_length)
            st.session_state.ema_length = st.sidebar.slider("EMA Length", 5, 25, st.session_state.ema_length)
            
        elif strategy == "RSI Strategy":
            st.session_state.rsi_period = st.sidebar.slider("RSI Period", 7, 21, st.session_state.rsi_period)
            st.session_state.rsi_overbought = st.sidebar.slider("RSI Overbought", 70, 85, 70)
            st.session_state.rsi_oversold = st.sidebar.slider("RSI Oversold", 15, 30, 30)
            
        elif strategy == "MACD Strategy":
            st.session_state.macd_fast = st.sidebar.slider("MACD Fast Period", 8, 20, st.session_state.macd_fast)
            st.session_state.macd_slow = st.sidebar.slider("MACD Slow Period", 20, 40, st.session_state.macd_slow)
            st.session_state.macd_signal = st.sidebar.slider("MACD Signal Period", 5, 15, st.session_state.macd_signal)
            
        elif strategy == "Options Greeks":
            st.session_state.delta_threshold = st.sidebar.slider("Delta Threshold", 0.3, 0.7, 0.5)
            st.session_state.gamma_threshold = st.sidebar.slider("Gamma Threshold", 0.01, 0.05, 0.02)
            st.session_state.theta_threshold = st.sidebar.slider("Theta Threshold", -1.0, -0.1, -0.5)
            
            # Risk Management Parameters
            st.sidebar.subheader("Risk Management")
            st.session_state.max_position_size = st.sidebar.slider("Max Position Size (%)", 1, 10, int(st.session_state.max_position_size * 100)) / 100
            st.session_state.stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 1, 10, int(st.session_state.stop_loss_pct * 100)) / 100
            st.session_state.take_profit_pct = st.sidebar.slider("Take Profit (%)", 1, 20, int(st.session_state.take_profit_pct * 100)) / 100

    def get_nse_option_chain(self, symbol):
        """Get NSE options chain data"""
        try:
            # Get current spot price
            spot_price = st.session_state.current_price
            
            # Calculate ATM strike
            atm_strike = round(spot_price / 50) * 50
            
            # Generate strikes around ATM
            strikes = range(atm_strike - 500, atm_strike + 550, 50)
            
            # Calculate days to expiry (next month expiry)
            today = datetime.now()
            expiry = today + timedelta(days=30 - today.day)
            days_to_expiry = (expiry - today).days / 365  # In years
            
            # Risk-free rate (1-year T-bill rate)
            risk_free_rate = 0.05
            
            # Historical volatility calculation
            if 'nifty_data' in st.session_state and st.session_state.nifty_data is not None:
                returns = np.log(st.session_state.nifty_data['Close'] / 
                               st.session_state.nifty_data['Close'].shift(1))
                historical_volatility = returns.std() * np.sqrt(252)
            else:
                historical_volatility = 0.20  # Default volatility if no data available
            
            # Generate options data
            calls = []
            puts = []
            
            for strike in strikes:
                # Calculate call option
                call_price = self.black_scholes(
                    spot_price, strike, days_to_expiry, 
                    risk_free_rate, historical_volatility, 'call'
                )
                
                call_greeks = self.calculate_greeks(
                    spot_price, strike, days_to_expiry,
                    risk_free_rate, historical_volatility, 'call'
                )
                
                # Generate simulated volume and OI data
                call_volume = np.random.randint(100, 1000)
                call_oi = np.random.randint(1000, 10000)
                
                calls.append({
                    'strike': strike,
                    'type': 'CE',
                    'expiry': expiry.strftime('%Y-%m-%d'),
                    'ltp': call_price,
                    'iv': historical_volatility * 100,
                    'volume': call_volume,
                    'oi': call_oi,
                    'change': np.random.uniform(-5, 5),
                    'delta': call_greeks['delta'],
                    'gamma': call_greeks['gamma'],
                    'theta': call_greeks['theta'],
                    'vega': call_greeks['vega']
                })
                
                # Calculate put option
                put_price = self.black_scholes(
                    spot_price, strike, days_to_expiry,
                    risk_free_rate, historical_volatility, 'put'
                )
                
                put_greeks = self.calculate_greeks(
                    spot_price, strike, days_to_expiry,
                    risk_free_rate, historical_volatility, 'put'
                )
                
                # Generate simulated volume and OI data
                put_volume = np.random.randint(100, 1000)
                put_oi = np.random.randint(1000, 10000)
                
                puts.append({
                    'strike': strike,
                    'type': 'PE',
                    'expiry': expiry.strftime('%Y-%m-%d'),
                    'ltp': put_price,
                    'iv': historical_volatility * 100,
                    'volume': put_volume,
                    'oi': put_oi,
                    'change': np.random.uniform(-5, 5),
                    'delta': put_greeks['delta'],
                    'gamma': put_greeks['gamma'],
                    'theta': put_greeks['theta'],
                    'vega': put_greeks['vega']
                })
            
            # Combine calls and puts into a DataFrame
            options_df = pd.DataFrame(calls + puts)
            
            # Sort by strike price
            options_df = options_df.sort_values('strike')
            
            return options_df, spot_price
            
        except Exception as e:
            st.error(f"Error fetching options chain: {str(e)}")
            return None, None

if __name__ == "__main__":
    trader = NiftyOptionsAI()
    trader.display_dashboard() 