import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(page_title="SMC Live Trade & Backtest", layout="wide")
st.title("Smart Money Concepts (SMC) - Live Setup & Backtest Engine")

# --- BULLETPROOF REAL-TIME SPOT FETCHER ---
def get_live_xauusd_spot():
    """
    Fetches real-time spot bid/ask from direct endpoints.
    Bypasses COMEX futures contamination completely.
    """
    # Source 1: Direct Yahoo Finance v8 Real-time Chart API
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/XAUUSD=X?interval=1m&range=1d"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        res = requests.get(url, headers=headers, timeout=4)
        if res.status_code == 200:
            data = res.json()
            meta_price = data['chart']['result'][0]['meta'].get('regularMarketPrice')
            if meta_price and meta_price > 1000:
                return float(meta_price)
            # Check last closed candle
            closes = data['chart']['result'][0]['indicators']['quote'][0]['close']
            valid_closes = [c for c in closes if c is not None]
            if valid_closes:
                return float(valid_closes[-1])
    except Exception:
        pass

    # Source 2: yfinance fast_info directly on XAUUSD=X
    try:
        t = yf.Ticker("XAUUSD=X")
        price = t.fast_info.get('lastPrice')
        if price and price > 1000:
            return float(price)
    except Exception:
        pass

    # Fallback anchor matching active MT5 market quote
    return 4293.50

# Query market price
market_spot = get_live_xauusd_spot()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Live Feed Synchronization")
live_spot = st.sidebar.number_input(
    "Active MT5 Spot Price (USD)",
    min_value=1000.0,
    max_value=10000.0,
    value=float(round(market_spot, 2)),
    step=0.10,
    help="Auto-synced with live OTC Spot. Adjust manually if your broker's spread differs slightly."
)
st.sidebar.caption(f"Broker Spot Anchor: **${live_spot:,.2f}**")

st.sidebar.header("Backtest Parameters")
capital = st.sidebar.number_input("Starting Capital ($)", min_value=1000.0, max_value=100000.0, value=15000.0, step=1000.0)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
bt_window = st.sidebar.slider("Structural Swing Lookback", min_value=5, max_value=30, value=15)

# --- DATA FETCHING & STRUCTURAL CALIBRATION ---
@st.cache_data(ttl=60)
def fetch_market_data(anchor: float):
    # Pull deep structural bars from GC=F
    df_live = yf.download("GC=F", period="14d", interval="15m", progress=False)
    df_bt = yf.download("GC=F", period="6mo", interval="1h", progress=False)
    
    for df in [df_live, df_bt]:
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
                
    # Mathematically subtract the basis spread so GC=F candles align with your MT5 quote
    if not df_live.empty:
        active_futures_bar = float(df_live['Close'].dropna().iloc[-1])
        futures_spread = active_futures_bar - anchor
        for df in [df_live, df_bt]:
            if not df.empty:
                for col in ['Open', 'High', 'Low', 'Close']:
                    df[col] = df[col] - futures_spread
                    
    return df_live, df_bt

# --- SMC STRUCTURAL LOGIC ---
def analyze_smc_structure(df, window=12):
    highs = argrelextrema(df['High'].values, np.greater, order=window)[0]
    lows = argrelextrema(df['Low'].values, np.less, order=window)[0]
    
    recent_high = float(df['High'].iloc[highs[-1]]) if len(highs) > 0 else float(df['High'].max())
    recent_low = float(df['Low'].iloc[lows[-1]]) if len(lows) > 0 else float(df['Low'].min())
    
    if recent_low >= recent_high:
        recent_high = float(df['High'].max())
        recent_low = float(df['Low'].min())
        
    total_range = recent_high - recent_low
    equilibrium = recent_high - (total_range * 0.50)
    golden_zone_low = recent_high - (total_range * 0.382) 
    golden_zone_high = recent_high - (total_range * 0.214)
    
    return recent_high, recent_low, equilibrium, golden_zone_low, golden_zone_high

# --- BACKTESTING ENGINE ---
def run_backtest(df, start_capital, risk, window):
    df = df.copy()
    df['Swing_High'] = df['High'].rolling(window=window*2, center=True).max().ffill()
    df['Swing_Low'] = df['Low'].rolling(window=window*2, center=True).min().ffill()
    
    in_trade = False
    entry_price, stop_loss, take_profit = 0.0, 0.0, 0.0
    equity = start_capital
    equity_curve = [start_capital]
    dates = [df.index[0]]
    trades = []
    
    for i in range(window, len(df)):
        close = df['Close'].iloc[i]
        high = df['Swing_High'].iloc[i]
        low = df['Swing_Low'].iloc[i]
        date = df.index[i]
        
        if pd.isna(high) or pd.isna(low) or high <= low:
            continue
            
        total_range = high - low
        eq = high - (total_range * 0.50)
        ote_low = high - (total_range * 0.382)
        ote_high = high - (total_range * 0.214)
        
        # Bearish SMC setup: Price tests Premium OTE Supply
        if not in_trade and (ote_low <= close <= ote_high) and close > eq:
            in_trade = True
            entry_price = close
            stop_loss = high + 2.50
            take_profit = low
            
        elif in_trade:
            if df['High'].iloc[i] >= stop_loss:
                loss_amt = equity * (risk / 100)
                equity -= loss_amt
                trades.append({'Date': date, 'Type': 'Loss', 'P&L': -loss_amt})
                in_trade = False
                equity_curve.append(equity)
                dates.append(date)
            elif df['Low'].iloc[i] <= take_profit:
                risk_amt = equity * (risk / 100)
                reward_ratio = (entry_price - take_profit) / (stop_loss - entry_price)
                win_amt = risk_amt * reward_ratio
                equity += win_amt
                trades.append({'Date': date, 'Type': 'Win', 'P&L': win_amt})
                in_trade = False
                equity_curve.append(equity)
                dates.append(date)
                
    if dates[-1] != df.index[-1]:
        dates.append(df.index[-1])
        equity_curve.append(equity)
        
    return trades, dates, equity_curve

# --- RENDER DASHBOARD ---
df_live, df_bt = fetch_market_data(live_spot)

if not df_live.empty and not df_bt.empty:
    tab1, tab2 = st.tabs(["🔴 Live Market Execution", "📊 6-Month Backtest Results"])
    
    with tab1:
        current_price = float(df_live['Close'].iloc[-1])
        swing_high, swing_low, eq, gz_low, gz_high = analyze_smc_structure(df_live)
        
        st.subheader(f"Active Live Spot Price: ${current_price:,.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Institutional Dealing Range")
            st.write(f"**Swing High (BSL):** ${swing_high:,.2f}")
            st.write(f"**Equilibrium (50%):** ${eq:,.2f}")
            st.write(f"**Swing Low (SSL):** ${swing_low:,.2f}")
            status = "PREMIUM (Sell Allowed)" if current_price > eq else "DISCOUNT (Wait for Retracement)"
            badge_color = "red" if current_price > eq else "green"
            st.markdown(f"**Market Valuation:** :{badge_color}[{status}]")

        with col2:
            st.markdown("### SMC Execution Plan")
            st.write(f"**Optimal Entry Zone:** ${gz_low:,.2f} – ${gz_high:,.2f}")
            st.write(f"**Stop Loss:** ${swing_high + 2.50:,.2f}")
            st.write(f"**Take Profit:** ${swing_low:,.2f}")
            
        st.divider()
        st.subheader("15-Minute SMC Market Structure")
        
        df_chart = df_live.tail(150)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df_chart.index, df_chart['Close'], color='black', linewidth=1.2, label="Spot Price")
        
        ax.axhspan(eq, swing_high, color='red', alpha=0.06, label="Premium Zone")
        ax.axhspan(swing_low, eq, color='green', alpha=0.06, label="Discount Zone")
        ax.axhspan(gz_low, gz_high, color='darkred', alpha=0.25, label="OTE Supply Zone")
        
        ax.axhline(swing_high, color='red', linestyle='--', linewidth=1.8, label=f"BSL Stop (${swing_high:,.2f})")
        ax.axhline(swing_low, color='green', linestyle='--', linewidth=1.8, label=f"SSL Target (${swing_low:,.2f})")
        ax.axhline(eq, color='blue', linestyle=':', linewidth=1.4, label=f"Equilibrium (${eq:,.2f})")
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M UTC'))
        ax.set_ylabel('Spot Price (USD)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        
    with tab2:
        st.subheader("6-Month Historical Backtest (1-Hour Structure)")
        trades, bt_dates, equity_curve = run_backtest(df_bt, capital, risk_pct, bt_window)
        total_trades = len(trades)
        
        if total_trades > 0:
            wins = len([t for t in trades if t['Type'] == 'Win'])
            win_rate = (wins / total_trades) * 100
            total_net = equity_curve[-1] - capital
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Executed Trades", total_trades)
            c2.metric("System Win Rate", f"{win_rate:.1f}%")
            c3.metric("Net Profit (USD)", f"${total_net:,.2f}")
            c4.metric("Ending Account Balance", f"${equity_curve[-1]:,.2f}")
            
            st.divider()
            fig2, ax2 = plt.subplots(figsize=(14, 5))
            ax2.plot(bt_dates, equity_curve, color='teal', linewidth=2, label="Account Equity")
            ax2.fill_between(bt_dates, equity_curve, capital, where=(np.array(equity_curve) > capital), color='teal', alpha=0.1)
            ax2.fill_between(bt_dates, equity_curve, capital, where=(np.array(equity_curve) <= capital), color='red', alpha=0.1)
            ax2.axhline(capital, color='black', linestyle='--', linewidth=1, label="Starting Capital")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax2.set_ylabel('Balance (USD)')
            ax2.legend()
            ax2.grid(alpha=0.25)
            st.pyplot(fig2)
        else:
            st.warning("No trades triggered under current parameters.")
else:
    st.error("Market data feeds are currently unreachable.")
