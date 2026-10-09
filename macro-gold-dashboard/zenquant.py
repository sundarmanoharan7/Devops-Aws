import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from datetime import datetime
import pytz

st.set_page_config(page_title="Zeno Gold Replica | XAUUSD", layout="wide")
st.title("Gold (XAUUSD) — Smart Entry & Session Scalping Strategy")

# --- REAL-TIME SPOT TICKER ---
@st.cache_data(ttl=5)
def get_live_spot(df_fallback):
    """Fetches real-time ticking spot price."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = "https://scanner.tradingview.com/cfd/scan"
        payload = {"symbols": {"tickers": ["FXCM:XAUUSD", "OANDA:XAUUSD"]}, "columns": ["close"]}
        res = requests.post(url, json=payload, headers=headers, timeout=3)
        if res.status_code == 200:
            tv_price = res.json().get('data', [])[0].get('d', [0])[0]
            if tv_price > 1000: return float(tv_price)
    except: pass
    return float(df_fallback['Close'].iloc[-1])

# --- DATA INGESTION & TIMEFRAME SYNC ---
@st.cache_data(ttl=60)
def fetch_data():
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    
    # Fetch 15m and 1H data
    d_1h = yf.download("XAUUSD=X", period="6mo", interval="1h", progress=False, session=session)
    d_15m = yf.download("XAUUSD=X", period="14d", interval="15m", progress=False, session=session)
    
    if isinstance(d_1h.columns, pd.MultiIndex): d_1h.columns = d_1h.columns.get_level_values(0)
    if isinstance(d_15m.columns, pd.MultiIndex): d_15m.columns = d_15m.columns.get_level_values(0)
        
    df_1h = d_1h.dropna()
    df_15m = d_15m.dropna()
    
    # Resample 1H to 4H for Macro Trend Bias
    df_4h = df_1h.resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
    
    # Standardize to IST Timezone
    for df in [df_15m, df_1h, df_4h]:
        if df.index.tz is None: 
            df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
        else: 
            df.index = df.index.tz_convert('Asia/Kolkata')
            
    return df_15m, df_4h

# --- STRATEGY LOGIC ENGINE ---
def apply_zeno_logic(df_15, df_4h):
    df = df_15.copy()
    
    # 1. Macro Trend (4H Bias)
    df_4h['EMA_50'] = df_4h['Close'].ewm(span=50).mean()
    bias_4h = "BULLISH" if df_4h['Close'].iloc[-1] > df_4h['EMA_50'].iloc[-1] else "BEARISH"
    
    # 2. Volatility (ATR) for Dynamic Stops/Targets
    df['High-Low'] = df['High'] - df['Low']
    df['High-PrevClose'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-PrevClose'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()
    
    # 3. Dynamic Baseline (SMA Retest Zone)
    df['SMA_20'] = df['Close'].rolling(20).mean()
    
    # 4. Session Filter (Skip Asian Session)
    # Asian Session roughly 02:30 to 11:30 IST. London opens ~12:30 PM IST.
    current_hour = datetime.now(pytz.timezone('Asia/Kolkata')).hour
    is_active_session = current_hour >= 12 and current_hour <= 22 # Active London/NY
    
    return df, bias_4h, is_active_session

# --- EXECUTION & DASHBOARD ---
df_15m, df_4h = fetch_data()
df_15m_logic, trend_4h, is_active_session = apply_zeno_logic(df_15m, df_4h)

live_spot = get_live_spot(df_15m_logic)
last_candle = df_15m_logic.iloc[-2] # Lock to closed candle
current_atr = last_candle['ATR']
baseline = last_candle['SMA_20']

# Smart Entry Trigger Logic (Pullback Retest)
signal = "IDLE (Waiting for Setup)"
color = "gray"

# Check if price is pulling back to retest the 20 SMA baseline
is_retesting = abs(live_spot - baseline) <= (current_atr * 0.2) 

if is_active_session and is_retesting:
    if trend_4h == "BULLISH" and live_spot >= baseline:
        signal = "LONG (Smart Entry Retest)"
        color = "green"
    elif trend_4h == "BEARISH" and live_spot <= baseline:
        signal = "SHORT (Smart Entry Retest)"
        color = "red"
elif not is_active_session:
    signal = "SKIPPED (Asian Consolidation)"
    color = "orange"

# UI Rendering
col1, col2, col3 = st.columns(3)
col1.metric("Live XAUUSD Spot", f"${live_spot:,.2f}")
col2.metric("Execution Status", f":{color}[{signal}]")
col3.metric("Macro Trend (4H)", trend_4h)

st.divider()

# --- LIVE TRADE TICKET (ATR RISK ENGINE) ---
st.markdown("### 📝 Smart Entry Trade Ticket")

if "LONG" in signal:
    entry = live_spot
    sl = entry - (current_atr * 1.5)
    tp1 = entry + (current_atr * 1.5) # 1:1 RR -> Move to Break Even
    tp2 = entry + (current_atr * 3.0) # Full Exit
elif "SHORT" in signal:
    entry = live_spot
    sl = entry + (current_atr * 1.5)
    tp1 = entry - (current_atr * 1.5) 
    tp2 = entry - (current_atr * 3.0) 
else:
    entry = baseline
    sl, tp1, tp2 = 0, 0, 0

t1, t2, t3, t4 = st.columns(4)
t1.metric("Dynamic Retest Entry", f"${entry:,.2f}")
t2.metric("Stop Loss (1.5x ATR)", f"${sl:,.2f}" if sl else "N/A")
t3.metric("TP1 (50% Close + BE)", f"${tp1:,.2f}" if tp1 else "N/A")
t4.metric("TP2 (Full Exit)", f"${tp2:,.2f}" if tp2 else "N/A")

st.divider()

# --- PIPELINE STATUS ---
st.markdown("### 🔍 Logic Pipeline Validation")
v1, v2, v3 = st.columns(3)

if is_active_session: v1.success("✅ Session Filter\nLondon/NY Active")
else: v1.warning("⏳ Session Filter\nSkipping Asian Chop")

v2.info(f"⚡ Macro Trend\nAligned {trend_4h}")

if is_retesting: v3.success("✅ Pullback Retest\nPrice at Baseline")
else: v3.warning("⏳ Pullback Retest\nAwaiting Mean Reversion")

# --- CHART RENDERING ---
st.subheader("15M Retest Chart & Baseline Envelope")
df_plot = df_15m_logic.tail(100)
fig, ax = plt.subplots(figsize=(14, 5))

# Plot Price and Dynamic SMA Baseline
ax.plot(df_plot.index, df_plot['Close'], color='black', linewidth=1.5, label='XAUUSD Price')
ax.plot(df_plot.index, df_plot['SMA_20'], color='orange', linewidth=2, alpha=0.7, label='Dynamic Entry Baseline (SMA 20)')

# Highlight Current Live Spot
ax.axhline(live_spot, color='blue', linestyle=':', label=f"Live Spot ${live_spot:,.2f}")

if sl:
    ax.axhline(sl, color='red', linestyle='--', label=f"Dynamic Stop Loss")
    ax.axhline(tp1, color='green', linestyle='--', label=f"TP1 (+ BE)")
    ax.axhline(tp2, color='darkgreen', linestyle='-', linewidth=2, label=f"TP2 Full Exit")

ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d - %H:%M\nIST'))
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
ax.grid(alpha=0.2)
st.pyplot(fig)
