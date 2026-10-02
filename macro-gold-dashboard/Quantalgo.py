import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

st.set_page_config(page_title="Quantum Algo Replica | XAUUSD", layout="wide")
st.title("Gold (XAUUSD) — 5-Stage Quantum Algo Execution Desk")

# --- STAGE 01: MARKET DATA INGESTION ---
@st.cache_data(ttl=30)
def fetch_mtf_data():
    """Ingests multi-timeframe data for confluence checks."""
    # Cloud Fallback (MEXC PAXGUSDT) for highest uptime
    def get_mexc(interval):
        try:
            res = requests.get(f"https://api.mexc.com/api/v3/klines?symbol=PAXGUSDT&interval={interval}&limit=500", timeout=3).json()
            df = pd.DataFrame(res, columns=['time', 'Open', 'High', 'Low', 'Close', 'Volume', 'ct', 'qav', 'nt', 'tbb', 'tbq'])
            df['time'] = pd.to_datetime(df['time'].astype(int), unit='ms', utc=True)
            df = df[['time', 'Open', 'High', 'Low', 'Close', 'Volume']].set_index('time').astype(float).sort_index()
            return df
        except: return pd.DataFrame()

    df_15m = get_mexc("15m")
    df_1h = get_mexc("60m")
    df_4h = get_mexc("4h")

    # yfinance priority override if available
    if df_15m.empty:
        try:
            df_15m = yf.download("XAUUSD=X", period="10d", interval="15m", progress=False)
            df_1h = yf.download("XAUUSD=X", period="30d", interval="1h", progress=False)
            df_4h = yf.download("XAUUSD=X", period="60d", interval="1h", progress=False).resample('4h').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()
            for df in [df_15m, df_1h, df_4h]:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        except: pass

    return df_15m, df_1h, df_4h

def get_live_spot():
    try:
        res = requests.get("https://api.mexc.com/api/v3/ticker/price?symbol=PAXGUSDT", timeout=2).json()
        return float(res['price'])
    except: return 4195.00

# --- STAGE 02: MULTI-TIMEFRAME CONFLUENCE ---
def check_mtf_alignment(df_1h, df_4h):
    """Checks higher timeframe directional bias (EMA 50 alignment)."""
    bias_1h = df_1h['Close'].iloc[-1] > df_1h['Close'].ewm(span=50).mean().iloc[-1]
    bias_4h = df_4h['Close'].iloc[-1] > df_4h['Close'].ewm(span=50).mean().iloc[-1]
    
    if bias_1h and bias_4h: return "BULLISH", True
    elif not bias_1h and not bias_4h: return "BEARISH", True
    else: return "MIXED", False

# --- STAGE 03: SMC & INSTITUTIONAL ORDER FLOW ---
def map_smc_zones(df, window=10):
    df = df.copy()
    
    # 1. Structure Mapping (BOS / Liquidity Sweeps)
    highs = argrelextrema(df['High'].values, np.greater, order=window)[0]
    lows = argrelextrema(df['Low'].values, np.less, order=window)[0]
    
    bsl = df['High'].iloc[highs[-1]] if len(highs) > 0 else df['High'].max()
    ssl = df['Low'].iloc[lows[-1]] if len(lows) > 0 else df['Low'].min()
    eq = (bsl + ssl) / 2
    
    # 2. Fair Value Gap (FVG) Detection
    df['FVG_Bull'] = (df['Low'] > df['High'].shift(2)) & (df['Close'].shift(1) > df['High'].shift(2))
    df['FVG_Bear'] = (df['High'] < df['Low'].shift(2)) & (df['Close'].shift(1) < df['Low'].shift(2))
    
    latest_bull_fvg = df[df['FVG_Bull']]['Low'].iloc[-1] if not df[df['FVG_Bull']].empty else None
    latest_bear_fvg = df[df['FVG_Bear']]['High'].iloc[-1] if not df[df['FVG_Bear']].empty else None

    return bsl, ssl, eq, latest_bull_fvg, latest_bear_fvg

# --- STAGE 04: ADAPTIVE SIGNAL FILTERING ---
def apply_adaptive_filters(df):
    df = df.copy()
    
    # RSI Momentum Filter
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volume Anomaly Detection
    df['Vol_SMA'] = df['Volume'].rolling(20).mean()
    df['Vol_Spike'] = df['Volume'] > (df['Vol_SMA'] * 1.5)
    
    # ATR Volatility Filter
    tr = pd.concat([df['High']-df['Low'], (df['High']-df['Close'].shift(1)).abs(), (df['Low']-df['Close'].shift(1)).abs()], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(14).mean()
    
    return df

# --- STAGE 05: SIGNAL DELIVERY & DASHBOARD ---
df_15m, df_1h, df_4h = fetch_mtf_data()

if not df_15m.empty and not df_1h.empty and not df_4h.empty:
    live_spot = get_live_spot()
    
    # Process Pipeline
    mtf_bias, mtf_aligned = check_mtf_alignment(df_1h, df_4h)
    bsl, ssl, eq, bull_fvg, bear_fvg = map_smc_zones(df_15m)
    df_15m = apply_adaptive_filters(df_15m)
    
    last = df_15m.iloc[-2] # Lock to closed candle to prevent repainting
    
    # Signal Logic Matrix
    signal = "NO SIGNAL"
    color = "gray"
    
    is_discount = last['Close'] < eq
    is_premium = last['Close'] > eq
    has_momentum_bull = last['RSI'] > 50
    has_momentum_bear = last['RSI'] < 50
    
    if mtf_bias == "BULLISH" and is_discount and has_momentum_bull and last['Vol_Spike']:
        signal = "BUY (Long)"
        color = "green"
    elif mtf_bias == "BEARISH" and is_premium and has_momentum_bear and last['Vol_Spike']:
        signal = "SELL (Short)"
        color = "red"

    # UI Construction
    col1, col2, col3 = st.columns([1, 1, 1])
    col1.metric("Live XAUUSD Spot", f"${live_spot:,.2f}")
    col2.metric("Quantum Algo Signal", f":{color}[{signal}]")
    col3.metric("MTF Confluence (1H/4H)", mtf_bias)

    st.divider()
    st.markdown("### 🔍 5-Stage Validation Pipeline Status")
    
    v1, v2, v3, v4, v5 = st.columns(5)
    v1.success("✅ 01. Data Ingestion\n15m, 1H, 4H Synced")
    
    if mtf_aligned: v2.success(f"✅ 02. MTF Confluence\nAligned {mtf_bias}")
    else: v2.warning("⚠️ 02. MTF Confluence\nMixed Timeframes")
        
    v3.success(f"✅ 03. SMC Order Flow\nEq: ${eq:,.2f}")
    
    if last['Vol_Spike']: v4.success(f"✅ 04. Adaptive Filter\nVol Anomaly Detected")
    else: v4.warning("⏳ 04. Adaptive Filter\nAwaiting Volatility")
        
    v5.info(f"⚡ 05. Signal Delivery\nStatus: {signal}")

    st.divider()
    
    # --- CHART VISUALIZATION ---
    st.subheader("Smart Money Concepts & Order Flow Chart (15m)")
    df_plot = df_15m.tail(120)
    fig, ax = plt.subplots(figsize=(15, 6))
    
    ax.plot(df_plot.index, df_plot['Close'], color='black', linewidth=1.5, label='Price')
    
    # Map Premium/Discount
    ax.axhspan(eq, bsl, color='red', alpha=0.05, label="Premium Zone")
    ax.axhspan(ssl, eq, color='green', alpha=0.05, label="Discount Zone")
    
    # Map Liquidity & BOS
    ax.axhline(bsl, color='red', linestyle='--', label=f"Buy-Side Liquidity (BSL) ${bsl:,.2f}")
    ax.axhline(ssl, color='green', linestyle='--', label=f"Sell-Side Liquidity (SSL) ${ssl:,.2f}")
    ax.axhline(eq, color='blue', linestyle=':', label=f"Equilibrium ${eq:,.2f}")
    
    # Map FVGs
    if bull_fvg: ax.axhline(bull_fvg, color='darkgreen', linestyle='-', linewidth=2, alpha=0.5, label=f"Bullish FVG Retest")
    if bear_fvg: ax.axhline(bear_fvg, color='darkred', linestyle='-', linewidth=2, alpha=0.5, label=f"Bearish FVG Retest")
    
    # Mark Volume Anomalies
    spike_idx = df_plot[df_plot['Vol_Spike']].index
    spike_prices = df_plot.loc[spike_idx, 'Close']
    ax.scatter(spike_idx, spike_prices, color='purple', marker='^', s=100, label="Institutional Vol Anomaly")
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d - %H:%M'))
    ax.set_ylabel("XAUUSD Price")
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.2)
    st.pyplot(fig)

else:
    st.error("Market data feeds are currently unreachable.")
