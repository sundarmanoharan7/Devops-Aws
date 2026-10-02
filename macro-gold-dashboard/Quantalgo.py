import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(page_title="QuantAlgo XAUUSD Engine", layout="wide")
st.title("XAUUSD Trading Desk: Adaptive Supertrend, SMC & Volatility Squeeze")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Strategy Parameters")
timeframe = st.sidebar.selectbox("Timeframe", ["15m", "1h", "4h"], index=1)
atr_period = st.sidebar.number_input("Adaptive Supertrend ATR Period", value=14)
base_mult = st.sidebar.number_input("Adaptive Supertrend Base Multiplier", value=3.0, step=0.5)
smc_lookback = st.sidebar.slider("SMC Swing Lookback", 5, 50, 15)

# --- DATA FETCHING ---
@st.cache_data(ttl=60)
def fetch_data(tf):
    period = "60d" if tf in ["15m", "1h"] else "1y"
    df = yf.download("XAUUSD=X", period=period, interval=tf, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df

# --- QUANTALGO REPLICATION LOGIC ---

def adaptive_supertrend(df, period, base_multiplier):
    """Dynamically adjusts the Supertrend multiplier based on current ATR vs Historical ATR."""
    df = df.copy()
    high, low, close = df['High'], df['Low'], df['Close']
    
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    # Volatility Ratio for Adaptation
    atr_mean = atr.rolling(window=100).mean()
    vol_ratio = (atr / atr_mean).fillna(1)
    dynamic_mult = base_multiplier * vol_ratio.clip(lower=0.5, upper=2.0)
    
    hl2 = (high + low) / 2
    upper_band = hl2 + (dynamic_mult * atr)
    lower_band = hl2 - (dynamic_mult * atr)
    
    supertrend = pd.Series(index=df.index, dtype='float64')
    direction = pd.Series(index=df.index, dtype='int')
    
    for i in range(1, len(df)):
        if close.iloc[i] > upper_band.iloc[i-1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lower_band.iloc[i-1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = direction.iloc[i-1]
            if direction.iloc[i] == 1 and lower_band.iloc[i] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            if direction.iloc[i] == -1 and upper_band.iloc[i] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i-1]
                
        supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]
        
    df['Supertrend'] = supertrend
    df['ST_Direction'] = direction
    return df

def smc_structure(df, window):
    """Maps Smart Money Concepts dealing ranges."""
    highs = argrelextrema(df['High'].values, np.greater, order=window)[0]
    lows = argrelextrema(df['Low'].values, np.less, order=window)[0]
    
    recent_high = df['High'].iloc[highs[-1]] if len(highs) > 0 else df['High'].max()
    recent_low = df['Low'].iloc[lows[-1]] if len(lows) > 0 else df['Low'].min()
    
    if recent_low >= recent_high:
        recent_high, recent_low = df['High'].max(), df['Low'].min()
        
    eq = (recent_high + recent_low) / 2
    return recent_high, recent_low, eq

def volatility_squeezed_momentum(df, length=20, mult=2.0):
    """Replicates a squeeze momentum oscillator (Bollinger Bands vs Keltner Channels)."""
    df = df.copy()
    close = df['Close']
    
    # Bollinger Bands
    basis = close.rolling(window=length).mean()
    dev = mult * close.rolling(window=length).std()
    upper_bb = basis + dev
    lower_bb = basis - dev
    
    # Keltner Channels
    tr = pd.concat([df['High']-df['Low'], (df['High']-close.shift(1)).abs(), (df['Low']-close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=length).mean()
    upper_kc = basis + (mult * atr)
    lower_kc = basis - (mult * atr)
    
    # Squeeze Condition (BB falls inside KC)
    df['Squeeze_On'] = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    
    # Momentum Calculation (Linear Regression of price deviation)
    highest_high = df['High'].rolling(window=length).max()
    lowest_low = df['Low'].rolling(window=length).min()
    avg_price = (highest_high + lowest_low) / 2
    price_diff = close - ((avg_price + basis) / 2)
    
    # Simplified Momentum Oscillator
    df['Momentum'] = price_diff.rolling(window=length).mean()
    return df

# --- DASHBOARD RENDERING ---
df_raw = fetch_data(timeframe)

if not df_raw.empty:
    df = adaptive_supertrend(df_raw, atr_period, base_mult)
    df = volatility_squeezed_momentum(df)
    bsl, ssl, eq = smc_structure(df, smc_lookback)
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Signal Logic Combination
    trend_bullish = last['ST_Direction'] == 1
    trend_bearish = last['ST_Direction'] == -1
    momentum_firing = (last['Momentum'] > 0 and prev['Momentum'] <= 0) or (last['Momentum'] < 0 and prev['Momentum'] >= 0)
    squeeze_firing = prev['Squeeze_On'] and not last['Squeeze_On']
    
    signal = "NEUTRAL"
    if trend_bullish and squeeze_firing and last['Momentum'] > 0 and last['Close'] < eq:
        signal = "BUY (SMC Discount + Volatility Expansion)"
        color = "green"
    elif trend_bearish and squeeze_firing and last['Momentum'] < 0 and last['Close'] > eq:
        signal = "SELL (SMC Premium + Volatility Expansion)"
        color = "red"
    else:
        color = "gray"

    # Execution Metrics
    st.subheader(f"Current Live Spot Price: ${last['Close']:,.2f}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Composite Signal", f":{color}[{signal}]")
    col2.metric("Adaptive Supertrend", "BULLISH" if trend_bullish else "BEARISH")
    col3.metric("Squeeze Status", "COMPRESSING" if last['Squeeze_On'] else "EXPANDING")
    col4.metric("SMC Range (BSL - SSL)", f"${bsl:,.2f} - ${ssl:,.2f}")
    
    st.divider()

    # Visual Mapping
    st.subheader("Price Action, SMC Levels & Adaptive Supertrend")
    df_plot = df.tail(150)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={'height_ratios': [2.5, 1]}, sharex=True)
    
    # Main Chart
    ax1.plot(df_plot.index, df_plot['Close'], color='black', label="Spot Close")
    
    # Supertrend Plotting
    st_bull = df_plot['Supertrend'].where(df_plot['ST_Direction'] == 1)
    st_bear = df_plot['Supertrend'].where(df_plot['ST_Direction'] == -1)
    ax1.plot(df_plot.index, st_bull, color='green', linewidth=2, label="Adaptive ST (Support)")
    ax1.plot(df_plot.index, st_bear, color='red', linewidth=2, label="Adaptive ST (Resistance)")
    
    # SMC Zones
    ax1.axhline(bsl, color='red', linestyle='--', label=f"BSL Liquidity (${bsl:,.2f})")
    ax1.axhline(ssl, color='green', linestyle='--', label=f"SSL Liquidity (${ssl:,.2f})")
    ax1.axhline(eq, color='blue', linestyle=':', label=f"Equilibrium (${eq:,.2f})")
    ax1.axhspan(eq, bsl, color='red', alpha=0.05, label="Premium")
    ax1.axhspan(ssl, eq, color='green', alpha=0.05, label="Discount")
    ax1.set_ylabel("Price (USD)")
    ax1.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    ax1.grid(alpha=0.2)
    
    # Momentum Squeeze Chart
    colors = ['green' if val > 0 else 'red' for val in df_plot['Momentum']]
    ax2.bar(df_plot.index, df_plot['Momentum'], color=colors, alpha=0.7, label="Momentum Oscillator")
    
    squeeze_points = df_plot[df_plot['Squeeze_On']].index
    ax2.scatter(squeeze_points, [0]*len(squeeze_points), color='black', marker='x', label="Squeeze Compression")
    
    ax2.set_ylabel("Momentum")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d - %H:%M'))
    ax2.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
    ax2.grid(alpha=0.2)
    
    st.pyplot(fig)
