import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(page_title="SMC Live Trade Setup", layout="wide")
st.title("Smart Money Concepts (SMC) - Live XAUUSD Setup")

# --- SIDEBAR CONTROLS ---
st.sidebar.header("SMC Engine Settings")
spot_anchor = st.sidebar.number_input(
    "Spot Anchor Price (USD)",
    min_value=1000.0,
    max_value=10000.0,
    value=4265.70,
    step=0.50,
    help="Anchors futures structure to live spot price."
)

lookback_days = st.sidebar.slider(
    "Chart Lookback (Days)",
    min_value=1,
    max_value=10,
    value=5
)

# --- ROBUST SMC DATA FETCHER ---
@st.cache_data(ttl=120)
def fetch_smc_data(anchor_price: float, days: int):
    # Pull 15m futures data (Yahoo reliably serves 15m on GC=F)
    period_str = f"{max(days, 5)}d"
    df = yf.download("GC=F", period=period_str, interval="15m", progress=False)
    
    # Fallback to 1h if 15m is unavailable
    if df.empty:
        df = yf.download("GC=F", period=f"{max(days, 14)}d", interval="1h", progress=False)
        
    if df.empty:
        return pd.DataFrame()
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Standardize timezone to UTC
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
        
    # Spot Calibration Offset
    latest_futures_close = float(df['Close'].dropna().iloc[-1])
    offset = latest_futures_close - anchor_price
    
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = df[col] - offset
        
    return df

def analyze_smc_structure(df, window=12):
    # Detect swing highs and lows
    highs = argrelextrema(df['High'].values, np.greater, order=window)[0]
    lows = argrelextrema(df['Low'].values, np.less, order=window)[0]
    
    if len(highs) == 0 or len(lows) == 0:
        # Fallback to range boundaries if array is too small
        recent_high = float(df['High'].max())
        recent_low = float(df['Low'].min())
    else:
        recent_high = float(df['High'].iloc[highs[-1]])
        recent_low = float(df['Low'].iloc[lows[-1]])
    
    # Mathematical guard: High must be above low
    if recent_low >= recent_high:
        recent_high = float(df['High'].max())
        recent_low = float(df['Low'].min())
        
    # Institutional Fibonacci & Dealing Range Calculation
    total_range = recent_high - recent_low
    equilibrium = recent_high - (total_range * 0.50)
    golden_zone_low = recent_high - (total_range * 0.382)   # 61.8% retracement level
    golden_zone_high = recent_high - (total_range * 0.214)  # 78.6% retracement level
    
    return recent_high, recent_low, equilibrium, golden_zone_low, golden_zone_high

# --- EXECUTION ---
df = fetch_smc_data(spot_anchor, lookback_days)

if not df.empty:
    current_price = float(df['Close'].iloc[-1])
    swing_high, swing_low, eq, gz_low, gz_high = analyze_smc_structure(df)
    
    st.subheader(f"Active Market Price: ${current_price:,.2f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Institutional Dealing Range")
        st.write(f"**Swing High (Buy-Side Liquidity):** ${swing_high:,.2f}")
        st.write(f"**Equilibrium (50% Fair Value):** ${eq:,.2f}")
        st.write(f"**Swing Low (Sell-Side Liquidity):** ${swing_low:,.2f}")
        
        status = "PREMIUM (Sell Allowed)" if current_price > eq else "DISCOUNT (Wait for Retracement)"
        badge_color = "red" if current_price > eq else "green"
        st.markdown(f"**Market Valuation:** :{badge_color}[{status}]")

    with col2:
        st.markdown("### SMC Execution Plan (Bearish Bias)")
        st.write(f"**Optimal Entry Zone (Order Block):** ${gz_low:,.2f} – ${gz_high:,.2f}")
        st.write(f"**Stop Loss (Hard Invalidation):** ${swing_high + 2.50:,.2f}")
        st.write(f"**Take Profit (Target Liquidity):** ${swing_low:,.2f}")
        
    st.divider()
    
    # --- CHART VISUALIZATION ---
    st.subheader("15-Minute SMC Market Structure")
    
    # Filter by user lookback
    cutoff_time = df.index.max() - pd.Timedelta(days=lookback_days)
    df_chart = df[df.index >= cutoff_time]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_chart.index, df_chart['Close'], color='black', linewidth=1.2, label="XAUUSD Spot")
    
    # Zone Shading
    ax.axhspan(eq, swing_high, color='red', alpha=0.06, label="Premium Zone (Supply Favor)")
    ax.axhspan(swing_low, eq, color='green', alpha=0.06, label="Discount Zone (Demand Favor)")
    ax.axhspan(gz_low, gz_high, color='darkred', alpha=0.25, label="Optimal Trade Entry (OTE Supply)")
    
    # Boundary Lines
    ax.axhline(swing_high, color='red', linestyle='--', linewidth=1.8, label=f"BSL Stop (${swing_high:,.2f})")
    ax.axhline(swing_low, color='green', linestyle='--', linewidth=1.8, label=f"SSL Target (${swing_low:,.2f})")
    ax.axhline(eq, color='blue', linestyle=':', linewidth=1.4, label=f"Equilibrium (${eq:,.2f})")
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M UTC'))
    ax.set_ylabel('Spot Price (USD)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1))
    ax.grid(alpha=0.25)
    
    st.pyplot(fig)
else:
    st.error("Market data feeds are currently unreachable. Verify network or API availability.")
