import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(page_title="SMC Live Trade Setup", layout="wide")
st.title("Smart Money Concepts (SMC) - Live XAUUSD Setup")

@st.cache_data(ttl=60)
def fetch_smc_data():
    # Fetch 15-Minute intraday data for precise SMC order block detection
    df = yf.download("XAUUSD=X", period="10d", interval="15m", progress=False)
    if df.empty:
        return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
        
    return df

def analyze_smc_structure(df, window=15):
    # Locate Swing Highs and Lows
    highs = argrelextrema(df['High'].values, np.greater, order=window)[0]
    lows = argrelextrema(df['Low'].values, np.less, order=window)[0]
    
    recent_high_idx = highs[-1] if len(highs) > 0 else 0
    recent_low_idx = lows[-1] if len(lows) > 0 else 0
    
    recent_high = float(df['High'].iloc[recent_high_idx])
    recent_low = float(df['Low'].iloc[recent_low_idx])
    
    # Ensure High is mathematically above Low for the dealing range
    if recent_low > recent_high:
        recent_high = float(df['High'].max())
        
    # Calculate Fibonacci / Premium vs Discount Zones
    range_distance = recent_high - recent_low
    equilibrium = recent_high - (range_distance * 0.50)
    golden_zone_low = recent_high - (range_distance * 0.382) # 61.8% retracement up from low
    golden_zone_high = recent_high - (range_distance * 0.214) # 78.6% retracement up from low
    
    return recent_high, recent_low, equilibrium, golden_zone_low, golden_zone_high, df.index[recent_high_idx], df.index[recent_low_idx]

# --- EXECUTION ---
df = fetch_smc_data()

if not df.empty:
    live_price = float(df['Close'].iloc[-1])
    
    # Run SMC Analysis
    swing_high, swing_low, eq, gz_low, gz_high, time_high, time_low = analyze_smc_structure(df, window=20)
    
    st.subheader(f"Live Market Price: ${live_price:,.2f}")
    
    # --- TRADE SETUP GENERATOR ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Institutional Dealing Range")
        st.write(f"**Swing High (Buy-Side Liquidity):** ${swing_high:,.2f}")
        st.write(f"**Equilibrium (The 50% Fair Value):** ${eq:,.2f}")
        st.write(f"**Swing Low (Sell-Side Liquidity):** ${swing_low:,.2f}")
        
        st.info("SMC Rule: Never short in a 'Discount' (below Equilibrium). Wait for price to pull back into a 'Premium' (above Equilibrium) to trap retail buyers before selling.")

    with col2:
        st.markdown("### The Trading Plan (Bearish Bias)")
        st.write(f"**1. Wait for Entry Zone (Premium Supply):** ${gz_low:,.2f} to ${gz_high:,.2f}")
        st.write(f"**2. Set Stop Loss (Invalidation):** ${swing_high + 2.00:,.2f} (Just above the Swing High)")
        st.write(f"**3. Take Profit Target (Liquidity Sweep):** ${swing_low:,.2f} (Aiming for the Sell-Side Liquidity pool)")
        
    st.divider()
    
    # --- VISUALIZATION ---
    st.subheader("15-Minute SMC Chart Execution")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(df.index, df['Close'], color='black', linewidth=1.2, label="Live Price Action")
    
    # Highlight Premium & Discount
    ax.axhspan(eq, swing_high, color='red', alpha=0.05, label="Premium Zone (Look for Shorts)")
    ax.axhspan(swing_low, eq, color='green', alpha=0.05, label="Discount Zone (Avoid Shorts Here)")
    
    # Highlight the Entry Order Block Zone (Golden Fibonacci Pocket)
    ax.axhspan(gz_low, gz_high, color='darkred', alpha=0.2, label="High-Probability Entry Zone (Supply)")
    
    # Plot Boundaries
    ax.axhline(swing_high, color='red', linestyle='--', linewidth=2, label="Buy-Side Liquidity (Stop Loss)")
    ax.axhline(swing_low, color='green', linestyle='--', linewidth=2, label="Sell-Side Liquidity (Target)")
    ax.axhline(eq, color='blue', linestyle=':', linewidth=1.5, label="Equilibrium (50%)")
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
    ax.set_ylabel('Spot Price (USD)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.grid(alpha=0.3)
    
    st.pyplot(fig)
else:
    st.error("Market data unavailable. Please try again.")
