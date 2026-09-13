import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

st.set_page_config(page_title="XAUUSD Structural Levels", layout="wide")
st.title("Gold (XAUUSD) Macro & Execution Zones")

@st.cache_data(ttl=3600)
def fetch_gold_data(period="30d"):
    df_1h = yf.download('GC=F', period=period, interval='1h', progress=False)
    
    # Handle empty data if Yahoo Finance API fails or times out
    if df_1h.empty:
        st.error("Failed to fetch data from Yahoo Finance. The API might be down.")
        return pd.DataFrame(), pd.DataFrame()

    # Flatten MultiIndex columns (yfinance update fix)
    if isinstance(df_1h.columns, pd.MultiIndex):
        df_1h.columns = df_1h.columns.get_level_values(0)

    # Resample 1H data to create accurate 4H structural candles
    df_4h = df_1h.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    
    return df_1h, df_4h

def find_structural_levels(df, window):
    maxima_indices = argrelextrema(df['High'].values, np.greater, order=window)[0]
    resistances = df['High'].iloc[maxima_indices].values
    minima_indices = argrelextrema(df['Low'].values, np.less, order=window)[0]
    supports = df['Low'].iloc[minima_indices].values
    return supports, resistances

# Fetch the data
df_1h, df_4h = fetch_gold_data()

if not df_1h.empty:
    supports_all, resistances_all = find_structural_levels(df_4h, window=5)
    
    # Identify Live Price
    live_price = df_4h['Close'].iloc[-1]
    
    # Filter and sort levels relative to live price
    # Remove duplicates by converting to a set, then back to a sorted list
    res_above = sorted([r for r in set(resistances_all) if r > live_price])
    sup_below = sorted([s for s in set(supports_all) if s < live_price], reverse=True)
    
    # Assign specific levels (if they exist in the dataset)
    res_1 = res_above[0] if len(res_above) > 0 else None
    sup_minor = sup_below[0] if len(sup_below) > 0 else None
    sup_maj1 = sup_below[1] if len(sup_below) > 1 else None
    sup_maj2 = sup_below[2] if len(sup_below) > 2 else None

    # --- TEXT OUTPUT FORMATTING ---
    st.subheader("Key Institutional Levels")
    st.write(f"**Live Price:** ~${live_price:,.2f}")
    
    st.write("**Resistance (Supply Zones)**")
    if res_1: 
        st.write(f"1st Resistance Point: ${res_1:,.2f}")
    else:
        st.write("1st Resistance Point: N/A (Trading at recent highs)")
        
    st.write("**Support (Demand Zones)**")
    if sup_minor: st.write(f"Minor Support: ${sup_minor:,.2f}")
    if sup_maj1: st.write(f"Major Support 1: ${sup_maj1:,.2f}")
    if sup_maj2: st.write(f"Major Support 2: ${sup_maj2:,.2f}")
    
    st.divider()

    # --- CHART VISUALIZATION ---
    st.subheader("4-Hour Macro Structure")
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(df_4h.index, df_4h['Close'], label='H4 Close Price', color='black', linewidth=1.5)
    
    # Plot only the filtered, relevant levels with distinct colors and weights
    if res_1:
        ax.axhline(res_1, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Supply (Resistance)')
    if sup_minor:
        ax.axhline(sup_minor, color='lightgreen', linestyle='--', alpha=0.9, linewidth=1.5, label='Minor Support')
    if sup_maj1:
        ax.axhline(sup_maj1, color='green', linestyle='-', alpha=0.7, linewidth=2, label='Major Support 1')
    if sup_maj2:
        ax.axhline(sup_maj2, color='darkgreen', linestyle='-', alpha=0.9, linewidth=2.5, label='Major Support 2')
    
    ax.set_title('XAUUSD 4-Hour Structural Support & Resistance')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    
    # Place legend outside or neatly in the corner to avoid blocking price action
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(alpha=0.2)
    
    st.pyplot(fig)
