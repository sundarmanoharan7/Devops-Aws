import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

# --- PAGE CONFIG ---
st.set_page_config(page_title="XAUUSD Multi-Timeframe Levels", layout="wide")
st.title("Gold (XAUUSD) Multi-Timeframe Structural Levels")

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def fetch_gold_data():
    # 1. Fetch 30 days of 1-Hour data for 1H and 4H structures
    df_1h = yf.download('GC=F', period="30d", interval='1h', progress=False)
    
    # 2. Fetch 1 Year of Daily data for macro D1 structure
    df_1d = yf.download('GC=F', period="1y", interval='1d', progress=False)
    
    if df_1h.empty or df_1d.empty:
        st.error("Failed to fetch data from Yahoo Finance. The API might be temporarily unavailable.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Flatten MultiIndex columns if present (yfinance update fix)
    if isinstance(df_1h.columns, pd.MultiIndex):
        df_1h.columns = df_1h.columns.get_level_values(0)
    if isinstance(df_1d.columns, pd.MultiIndex):
        df_1d.columns = df_1d.columns.get_level_values(0)

    # Resample 1-hour candles into 4-hour structural bars
    df_4h = df_1h.resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    return df_1h, df_4h, df_1d

# --- LEVEL DETECTION ---
def find_structural_levels(df, window=5):
    maxima_indices = argrelextrema(df['High'].values, np.greater, order=window)[0]
    resistances = df['High'].iloc[maxima_indices].values
    
    minima_indices = argrelextrema(df['Low'].values, np.less, order=window)[0]
    supports = df['Low'].iloc[minima_indices].values
    
    return supports, resistances

def extract_key_levels(supports_all, resistances_all, live_price):
    res_above = sorted([r for r in set(resistances_all) if r > live_price])
    sup_below = sorted([s for s in set(supports_all) if s < live_price], reverse=True)
    
    res_1 = res_above[0] if len(res_above) > 0 else None
    sup_minor = sup_below[0] if len(sup_below) > 0 else None
    sup_maj1 = sup_below[1] if len(sup_below) > 1 else None
    sup_maj2 = sup_below[2] if len(sup_below) > 2 else None
    
    return res_1, sup_minor, sup_maj1, sup_maj2

# --- DATA PROCESSING ---
df_1h, df_4h, df_1d = fetch_gold_data()

if not df_1h.empty and not df_1d.empty:
    # Anchor to the latest available close
    live_price = float(df_1h['Close'].iloc[-1])
    
    # 1. Daily Levels (Macro Trend Anchor, window=7)
    supports_1d, resistances_1d = find_structural_levels(df_1d, window=7)
    r1_1d, s_min_1d, s_maj1_1d, s_maj2_1d = extract_key_levels(supports_1d, resistances_1d, live_price)

    # 2. 4-Hour Levels (Swing Order Blocks, window=5)
    supports_4h, resistances_4h = find_structural_levels(df_4h, window=5)
    r1_4h, s_min_4h, s_maj1_4h, s_maj2_4h = extract_key_levels(supports_4h, resistances_4h, live_price)
    
    # 3. 1-Hour Levels (Intraday Scalp / Entry Triggers, window=8)
    supports_1h, resistances_1h = find_structural_levels(df_1h, window=8)
    r1_1h, s_min_1h, s_maj1_1h, s_maj2_1h = extract_key_levels(supports_1h, resistances_1h, live_price)

    # --- WRITTEN FORMAT DISPLAY ---
    st.subheader(f"Live Market Price: ~${live_price:,.2f}")
    
    col_1d, col_4h, col_1h = st.columns(3)
    
    with col_1d:
        st.markdown("### Daily (D1) Macro")
        st.markdown("**Resistance (Supply Zones)**")
        st.write(f"1st Resistance Point: ${r1_1d:,.2f}" if r1_1d else "1st Resistance Point: N/A")
        
        st.markdown("**Support (Demand Zones)**")
        st.write(f"Minor Support: ${s_min_1d:,.2f}" if s_min_1d else "Minor Support: N/A")
        st.write(f"Major Support 1: ${s_maj1_1d:,.2f}" if s_maj1_1d else "Major Support 1: N/A")
        st.write(f"Major Support 2: ${s_maj2_1d:,.2f}" if s_maj2_1d else "Major Support 2: N/A")

    with col_4h:
        st.markdown("### 4-Hour (4H) Swing")
        st.markdown("**Resistance (Supply Zones)**")
        st.write(f"1st Resistance Point: ${r1_4h:,.2f}" if r1_4h else "1st Resistance Point: N/A")
        
        st.markdown("**Support (Demand Zones)**")
        st.write(f"Minor Support: ${s_min_4h:,.2f}" if s_min_4h else "Minor Support: N/A")
        st.write(f"Major Support 1: ${s_maj1_4h:,.2f}" if s_maj1_4h else "Major Support 1: N/A")
        st.write(f"Major Support 2: ${s_maj2_4h:,.2f}" if s_maj2_4h else "Major Support 2: N/A")

    with col_1h:
        st.markdown("### 1-Hour (1H) Intraday")
        st.markdown("**Resistance (Supply Zones)**")
        st.write(f"1st Resistance Point: ${r1_1h:,.2f}" if r1_1h else "1st Resistance Point: N/A")
        
        st.markdown("**Support (Demand Zones)**")
        st.write(f"Minor Support: ${s_min_1h:,.2f}" if s_min_1h else "Minor Support: N/A")
        st.write(f"Major Support 1: ${s_maj1_1h:,.2f}" if s_maj1_1h else "Major Support 1: N/A")
        st.write(f"Major Support 2: ${s_maj2_1h:,.2f}" if s_maj2_1h else "Major Support 2: N/A")

    st.divider()

    # --- CHART VISUALIZATION (TABS) ---
    tab_1d, tab_4h, tab_1h = st.tabs(["Daily Macro Chart", "4-Hour Swing Chart", "1-Hour Intraday Chart"])

    with tab_1d:
        st.subheader("Daily Institutional Structure (1 Year)")
        fig_1d, ax_1d = plt.subplots(figsize=(14, 6))
        ax_1d.plot(df_1d.index, df_1d['Close'], label='D1 Close Price', color='black', linewidth=1.5)
        
        if r1_1d:
            ax_1d.axhline(r1_1d, color='red', linestyle='--', alpha=0.85, linewidth=2, label=f'D1 Supply (${r1_1d:,.2f})')
        if s_min_1d:
            ax_1d.axhline(s_min_1d, color='lightgreen', linestyle='--', alpha=0.9, linewidth=1.5, label=f'D1 Minor Demand (${s_min_1d:,.2f})')
        if s_maj1_1d:
            ax_1d.axhline(s_maj1_1d, color='green', linestyle='-', alpha=0.75, linewidth=2, label=f'D1 Major Demand 1 (${s_maj1_1d:,.2f})')
        if s_maj2_1d:
            ax_1d.axhline(s_maj2_1d, color='darkgreen', linestyle='-', alpha=0.9, linewidth=2.5, label=f'D1 Major Demand 2 (${s_maj2_1d:,.2f})')
        
        ax_1d.set_title('XAUUSD Daily Macro Structure')
        ax_1d.set_ylabel('Price (USD)')
        ax_1d.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax_1d.grid(alpha=0.2)
        st.pyplot(fig_1d)

    with tab_4h:
        st.subheader("4-Hour Structural Chart (30 Days)")
        fig_4h, ax_4h = plt.subplots(figsize=(14, 6))
        ax_4h.plot(df_4h.index, df_4h['Close'], label='H4 Close Price', color='black', linewidth=1.5)
        
        if r1_4h:
            ax_4h.axhline(r1_4h, color='red', linestyle='--', alpha=0.85, linewidth=2, label=f'H4 Supply (${r1_4h:,.2f})')
        if s_min_4h:
            ax_4h.axhline(s_min_4h, color='lightgreen', linestyle='--', alpha=0.9, linewidth=1.5, label=f'H4 Minor Demand (${s_min_4h:,.2f})')
        if s_maj1_4h:
            ax_4h.axhline(s_maj1_4h, color='green', linestyle='-', alpha=0.75, linewidth=2, label=f'H4 Major Demand 1 (${s_maj1_4h:,.2f})')
        if s_maj2_4h:
            ax_4h.axhline(s_maj2_4h, color='darkgreen', linestyle='-', alpha=0.9, linewidth=2.5, label=f'H4 Major Demand 2 (${s_maj2_4h:,.2f})')
        
        ax_4h.set_title('XAUUSD 4-Hour Macro Structure')
        ax_4h.set_ylabel('Price (USD)')
        ax_4h.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax_4h.grid(alpha=0.2)
        st.pyplot(fig_4h)

    with tab_1h:
        st.subheader("1-Hour Intraday Execution Chart (Last 10 Days)")
        df_1h_recent = df_1h.tail(240)
        fig_1h, ax_1h = plt.subplots(figsize=(14, 6))
        ax_1h.plot(df_1h_recent.index, df_1h_recent['Close'], label='H1 Close Price', color='darkblue', linewidth=1.2)
        
        if r1_1h:
            ax_1h.axhline(r1_1h, color='red', linestyle='--', alpha=0.85, linewidth=2, label=f'H1 Supply (${r1_1h:,.2f})')
        if s_min_1h:
            ax_1h.axhline(s_min_1h, color='lightgreen', linestyle='--', alpha=0.9, linewidth=1.5, label=f'H1 Minor Demand (${s_min_1h:,.2f})')
        if s_maj1_1h:
            ax_1h.axhline(s_maj1_1h, color='green', linestyle='-', alpha=0.75, linewidth=2, label=f'H1 Major Demand 1 (${s_maj1_1h:,.2f})')
        if s_maj2_1h:
            ax_1h.axhline(s_maj2_1h, color='darkgreen', linestyle='-', alpha=0.9, linewidth=2.5, label=f'H1 Major Demand 2 (${s_maj2_1h:,.2f})')
        
        ax_1h.set_title('XAUUSD 1-Hour Intraday Structure')
        ax_1h.set_ylabel('Price (USD)')
        ax_1h.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax_1h.grid(alpha=0.2)
        st.pyplot(fig_1h)
