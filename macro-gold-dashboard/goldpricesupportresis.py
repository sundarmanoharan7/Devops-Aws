import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- PAGE CONFIG ---
st.set_page_config(page_title="XAUUSD Multi-Timeframe Levels", layout="wide")
st.title("Gold (XAUUSD) Multi-Timeframe Structural Levels")

# --- SPOT BASIS CALIBRATION CONTROLS ---
st.sidebar.header("Price & Session Settings")
target_spot_price = st.sidebar.number_input(
    "Target Spot Close (XAUUSD)",
    min_value=1000.0,
    max_value=10000.0,
    value=4348.00,
    step=0.50,
    help="Anchors the COMEX futures structure to your broker's exact spot closing price."
)

h1_view_days = st.sidebar.slider(
    "1-Hour Chart Lookback (Days)",
    min_value=3,
    max_value=90,
    value=7,
    help="Zoom in on recent days to see session overlap shading clearly."
)

# --- DATA FETCHING ---
@st.cache_data(ttl=1800)
def fetch_gold_data(spot_anchor: float):
    # Fetch 90 days of 1-Hour data for 1H and 4H structures
    df_1h = yf.download('GC=F', period="90d", interval='1h', progress=False)
    
    # Fetch 1 Year of Daily data for macro D1 structure
    df_1d = yf.download('GC=F', period="1y", interval='1d', progress=False)
    
    if df_1h.empty or df_1d.empty:
        st.error("Failed to fetch data from Yahoo Finance. The API might be temporarily unavailable.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Flatten MultiIndex columns if present
    if isinstance(df_1h.columns, pd.MultiIndex):
        df_1h.columns = df_1h.columns.get_level_values(0)
    if isinstance(df_1d.columns, pd.MultiIndex):
        df_1d.columns = df_1d.columns.get_level_values(0)

    # Standardize datetime index to UTC
    if df_1h.index.tz is None:
        df_1h.index = df_1h.index.tz_localize('UTC')
    else:
        df_1h.index = df_1h.index.tz_convert('UTC')

    if df_1d.index.tz is None:
        df_1d.index = df_1d.index.tz_localize('UTC')
    else:
        df_1d.index = df_1d.index.tz_convert('UTC')

    # Calculate basis premium (Futures Premium over Spot)
    latest_futures_close = float(df_1h['Close'].dropna().iloc[-1])
    basis_offset = latest_futures_close - spot_anchor

    # Calibrate datasets to true Spot prices
    for col in ['Open', 'High', 'Low', 'Close']:
        df_1h[col] = df_1h[col] - basis_offset
        df_1d[col] = df_1d[col] - basis_offset

    # Resample calibrated 1-hour candles into 4-hour structural bars
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

# --- SESSION OVERLAP SHADING HELPER ---
def shade_london_ny_overlap(ax, df_subset):
    """Shades London & New York session overlap (12:00 to 16:00 UTC)"""
    unique_dates = sorted(list(set(df_subset.index.date)))
    shaded_label_added = False
    
    for day in unique_dates:
        # Define start and end of overlap window in UTC
        session_start = pd.Timestamp(day, tz='UTC') + pd.Timedelta(hours=12)
        session_end = pd.Timestamp(day, tz='UTC') + pd.Timedelta(hours=16)
        
        # Only shade if within subset boundaries
        if session_end >= df_subset.index.min() and session_start <= df_subset.index.max():
            lbl = "London/NY Overlap (Peak Volume)" if not shaded_label_added else ""
            ax.axvspan(session_start, session_end, color='goldenrod', alpha=0.18, label=lbl)
            shaded_label_added = True

# --- DATA PROCESSING ---
df_1h, df_4h, df_1d = fetch_gold_data(target_spot_price)

if not df_1h.empty and not df_1d.empty:
    live_price = float(df_1h['Close'].iloc[-1])
    
    # 1. Daily Levels (window=7)
    supports_1d, resistances_1d = find_structural_levels(df_1d, window=7)
    r1_1d, s_min_1d, s_maj1_1d, s_maj2_1d = extract_key_levels(supports_1d, resistances_1d, live_price)

    # 2. 4-Hour Levels (window=5)
    supports_4h, resistances_4h = find_structural_levels(df_4h, window=5)
    r1_4h, s_min_4h, s_maj1_4h, s_maj2_4h = extract_key_levels(supports_4h, resistances_4h, live_price)
    
    # 3. 1-Hour Levels (window=8)
    supports_1h, resistances_1h = find_structural_levels(df_1h, window=8)
    r1_1h, s_min_1h, s_maj1_1h, s_maj2_1h = extract_key_levels(supports_1h, resistances_1h, live_price)

    # --- WRITTEN FORMAT DISPLAY ---
    st.subheader(f"Live Calibrated Spot Price: ~${live_price:,.2f}")
    
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
        st.subheader("Daily Institutional Structure (1 Year - Spot Calibrated)")
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
        st.subheader("4-Hour Structural Chart (90 Days - Spot Calibrated)")
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
        st.subheader(f"1-Hour Intraday Chart (Last {h1_view_days} Days - Session Highlighted)")
        
        # Filter 1H dataset based on sidebar slider for clear inspection
        cutoff_date = df_1h.index.max() - pd.Timedelta(days=h1_view_days)
        df_1h_view = df_1h[df_1h.index >= cutoff_date]
        
        fig_1h, ax_1h = plt.subplots(figsize=(14, 6))
        
        # 1. Overlay Institutional Session Shading (London/NY Overlap)
        shade_london_ny_overlap(ax_1h, df_1h_view)
        
        # 2. Plot Price
        ax_1h.plot(df_1h_view.index, df_1h_view['Close'], label='H1 Close Price', color='navy', linewidth=1.4)
        
        # 3. Horizontal Support & Resistance Levels
        if r1_1h:
            ax_1h.axhline(r1_1h, color='red', linestyle='--', alpha=0.85, linewidth=2, label=f'H1 Supply (${r1_1h:,.2f})')
        if s_min_1h:
            ax_1h.axhline(s_min_1h, color='lightgreen', linestyle='--', alpha=0.9, linewidth=1.5, label=f'H1 Minor Demand (${s_min_1h:,.2f})')
        if s_maj1_1h:
            ax_1h.axhline(s_maj1_1h, color='green', linestyle='-', alpha=0.75, linewidth=2, label=f'H1 Major Demand 1 (${s_maj1_1h:,.2f})')
        if s_maj2_1h:
            ax_1h.axhline(s_maj2_1h, color='darkgreen', linestyle='-', alpha=0.9, linewidth=2.5, label=f'H1 Major Demand 2 (${s_maj2_1h:,.2f})')
        
        # Format X-axis dates clearly
        ax_1h.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M UTC'))
        
        ax_1h.set_title('XAUUSD 1-Hour Intraday Structure with Session Overlap')
        ax_1h.set_ylabel('Price (USD)')
        ax_1h.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax_1h.grid(alpha=0.2)
        st.pyplot(fig_1h)
        st.caption("Golden bands indicate the London & New York Session Overlap (12:00 – 16:00 UTC), representing the highest probability execution windows.")
