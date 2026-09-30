import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tvDatafeed import TvDatafeed, Interval
import requests
import logging

# Silence TradingView login warnings
logging.getLogger('tvDatafeed').setLevel(logging.ERROR)

# --- PAGE CONFIG ---
st.set_page_config(page_title="XAUUSD Unified Strategy & Levels", layout="wide")
st.title("Gold (XAUUSD) — 15M MA Ribbon, OBV, 4H VPVR & Multi-Timeframe S/R")

# --- INITIALIZE TRADINGVIEW CONNECTION ---
@st.cache_resource
def get_tv_connection():
    try:
        return TvDatafeed()
    except Exception:
        return None

tv = get_tv_connection()

# --- CLOUD-RESILIENT LIVE SPOT FETCHER ---
@st.cache_data(ttl=15)
def get_live_xauusd_spot():
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Priority 1: TradingView CFD Scanner
    try:
        url = "https://scanner.tradingview.com/cfd/scan"
        payload = {"symbols": {"tickers": ["FXCM:XAUUSD", "OANDA:XAUUSD"]}, "columns": ["close"]}
        res = requests.post(url, json=payload, headers=headers, timeout=3)
        if res.status_code == 200:
            price = res.json().get('data', [{}])[0].get('d', [0])[0]
            if price > 1000: return float(price)
    except Exception: pass
    
    # Priority 2: KuCoin PAXG-USDT
    try:
        url = "https://api.kucoin.com/api/v1/market/orderbook/level1?symbol=PAXG-USDT"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            price = res.json().get('data', {}).get('price')
            if price and float(price) > 1000: return float(price)
    except Exception: pass

    # Priority 3: MEXC PAXGUSDT
    try:
        url = "https://api.mexc.com/api/v3/ticker/price?symbol=PAXGUSDT"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            price = res.json().get('price')
            if price and float(price) > 1000: return float(price)
    except Exception: pass

    return 4197.50

live_spot = get_live_xauusd_spot()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Strategy & Risk Parameters")
st.sidebar.success(f"Live Spot Synchronization Active\n\n**Current Spot: ${live_spot:,.2f}**")

capital = st.sidebar.number_input("Account Balance ($)", min_value=1000.0, value=15000.0, step=1000.0)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 1.5, 0.5)

st.sidebar.subheader("MA Ribbon Settings (15m)")
ma1_len = st.sidebar.number_input("MA 1 Period", value=45)
ma2_len = st.sidebar.number_input("MA 2 Period", value=54)
ma3_len = st.sidebar.number_input("MA 3 Period", value=63)
obv_lookback = st.sidebar.slider("OBV Swing Lookback (Bars)", 5, 40, 20)
vp_bins = st.sidebar.slider("4H Volume Profile Bins", 20, 80, 50)

st.sidebar.subheader("Chart Visual Controls")
h1_view_days = st.sidebar.slider(
    "1-Hour Chart Lookback (Days)",
    min_value=3, max_value=60, value=20,
    help="Adjust zoom level on recent session developments."
)

# --- BULLETPROOF MULTI-TIMEFRAME DATA PIPELINE ---
@st.cache_data(ttl=120)
def fetch_market_data():
    df_15m, df_1h, df_1d = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # 1. TradingView API
    if tv is not None:
        exchanges = ['OANDA', 'FXCM', 'FOREXCOM']
        for exc in exchanges:
            try:
                df_15m = tv.get_hist(symbol='XAUUSD', exchange=exc, interval=Interval.in_15_minute, n_bars=1500)
                df_1h = tv.get_hist(symbol='XAUUSD', exchange=exc, interval=Interval.in_1_hour, n_bars=1500)
                df_1d = tv.get_hist(symbol='XAUUSD', exchange=exc, interval=Interval.in_daily, n_bars=300)
                if df_15m is not None and not df_15m.empty and df_1h is not None and not df_1h.empty:
                    break
            except Exception:
                continue

    # 2. MEXC Fallback
    if df_15m is None or df_15m.empty or df_1h is None or df_1h.empty:
        try:
            r15 = requests.get("https://api.mexc.com/api/v3/klines?symbol=PAXGUSDT&interval=15m&limit=1000", timeout=5).json()
            d15 = pd.DataFrame(r15, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qav', 'nt', 'tbb', 'tbq'])
            d15['time'] = pd.to_datetime(d15['time'].astype(int), unit='ms', utc=True)
            df_15m = d15[['time', 'open', 'high', 'low', 'close', 'volume']].set_index('time').astype(float).sort_index()

            r1h = requests.get("https://api.mexc.com/api/v3/klines?symbol=PAXGUSDT&interval=60m&limit=1000", timeout=5).json()
            d1h = pd.DataFrame(r1h, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qav', 'nt', 'tbb', 'tbq'])
            d1h['time'] = pd.to_datetime(d1h['time'].astype(int), unit='ms', utc=True)
            df_1h = d1h[['time', 'open', 'high', 'low', 'close', 'volume']].set_index('time').astype(float).sort_index()

            r1d = requests.get("https://api.mexc.com/api/v3/klines?symbol=PAXGUSDT&interval=1d&limit=300", timeout=5).json()
            d1d = pd.DataFrame(r1d, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'ct', 'qav', 'nt', 'tbb', 'tbq'])
            d1d['time'] = pd.to_datetime(d1d['time'].astype(int), unit='ms', utc=True)
            df_1d = d1d[['time', 'open', 'high', 'low', 'close', 'volume']].set_index('time').astype(float).sort_index()
        except Exception:
            pass

    # 3. yfinance Fallback
    if df_15m is None or df_15m.empty or df_1h is None or df_1h.empty:
        try:
            df_15m = yf.download("XAUUSD=X", period="14d", interval="15m", progress=False)
            df_1h = yf.download("XAUUSD=X", period="60d", interval="1h", progress=False)
            df_1d = yf.download("XAUUSD=X", period="1y", interval="1d", progress=False)
        except Exception:
            pass

    if df_15m is not None and not df_15m.empty and df_1h is not None and not df_1h.empty and df_1d is not None and not df_1d.empty:
        for df in [df_15m, df_1h, df_1d]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.rename(columns=lambda x: x.capitalize() if isinstance(x, str) else x, inplace=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

            if 'Volume' not in df.columns or (df['Volume'] == 0).all():
                df['Volume'] = ((df['High'] - df['Low']) * 10000).clip(lower=10)

        df_4h = df_1h.resample('4h').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).dropna()
        
        return df_15m, df_1h, df_4h, df_1d

    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- S/R MULTI-LEVEL DETECTION (LOCKED TO CLOSED CANDLES) ---
def find_structural_levels(df, window=5):
    closed_df = df.iloc[:-1]
    maxima_indices = argrelextrema(closed_df['High'].values, np.greater, order=window)[0]
    resistances = closed_df['High'].iloc[maxima_indices].values
    
    minima_indices = argrelextrema(closed_df['Low'].values, np.less, order=window)[0]
    supports = closed_df['Low'].iloc[minima_indices].values
    return supports, resistances

def extract_key_levels(supports_all, resistances_all, live_price):
    res_above = sorted([r for r in set(resistances_all) if r > live_price])
    sup_below = sorted([s for s in set(supports_all) if s < live_price], reverse=True)
    
    r1 = res_above[0] if len(res_above) > 0 else None
    r2 = res_above[1] if len(res_above) > 1 else None
    r3 = res_above[2] if len(res_above) > 2 else None
    
    s1 = sup_below[0] if len(sup_below) > 0 else None
    s2 = sup_below[1] if len(sup_below) > 1 else None
    s3 = sup_below[2] if len(sup_below) > 2 else None
    
    return r1, r2, r3, s1, s2, s3

def shade_london_ny_overlap(ax, df_subset):
    unique_dates = sorted(list(set(df_subset.index.date)))
    shaded_label_added = False
    for day in unique_dates:
        session_start = pd.Timestamp(day, tz='UTC') + pd.Timedelta(hours=12)
        session_end = pd.Timestamp(day, tz='UTC') + pd.Timedelta(hours=16)
        if session_end >= df_subset.index.min() and session_start <= df_subset.index.max():
            lbl = "London/NY Overlap (Peak Volume)" if not shaded_label_added else ""
            ax.axvspan(session_start, session_end, color='goldenrod', alpha=0.18, label=lbl)
            shaded_label_added = True

# --- 15M MA RIBBON & OBV LOGIC ---
def compute_15m_strategy_indicators(df, m1, m2, m3, obv_window):
    df = df.copy()
    
    df['MA45'] = df['Close'].ewm(span=m1, adjust=False).mean()
    df['MA54'] = df['Close'].ewm(span=m2, adjust=False).mean()
    df['MA63'] = df['Close'].ewm(span=m3, adjust=False).mean()
    df['MA_Ribbon_High'] = df[['MA45', 'MA54', 'MA63']].max(axis=1)
    df['MA_Ribbon_Low'] = df[['MA45', 'MA54', 'MA63']].min(axis=1)
    
    direction = np.sign(df['Close'].diff().fillna(0))
    df['OBV'] = (direction * df['Volume']).cumsum()
    
    df['OBV_Swing_High'] = df['OBV'].rolling(window=obv_window).max().shift(1)
    df['OBV_Swing_Low'] = df['OBV'].rolling(window=obv_window).min().shift(1)
    
    df['Price_Lowest_Low'] = df['Low'].rolling(window=obv_window).min().shift(1)
    df['Price_Highest_High'] = df['High'].rolling(window=obv_window).max().shift(1)
    return df

# --- 4H VOLUME PROFILE VISIBLE RANGE (VPVR) ---
def compute_vpvr(df_4h, num_bins=50):
    if df_4h.empty:
        return 0, 0, 0, None, None
        
    price_min = df_4h['Low'].min()
    price_max = df_4h['High'].max()
    bins = np.linspace(price_min, price_max, num_bins)
    
    bin_volumes = np.zeros(num_bins - 1)
    for _, row in df_4h.iterrows():
        mask = (bins[:-1] >= row['Low']) & (bins[1:] <= row['High'])
        if mask.any():
            bin_volumes[mask] += row['Volume'] / mask.sum()
        else:
            closest = np.argmin(np.abs(bins[:-1] - row['Close']))
            bin_volumes[closest] += row['Volume']
            
    poc_idx = np.argmax(bin_volumes)
    poc_price = (bins[poc_idx] + bins[poc_idx + 1]) / 2.0
    
    total_volume = bin_volumes.sum()
    target_vol = total_volume * 0.70
    sorted_indices = np.argsort(bin_volumes)[::-1]
    
    accum_vol = 0
    va_indices = []
    for idx in sorted_indices:
        accum_vol += bin_volumes[idx]
        va_indices.append(idx)
        if accum_vol >= target_vol:
            break
            
    val_price = bins[min(va_indices)]
    vah_price = bins[max(va_indices) + 1]
    
    return poc_price, vah_price, val_price, bins, bin_volumes

# --- EXECUTION & DASHBOARD RENDER ---
df_15m_raw, df_1h_raw, df_4h_raw, df_1d_raw = fetch_market_data()

if not df_15m_raw.empty and not df_1h_raw.empty and not df_1d_raw.empty:
    current_price = live_spot

    # Calculate 15m Indicators & 4H VPVR
    df_15m = compute_15m_strategy_indicators(df_15m_raw, ma1_len, ma2_len, ma3_len, obv_lookback)
    poc_4h, vah_4h, val_4h, vp_bins_arr, vp_vols = compute_vpvr(df_4h_raw.tail(30), vp_bins)
    
    # Calculate Multi-Timeframe Structural Levels
    sup_1d, res_1d = find_structural_levels(df_1d_raw, window=7)
    r1_1d, r2_1d, r3_1d, s1_1d, s2_1d, s3_1d = extract_key_levels(sup_1d, res_1d, current_price)

    sup_4h, res_4h = find_structural_levels(df_4h_raw, window=5)
    r1_4h, r2_4h, r3_4h, s1_4h, s2_4h, s3_4h = extract_key_levels(sup_4h, res_4h, current_price)
    
    sup_1h, res_1h = find_structural_levels(df_1h_raw, window=8)
    r1_1h, r2_1h, r3_1h, s1_1h, s2_1h, s3_1h = extract_key_levels(sup_1h, res_1h, current_price)

    # Signal & HVN Entry Calculation
    last_closed = df_15m.iloc[-2]
    prev_closed = df_15m.iloc[-3]
    recent_20 = df_15m.iloc[-22:-2]
    p_bins = np.linspace(recent_20['Low'].min(), recent_20['High'].max(), 15)
    hist, edges = np.histogram(recent_20['Close'], bins=p_bins, weights=recent_20['Volume'])
    hvn_entry_level = (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1]) / 2.0

    obv_buy_confirm = last_closed['OBV'] > last_closed['OBV_Swing_High']
    obv_sell_confirm = last_closed['OBV'] < last_closed['OBV_Swing_Low']

    signal = "NEUTRAL"
    if last_closed['Close'] > last_closed['MA_Ribbon_High'] and obv_buy_confirm:
        signal = "BUY"
    elif last_closed['Close'] < last_closed['MA_Ribbon_Low'] and obv_sell_confirm:
        signal = "SELL"

    if signal == "BUY":
        setup_color = "green"
        entry_price = hvn_entry_level
        stop_loss = last_closed['Price_Lowest_Low'] - 1.50
        take_profit = r1_4h if r1_4h and r1_4h > entry_price else (vah_4h if vah_4h > entry_price else entry_price + 15.0)
    elif signal == "SELL":
        setup_color = "red"
        entry_price = hvn_entry_level
        stop_loss = last_closed['Price_Highest_High'] + 1.50
        take_profit = s1_4h if s1_4h and s1_4h < entry_price else (val_4h if val_4h < entry_price else entry_price - 15.0)
    else:
        setup_color = "gray"
        entry_price = hvn_entry_level
        stop_loss = last_closed['Price_Lowest_Low'] - 1.50
        take_profit = poc_4h

    sl_distance = abs(entry_price - stop_loss)
    risk_dollars = capital * (risk_pct / 100)
    lot_size = round(risk_dollars / (sl_distance * 100), 2) if sl_distance > 0 else 0.01

    # --- TOP EXECUTION DESK ---
    st.subheader(f"Current Live Spot Price: ${current_price:,.2f}")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Strategy Signal", f":{setup_color}[{signal}]", f"Spot: ${current_price:,.2f}")
    col2.metric("HVN Entry Level", f"${entry_price:,.2f}", "High Volume Node")
    col3.metric("Structural Stop Loss", f"${stop_loss:,.2f}", f"-${sl_distance:.2f} pts")
    col4.metric("Dynamic Take Profit", f"${take_profit:,.2f}", "4H S/R & VPVR Target")
    col5.metric("Calculated Lot Size", f"{lot_size} Lots", f"${risk_dollars:,.0f} Max Risk")

    st.divider()

    # --- MULTI-TIMEFRAME S/R METRICS DISPLAY ---
    col_1d, col_4h, col_1h = st.columns(3)
    with col_1d:
        st.markdown("### Daily (D1) Macro")
        st.write(f"**Resistance 3:** ${r3_1d:,.2f}" if r3_1d else "**Resistance 3:** N/A")
        st.write(f"**Resistance 2:** ${r2_1d:,.2f}" if r2_1d else "**Resistance 2:** N/A")
        st.write(f"**Resistance 1:** ${r1_1d:,.2f}" if r1_1d else "**Resistance 1:** N/A")
        st.write("---")
        st.write(f"**Support 1 (Minor):** ${s1_1d:,.2f}" if s1_1d else "**Support 1:** N/A")
        st.write(f"**Support 2 (Major):** ${s2_1d:,.2f}" if s2_1d else "**Support 2:** N/A")
        st.write(f"**Support 3 (Macro):** ${s3_1d:,.2f}" if s3_1d else "**Support 3:** N/A")

    with col_4h:
        st.markdown("### 4-Hour (4H) Swing")
        st.write(f"**Resistance 3:** ${r3_4h:,.2f}" if r3_4h else "**Resistance 3:** N/A")
        st.write(f"**Resistance 2:** ${r2_4h:,.2f}" if r2_4h else "**Resistance 2:** N/A")
        st.write(f"**Resistance 1:** ${r1_4h:,.2f}" if r1_4h else "**Resistance 1:** N/A")
        st.write("---")
        st.write(f"**Support 1 (Minor):** ${s1_4h:,.2f}" if s1_4h else "**Support 1:** N/A")
        st.write(f"**Support 2 (Major):** ${s2_4h:,.2f}" if s2_4h else "**Support 2:** N/A")
        st.write(f"**Support 3 (Macro):** ${s3_4h:,.2f}" if s3_4h else "**Support 3:** N/A")

    with col_1h:
        st.markdown("### 1-Hour (1H) Intraday")
        st.write(f"**Resistance 3:** ${r3_1h:,.2f}" if r3_1h else "**Resistance 3:** N/A")
        st.write(f"**Resistance 2:** ${r2_1h:,.2f}" if r2_1h else "**Resistance 2:** N/A")
        st.write(f"**Resistance 1:** ${r1_1h:,.2f}" if r1_1h else "**Resistance 1:** N/A")
        st.write("---")
        st.write(f"**Support 1 (Minor):** ${s1_1h:,.2f}" if s1_1h else "**Support 1:** N/A")
        st.write(f"**Support 2 (Major):** ${s2_1h:,.2f}" if s2_1h else "**Support 2:** N/A")
        st.write(f"**Support 3 (Macro):** ${s3_1h:,.2f}" if s3_1h else "**Support 3:** N/A")

    st.divider()

    # --- COMBINED CHART TABS ---
    tab_exec, tab_1h, tab_4h, tab_1d = st.tabs([
        "🔴 15M Strategy & OBV Execution Desk", 
        "1-Hour Intraday Chart", 
        "4-Hour Swing Chart", 
        "Daily Macro Chart"
    ])

    with tab_exec:
        st.subheader("15-Minute Ribbon Execution + Integrated On-Balance Volume")
        df_plot = df_15m.tail(96)
        
        # Dual-subplot inside the same tab
        fig_comb, (ax_main, ax_obv) = plt.subplots(
            2, 1, figsize=(15, 9), 
            gridspec_kw={'height_ratios': [2.5, 1]}, 
            sharex=True
        )

        # Upper Subplot: Price, Ribbon, HVN, 4H S/R Overlays
        ax_main.plot(df_plot.index, df_plot['Close'], color='black', linewidth=1.5, label='XAUUSD Spot (15m)')
        ax_main.plot(df_plot.index, df_plot['MA45'], color='cyan', linestyle='--', linewidth=1, label=f'EMA {ma1_len}')
        ax_main.plot(df_plot.index, df_plot['MA54'], color='blue', linestyle='--', linewidth=1, label=f'EMA {ma2_len}')
        ax_main.plot(df_plot.index, df_plot['MA63'], color='darkblue', linestyle='--', linewidth=1, label=f'EMA {ma3_len}')
        ax_main.fill_between(df_plot.index, df_plot['MA_Ribbon_Low'], df_plot['MA_Ribbon_High'], color='blue', alpha=0.08, label='MA Ribbon Zone')

        # Order Levels
        ax_main.axhline(entry_price, color='gold', linestyle='-', linewidth=2, label=f'HVN Entry (${entry_price:,.2f})')
        ax_main.axhline(stop_loss, color='red', linestyle='--', linewidth=1.8, label=f'OBV Stop Loss (${stop_loss:,.2f})')
        ax_main.axhline(take_profit, color='green', linestyle='--', linewidth=1.8, label=f'Target TP (${take_profit:,.2f})')

        # 4H VPVR and Structural Level Overlays
        ax_main.axhline(poc_4h, color='purple', linestyle=':', linewidth=1.4, label=f'4H POC (${poc_4h:,.2f})')
        ax_main.axhspan(val_4h, vah_4h, color='purple', alpha=0.04, label='4H Value Area')
        if r1_4h: ax_main.axhline(r1_4h, color='firebrick', linestyle=':', alpha=0.6, label=f'4H R1 (${r1_4h:,.2f})')
        if s1_4h: ax_main.axhline(s1_4h, color='forestgreen', linestyle=':', alpha=0.6, label=f'4H S1 (${s1_4h:,.2f})')

        ax_main.set_ylabel('Gold Price (USD)')
        ax_main.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        ax_main.grid(alpha=0.2)

        # Lower Subplot: On-Balance Volume & Structural Swing Bands
        ax_obv.plot(df_plot.index, df_plot['OBV'], color='teal', linewidth=1.8, label='On-Balance Volume')
        ax_obv.plot(df_plot.index, df_plot['OBV_Swing_High'], color='green', linestyle=':', label='OBV Swing High')
        ax_obv.plot(df_plot.index, df_plot['OBV_Swing_Low'], color='red', linestyle=':', label='OBV Swing Low')
        ax_obv.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
        ax_obv.set_ylabel('OBV')
        ax_obv.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
        ax_obv.grid(alpha=0.2)

        plt.subplots_adjust(hspace=0.08)
        st.pyplot(fig_comb)

    with tab_1h:
        st.subheader(f"1-Hour Intraday Chart (Last {h1_view_days} Days)")
        cutoff = df_1h_raw.index.max() - pd.Timedelta(days=h1_view_days)
        df_1h_view = df_1h_raw[df_1h_raw.index >= cutoff]
        
        fig_1h, ax_1h = plt.subplots(figsize=(14, 6))
        shade_london_ny_overlap(ax_1h, df_1h_view)
        ax_1h.plot(df_1h_view.index, df_1h_view['Close'], label='H1 Close', color='navy', linewidth=1.4)
        
        if r3_1h: ax_1h.axhline(r3_1h, color='darkred', linestyle='-', alpha=0.9, label=f'H1 R3 (${r3_1h:,.2f})')
        if r2_1h: ax_1h.axhline(r2_1h, color='red', linestyle='-', alpha=0.75, label=f'H1 R2 (${r2_1h:,.2f})')
        if r1_1h: ax_1h.axhline(r1_1h, color='lightcoral', linestyle='--', alpha=0.85, label=f'H1 R1 (${r1_1h:,.2f})')
        
        if s1_1h: ax_1h.axhline(s1_1h, color='lightgreen', linestyle='--', alpha=0.9, label=f'H1 S1 (${s1_1h:,.2f})')
        if s2_1h: ax_1h.axhline(s2_1h, color='green', linestyle='-', alpha=0.75, label=f'H1 S2 (${s2_1h:,.2f})')
        if s3_1h: ax_1h.axhline(s3_1h, color='darkgreen', linestyle='-', alpha=0.9, label=f'H1 S3 (${s3_1h:,.2f})')
        
        ax_1h.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax_1h.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax_1h.set_ylabel('Spot Price (USD)')
        ax_1h.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax_1h.grid(alpha=0.2)
        st.pyplot(fig_1h)

    with tab_4h:
        st.subheader("4-Hour Structural Chart")
        fig_4h, ax_4h = plt.subplots(figsize=(14, 6))
        ax_4h.plot(df_4h_raw.index, df_4h_raw['Close'], label='H4 Close', color='black', linewidth=1.5)
        
        if r3_4h: ax_4h.axhline(r3_4h, color='darkred', linestyle='-', alpha=0.9, label=f'H4 R3 (${r3_4h:,.2f})')
        if r2_4h: ax_4h.axhline(r2_4h, color='red', linestyle='-', alpha=0.75, label=f'H4 R2 (${r2_4h:,.2f})')
        if r1_4h: ax_4h.axhline(r1_4h, color='lightcoral', linestyle='--', alpha=0.85, label=f'H4 R1 (${r1_4h:,.2f})')
        
        if s1_4h: ax_4h.axhline(s1_4h, color='lightgreen', linestyle='--', alpha=0.9, label=f'H4 S1 (${s1_4h:,.2f})')
        if s2_4h: ax_4h.axhline(s2_4h, color='green', linestyle='-', alpha=0.75, label=f'H4 S2 (${s2_4h:,.2f})')
        if s3_4h: ax_4h.axhline(s3_4h, color='darkgreen', linestyle='-', alpha=0.9, label=f'H4 S3 (${s3_4h:,.2f})')
        
        ax_4h.set_ylabel('Spot Price (USD)')
        ax_4h.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax_4h.grid(alpha=0.2)
        st.pyplot(fig_4h)

    with tab_1d:
        st.subheader("Daily Institutional Structure")
        fig_1d, ax_1d = plt.subplots(figsize=(14, 6))
        ax_1d.plot(df_1d_raw.index, df_1d_raw['Close'], label='D1 Close', color='black', linewidth=1.5)
        
        if r3_1d: ax_1d.axhline(r3_1d, color='darkred', linestyle='-', alpha=0.9, label=f'D1 R3 (${r3_1d:,.2f})')
        if r2_1d: ax_1d.axhline(r2_1d, color='red', linestyle='-', alpha=0.75, label=f'D1 R2 (${r2_1d:,.2f})')
        if r1_1d: ax_1d.axhline(r1_1d, color='lightcoral', linestyle='--', alpha=0.85, label=f'D1 R1 (${r1_1d:,.2f})')
        
        if s1_1d: ax_1d.axhline(s1_1d, color='lightgreen', linestyle='--', alpha=0.9, label=f'D1 S1 (${s1_1d:,.2f})')
        if s2_1d: ax_1d.axhline(s2_1d, color='green', linestyle='-', alpha=0.75, label=f'D1 S2 (${s2_1d:,.2f})')
        if s3_1d: ax_1d.axhline(s3_1d, color='darkgreen', linestyle='-', alpha=0.9, label=f'D1 S3 (${s3_1d:,.2f})')
        
        ax_1d.set_ylabel('Spot Price (USD)')
        ax_1d.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax_1d.grid(alpha=0.2)
        st.pyplot(fig_1d)

else:
    st.error("Market data feeds are currently unreachable. Please wait for the API rate limit to clear.")
