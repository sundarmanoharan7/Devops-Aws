import streamlit as st
import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import datetime

# ==========================================
# CONFIGURATION & STATE
# ==========================================
st.set_page_config(page_title="Gold/Silver Stat Arb", layout="wide")
st.title("XAUXAG Statistical Arbitrage Dashboard")

# Strategy Parameters
SYMBOL_GOLD = "XAUUSD"
SYMBOL_SILVER = "XAGUSD"
TIMEFRAME = mt5.TIMEFRAME_H1
BAR_COUNT = 300
BASE_GOLD_LOTS = 0.10

Z_ENTRY = 2.0
Z_EXIT = 0.2
Z_STOP = 3.2
RSI_PERIOD = 14
SMA_PERIOD = 20

# Initialize Session State to remember active positions across refreshes
if 'active_position' not in st.session_state:
    st.session_state.active_position = None

# ==========================================
# HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def initialize_mt5():
    if not mt5.initialize():
        st.error(f"MT5 initialization failed: {mt5.last_error()}")
        st.stop()
    for sym in [SYMBOL_GOLD, SYMBOL_SILVER]:
        if not mt5.symbol_select(sym, True):
            st.error(f"Symbol {sym} not found in Market Watch.")
            st.stop()
    return True

def fetch_historical_ratio():
    rates_g = mt5.copy_rates_from_pos(SYMBOL_GOLD, TIMEFRAME, 0, BAR_COUNT)
    rates_s = mt5.copy_rates_from_pos(SYMBOL_SILVER, TIMEFRAME, 0, BAR_COUNT)
    
    if rates_g is None or rates_s is None:
        return None
        
    df_g = pd.DataFrame(rates_g)[['time', 'close']].rename(columns={'close': 'gold'})
    df_s = pd.DataFrame(rates_s)[['time', 'close']].rename(columns={'close': 'silver'})
    
    df = pd.merge(df_g, df_s, on='time')
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['ratio'] = df['gold'] / df['silver']
    return df

def calculate_metrics(df):
    df['sma20'] = df['ratio'].rolling(window=SMA_PERIOD).mean()
    df['std20'] = df['ratio'].rolling(window=SMA_PERIOD).std()
    df['z_score'] = (df['ratio'] - df['sma20']) / df['std20']
    df['rsi'] = ta.rsi(df['ratio'], length=RSI_PERIOD)
    return df

def get_silver_lots(gold_price, silver_price):
    gold_exposure = (BASE_GOLD_LOTS * 100) * gold_price
    req_silver_oz = gold_exposure / silver_price
    return round(req_silver_oz / 5000.0, 2)

# ==========================================
# UI & EXECUTION LOGIC
# ==========================================
initialize_mt5()

tick_g = mt5.symbol_info_tick(SYMBOL_GOLD)
tick_s = mt5.symbol_info_tick(SYMBOL_SILVER)

if tick_g is None or tick_s is None:
    st.warning("Fetching live ticks... retrying.")
    time.sleep(2)
    st.rerun()

live_gold = tick_g.bid
live_silver = tick_s.bid
live_ratio = live_gold / live_silver
silver_lots = get_silver_lots(live_gold, live_silver)

df = fetch_historical_ratio()
if df is not None:
    df = calculate_metrics(df)
    last_closed = df.iloc[-2]  # Use last closed candle to prevent repainting
    
    current_z = last_closed['z_score']
    current_rsi = last_closed['rsi']
    current_sma = last_closed['sma20']

    # --- TOP DASHBOARD ---
    st.markdown("### Live Market Data")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gold (XAUUSD)", f"${live_gold:.2f}")
    col2.metric("Silver (XAGUSD)", f"${live_silver:.3f}")
    col3.metric("Live Ratio", f"{live_ratio:.3f}")
    col4.metric("Silver Target Lot Size", f"{silver_lots} Lots", help="Based on 0.10 Gold Lots")

    st.markdown("### Quantitative Setup Metrics (H1)")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Z-Score", f"{current_z:.2f}", delta=f"{current_z - df.iloc[-3]['z_score']:.2f}")
    m_col2.metric("RSI (14)", f"{current_rsi:.1f}")
    m_col3.metric("20 SMA Baseline", f"{current_sma:.3f}")
    m_col4.metric("Active State", str(st.session_state.active_position))

    st.divider()

    # --- SIGNAL EVALUATION ---
    st.markdown("### Live Trade Alerts")
    
    pos = st.session_state.active_position

    if pos is None:
        # Check Entry Conditions
        if current_z >= Z_ENTRY and current_rsi >= 68.0:
            st.error(f"🚨 **SIGNAL: SHORT RATIO** (Gold Overvalued)\n\n"
                     f"**Execute:** SELL {BASE_GOLD_LOTS} Gold & BUY {silver_lots} Silver.")
            if st.button("Acknowledge & Set Status to SHORT_RATIO"):
                st.session_state.active_position = "SHORT_RATIO"
                
        elif current_z <= -Z_ENTRY and current_rsi <= 32.0:
            st.success(f"🚨 **SIGNAL: LONG RATIO** (Silver Overvalued)\n\n"
                       f"**Execute:** BUY {BASE_GOLD_LOTS} Gold & SELL {silver_lots} Silver.")
            if st.button("Acknowledge & Set Status to LONG_RATIO"):
                st.session_state.active_position = "LONG_RATIO"
        else:
            st.info("⚪ **FLAT**: No statistical anomaly detected. Waiting for Z-Score to breach ±2.0.")

    elif pos == "SHORT_RATIO":
        if current_z <= Z_EXIT:
            st.success("🎯 **PROFIT TARGET HIT (Mean Reversion)**: Close all positions.")
            if st.button("Reset State to FLAT"):
                st.session_state.active_position = None
        elif current_z >= Z_STOP:
            st.error("⚠️ **STOP LOSS TRIGGERED**: Structural breakdown. Close all positions.")
            if st.button("Reset State to FLAT"):
                st.session_state.active_position = None
        else:
            st.warning("📉 **POSITION ACTIVE: SHORT RATIO**. Waiting for Z-Score to return to 0.")
            if st.button("Force Reset (Manual Close)"):
                st.session_state.active_position = None

    elif pos == "LONG_RATIO":
        if current_z >= -Z_EXIT:
            st.success("🎯 **PROFIT TARGET HIT (Mean Reversion)**: Close all positions.")
            if st.button("Reset State to FLAT"):
                st.session_state.active_position = None
        elif current_z <= -Z_STOP:
            st.error("⚠️ **STOP LOSS TRIGGERED**: Structural breakdown. Close all positions.")
            if st.button("Reset State to FLAT"):
                st.session_state.active_position = None
        else:
            st.success("📈 **POSITION ACTIVE: LONG RATIO**. Waiting for Z-Score to return to 0.")
            if st.button("Force Reset (Manual Close)"):
                st.session_state.active_position = None

    # --- CHART VISUALIZATION ---
    st.markdown("### Historical Ratio Trend (Last 300 H1 Bars)")
    chart_data = df.set_index('time')[['ratio', 'sma20']]
    st.line_chart(chart_data)

# Auto-refresh app every 15 seconds
time.sleep(15)
st.rerun()
