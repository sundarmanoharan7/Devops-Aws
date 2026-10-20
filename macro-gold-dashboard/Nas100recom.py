import time
import datetime
import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import streamlit as st

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="NAS100 OBV Dashboard", layout="wide")
st.title("NAS100 Live OBV & VWAP Monitor")

# Copy the exact helper functions from the previous script here:
# fetch_rates(), calculate_session_vwap(), calculate_indicators()
# (Omitted here for brevity, but keep them exactly the same)

def initialize_mt5():
    if not mt5.initialize():
        st.error(f"MT5 initialization failed. Ensure MT5 is running. Error: {mt5.last_error()}")
        st.stop()
    if not mt5.symbol_select("NAS100", True):
        st.error("Failed to select NAS100 in Market Watch.")
        st.stop()

# ==========================================
# STREAMLIT UI & EXECUTION
# ==========================================
initialize_mt5()

# Fetch Live Tick
tick = mt5.symbol_info_tick("NAS100")
if tick is None:
    st.warning("Failed to retrieve live tick. Retrying in 2 seconds...")
    time.sleep(2)
    st.rerun()

live_bid = tick.bid
live_time = datetime.datetime.fromtimestamp(tick.time).strftime('%H:%M:%S')

# Fetch Data
df_m15 = fetch_rates("NAS100", mt5.TIMEFRAME_M15, 300)
df_h1 = fetch_rates("NAS100", mt5.TIMEFRAME_H1, 300)

if df_m15 is not None and df_h1 is not None:
    # Calculate indicators
    df_m15, macro_bullish, macro_bearish = calculate_indicators(df_m15, df_h1)
    last_closed = df_m15.iloc[-2]
    
    # UI Layout: Top Metrics
    st.markdown(f"### Last Update: {live_time} | Live Bid: **{live_bid:.2f}**")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("M15 Close", f"{last_closed['close']:.2f}")
    col2.metric("Session VWAP", f"{last_closed['session_vwap']:.2f}")
    col3.metric("RSI (14)", f"{last_closed['rsi']:.2f}")
    col4.metric("H1 Trend", "BULLISH" if macro_bullish else "BEARISH" if macro_bearish else "NEUTRAL")

    # UI Layout: OBV Data
    st.markdown("### Volume Structure")
    v_col1, v_col2 = st.columns(2)
    v_col1.metric("OBV Raw", f"{last_closed['obv']:,.0f}")
    v_col2.metric("OBV 20 EMA", f"{last_closed['obv_ema20']:,.0f}")

    # Signal Evaluation
    obv_bullish = last_closed['obv'] > last_closed['obv_ema20']
    
    if macro_bullish and last_closed['close'] > last_closed['session_vwap'] and obv_bullish:
        st.success("🟢 **STRONG BULLISH SETUP ACTIVE:** Price > VWAP | OBV > 20 EMA | H1 Trend Bullish")
    elif macro_bearish and last_closed['close'] < last_closed['session_vwap'] and not obv_bullish:
        st.error("🔴 **STRONG BEARISH SETUP ACTIVE:** Price < VWAP | OBV < 20 EMA | H1 Trend Bearish")
    else:
        st.info("⚪ **NEUTRAL:** Waiting for indicator confluence.")

    # Show recent data table
    st.markdown("### Recent M15 Closes")
    st.dataframe(df_m15.tail(5)[['time', 'close', 'session_vwap', 'obv', 'obv_ema20', 'rsi']], use_container_width=True)

# Auto-refresh the Streamlit app every 5 seconds
time.sleep(5)
st.rerun()
