import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="SMC Backtest Engine", layout="centered")
st.title("📊 SMC 6-Month Backtest Results")
st.write("Simulating Optimal Trade Entry (OTE) Strategy on XAUUSD...")

# --- CONFIGURATION ---
CAPITAL = 15000.0
RISK_PER_TRADE = 0.02  # 2% Risk ($300 per trade)
SYMBOL = "GC=F"
PERIOD = "6mo"
INTERVAL = "1h"
WINDOW = 15  # Swing high/low lookback

def run_smc_backtest():
    with st.spinner(f"Fetching {PERIOD} of {INTERVAL} data for {SYMBOL}..."):
        df = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL, progress=False)
    
    if df.empty:
        st.error("Failed to fetch data from Yahoo Finance.")
        return

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    # Calculate rolling structural highs and lows
    df['Swing_High'] = df['High'].rolling(window=WINDOW*2, center=True).max()
    df['Swing_Low'] = df['Low'].rolling(window=WINDOW*2, center=True).min()
    
    # Forward fill to maintain the dealing range until a new swing forms
    df['Swing_High'] = df['Swing_High'].ffill()
    df['Swing_Low'] = df['Swing_Low'].ffill()
    
    in_trade = False
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trades = []
    
    for i in range(WINDOW, len(df)):
        current_close = df['Close'].iloc[i]
        high = df['Swing_High'].iloc[i]
        low = df['Swing_Low'].iloc[i]
        
        if pd.isna(high) or pd.isna(low) or high <= low:
            continue
            
        total_range = high - low
        eq = high - (total_range * 0.50)
        ote_low = high - (total_range * 0.382)
        ote_high = high - (total_range * 0.214)
        
        # ENTRY LOGIC: Bearish Setup (Price rallies into Premium OTE Supply)
        if not in_trade and current_close >= ote_low and current_close <= ote_high and current_close > eq:
            in_trade = True
            entry_price = current_close
            stop_loss = high + 2.50 # Stop just above the structural high
            take_profit = low       # Target the structural low liquidity
            
        # EXIT LOGIC
        elif in_trade:
            # Stopped Out
            if df['High'].iloc[i] >= stop_loss:
                trades.append({'Type': 'Loss', 'P&L': -1})
                in_trade = False
            # Target Hit
            elif df['Low'].iloc[i] <= take_profit:
                reward = (entry_price - take_profit) / (stop_loss - entry_price)
                trades.append({'Type': 'Win', 'P&L': reward})
                in_trade = False

    # --- CALCULATE METRICS ---
    total_trades = len(trades)
    if total_trades == 0:
        st.warning("No trades triggered with these parameters.")
        return
        
    wins = len([t for t in trades if t['Type'] == 'Win'])
    losses = total_trades - wins
    win_rate = (wins / total_trades) * 100
    
    # Calculate Risk-Adjusted Returns
    total_pnl_r = sum([t['P&L'] for t in trades])
    net_profit_usd = total_pnl_r * (CAPITAL * RISK_PER_TRADE)
    final_balance = CAPITAL + net_profit_usd
    
    # --- STREAMLIT METRICS DISPLAY ---
    st.success("Backtest Complete!")
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades Executed", total_trades)
    col2.metric("Win Rate", f"{win_rate:.2f}%")
    col3.metric("Wins vs Losses", f"{wins} W / {losses} L")
    
    st.divider()
    
    col4, col5 = st.columns(2)
    col4.metric("Starting Capital", f"${CAPITAL:,.2f}")
    col4.metric("Risk Per Trade", f"${CAPITAL * RISK_PER_TRADE:,.2f} (2%)")
    
    col6, col7 = st.columns(2)
    col6.metric("Total Net Return", f"${net_profit_usd:,.2f}", delta=f"{(net_profit_usd/CAPITAL)*100:.2f}%")
    col7.metric("Ending Balance", f"${final_balance:,.2f}")

if __name__ == "__main__":
    run_smc_backtest()
