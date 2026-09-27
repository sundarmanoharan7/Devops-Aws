import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

st.set_page_config(page_title="Macro-SMC Unified Engine", layout="wide")
st.title("Unified Execution: Macro Filter + SMC Engine")

CURRENT_CPI = 3.35  # Static CPI baseline

# --- 1. REAL-TIME SPOT FETCHER ---
@st.cache_data(ttl=20)
def get_live_xauusd_spot():
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/XAUUSD=X?interval=1m&range=1d"
        res = requests.get(url, headers=headers, timeout=3)
        if res.status_code == 200:
            data = res.json()
            price = data['chart']['result'][0]['meta'].get('regularMarketPrice')
            if price and float(price) > 1000: return float(price)
    except: pass
    
    try:
        url = "https://api.metals.live/v1/spot/gold"
        res = requests.get(url, headers=headers, timeout=3)
        if res.status_code == 200:
            data = res.json()
            price = data[0].get('price') if isinstance(data, list) else data.get('price')
            if price and float(price) > 1000: return float(price)
    except: pass

    return 4285.00

market_spot = get_live_xauusd_spot()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Strategy Parameters")
capital = st.sidebar.number_input("Starting Capital ($)", min_value=1000.0, value=15000.0, step=1000.0)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.5)
bt_window = st.sidebar.slider("SMC Structural Lookback (1H)", 5, 30, 15)

# --- 2. UNIFIED DATA PIPELINE (MACRO + SMC) ---
@st.cache_data(ttl=3600)
def fetch_unified_data():
    # A. Fetch Daily Macro Data
    tickers = {"DX-Y.NYB": "DXY", "^TNX": "Nominal_10Y"}
    series_list = []
    for t, name in tickers.items():
        try:
            d = yf.download(t, period="6mo", interval="1d", progress=False)
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                s = d['Close']
                s.name = name
                series_list.append(s)
        except: pass
        
    df_macro = pd.concat(series_list, axis=1).ffill().dropna()
    df_macro['Real_Yield'] = df_macro['Nominal_10Y'] - CURRENT_CPI
    df_macro['DXY_SMA20'] = df_macro['DXY'].rolling(window=20).mean()
    df_macro['Yield_SMA20'] = df_macro['Real_Yield'].rolling(window=20).mean()
    
    # Generate Daily Macro Signal (1 = Long, -1 = Short, 0 = Cash)
    bull = (df_macro['DXY'] < df_macro['DXY_SMA20']) & (df_macro['Real_Yield'] < df_macro['Yield_SMA20'])
    bear = (df_macro['DXY'] > df_macro['DXY_SMA20']) & (df_macro['Real_Yield'] > df_macro['Yield_SMA20'])
    df_macro['Macro_Signal'] = np.select([bull, bear], [1, -1], default=0)
    
    # Standardize index to extract raw date for merging
    if df_macro.index.tz is not None: df_macro.index = df_macro.index.tz_localize(None)
    df_macro['Date_Only'] = df_macro.index.normalize()

    # B. Fetch 1-Hour SMC Data
    df_smc = yf.download("GC=F", period="6mo", interval="1h", progress=False)
    if not df_smc.empty and isinstance(df_smc.columns, pd.MultiIndex):
        df_smc.columns = df_smc.columns.get_level_values(0)
    
    if df_smc.index.tz is not None: df_smc.index = df_smc.index.tz_localize(None)
    df_smc['Date_Only'] = df_smc.index.normalize()
    
    # C. Merge Macro Filter onto Hourly SMC Data
    df_unified = pd.merge(df_smc, df_macro[['Date_Only', 'Macro_Signal']], on='Date_Only', how='left')
    df_unified.index = df_smc.index
    df_unified['Macro_Signal'] = df_unified['Macro_Signal'].ffill().fillna(0) # Forward fill weekends
    
    return df_unified, df_macro

# --- 3. FILTERED BACKTEST ENGINE ---
def run_macro_smc_backtest(df, start_capital, risk, window):
    df = df.copy()
    df['Swing_High'] = df['High'].rolling(window=window*2, center=True).max().ffill()
    df['Swing_Low'] = df['Low'].rolling(window=window*2, center=True).min().ffill()
    
    in_trade = False
    trade_type = None
    entry_price, stop_loss, take_profit = 0.0, 0.0, 0.0
    entry_date = None
    
    equity = start_capital
    equity_curve = [start_capital]
    dates = [df.index[0]]
    trades = []
    skipped_trades = 0 # Track how many bad setups the macro filter prevented
    
    for i in range(window, len(df)):
        close = df['Close'].iloc[i]
        curr_high = df['High'].iloc[i]
        curr_low = df['Low'].iloc[i]
        high = df['Swing_High'].iloc[i]
        low = df['Swing_Low'].iloc[i]
        macro_bias = df['Macro_Signal'].iloc[i]
        date = df.index[i]
        
        if pd.isna(high) or pd.isna(low) or high <= low: continue
            
        total_range = high - low
        eq = high - (total_range * 0.50)
        
        short_ote_low = high - (total_range * 0.382)
        short_ote_high = high - (total_range * 0.214)
        
        long_ote_high = low + (total_range * 0.382)
        long_ote_low = low + (total_range * 0.214)
        
        # --- MACRO FILTERED ENTRIES ---
        if not in_trade:
            is_short_setup = (short_ote_low <= close <= short_ote_high) and close > eq
            is_long_setup = (long_ote_low <= close <= long_ote_high) and close < eq
            
            if is_short_setup:
                if macro_bias == -1: # MACRO AGREES
                    in_trade, trade_type = True, 'Short'
                    entry_price, stop_loss, take_profit, entry_date = close, high + 2.50, low, date
                else: 
                    skipped_trades += 1 # MACRO REJECTS
                    
            elif is_long_setup:
                if macro_bias == 1: # MACRO AGREES
                    in_trade, trade_type = True, 'Long'
                    entry_price, stop_loss, take_profit, entry_date = close, low - 2.50, high, date
                else: 
                    skipped_trades += 1 # MACRO REJECTS
                
        # --- EXITS (Unchanged) ---
        elif in_trade:
            risk_amt = equity * (risk / 100)
            
            if trade_type == 'Short':
                if curr_high >= stop_loss:
                    equity -= risk_amt
                    trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Macro Bias': 'Bearish (-1)', 'Type': 'Short', 'Result': 'Loss', 'Net P&L': -risk_amt})
                    in_trade = False
                elif curr_low <= take_profit:
                    equity += risk_amt * ((entry_price - take_profit) / (stop_loss - entry_price))
                    trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Macro Bias': 'Bearish (-1)', 'Type': 'Short', 'Result': 'Win', 'Net P&L': risk_amt * ((entry_price - take_profit) / (stop_loss - entry_price))})
                    in_trade = False
                    
            elif trade_type == 'Long':
                if curr_low <= stop_loss:
                    equity -= risk_amt
                    trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Macro Bias': 'Bullish (+1)', 'Type': 'Long', 'Result': 'Loss', 'Net P&L': -risk_amt})
                    in_trade = False
                elif curr_high >= take_profit:
                    equity += risk_amt * ((take_profit - entry_price) / (entry_price - stop_loss))
                    trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Macro Bias': 'Bullish (+1)', 'Type': 'Long', 'Result': 'Win', 'Net P&L': risk_amt * ((take_profit - entry_price) / (entry_price - stop_loss))})
                    in_trade = False
                    
            if not in_trade:
                equity_curve.append(equity)
                dates.append(date)
                
    if dates[-1] != df.index[-1]:
        dates.append(df.index[-1])
        equity_curve.append(equity)
        
    return trades, dates, equity_curve, skipped_trades

# --- 4. RENDER DASHBOARD ---
df_unified, df_macro = fetch_unified_data()

if not df_unified.empty:
    st.subheader("6-Month Results: Macro-Filtered SMC Executions")
    
    trades, bt_dates, equity_curve, skipped = run_macro_smc_backtest(df_unified, capital, risk_pct, bt_window)
    total_trades = len(trades)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trades Filtered (Saved by Macro)", skipped, help="SMC setups ignored because they fought the Daily DXY/Yield trend.")
    c2.metric("Executed Trades", total_trades)
    
    if total_trades > 0:
        win_rate = (len([t for t in trades if t['Result'] == 'Win']) / total_trades) * 100
        total_net = equity_curve[-1] - capital
        
        c3.metric("Filtered Win Rate", f"{win_rate:.1f}%")
        c4.metric("Ending Account Balance", f"${equity_curve[-1]:,.2f}", f"${total_net:,.2f}")
        
        st.divider()
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(bt_dates, equity_curve, color='teal', linewidth=2, label="Account Equity")
        ax.fill_between(bt_dates, equity_curve, capital, where=(np.array(equity_curve) > capital), color='teal', alpha=0.1)
        ax.fill_between(bt_dates, equity_curve, capital, where=(np.array(equity_curve) <= capital), color='red', alpha=0.1)
        ax.axhline(capital, color='black', linestyle='--', linewidth=1, label="Starting Capital")
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.set_ylabel('Balance (USD)')
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        
        st.divider()
        st.subheader("📝 Macro-Approved Trade Ledger")
        trade_df = pd.DataFrame(trades)
        trade_df['Entry Date'] = trade_df['Entry Date'].dt.strftime('%b %d, %Y - %H:%M')
        trade_df['Exit Date'] = trade_df['Exit Date'].dt.strftime('%b %d, %Y - %H:%M')
        trade_df['Net P&L'] = trade_df['Net P&L'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(trade_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No trades triggered. The Macro trend and SMC structure did not align during this 6-month period.")
else:
    st.error("Market data feeds are currently unreachable.")
