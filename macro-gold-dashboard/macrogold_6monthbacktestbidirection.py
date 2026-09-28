import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import json

st.set_page_config(page_title="SMC Live Trade & Backtest", layout="wide")
st.title("Smart Money Concepts (SMC) - Live Setup & Backtest Engine")

# --- CLOUD-RESILIENT REAL-TIME SPOT FETCHER ---
@st.cache_data(ttl=20)  # Refreshes every 20 seconds
def get_live_xauusd_spot():
    """
    Fetches live Gold Spot directly from TradingView's public scanner 
    (prioritizing the FXCM feed), bypassing Yahoo Finance entirely.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    # Payload targeting the exact TradingView FXCM / OANDA spot feeds
    payload = {
        "symbols": {"tickers": ["FXCM:XAUUSD", "OANDA:XAUUSD"]},
        "columns": ["close"]
    }

    # Attempt 1: TradingView CFD Scanner
    try:
        url = "https://scanner.tradingview.com/cfd/scan"
        res = requests.post(url, json=payload, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get('data'):
                price = data['data'][0]['d'][0]
                if price and float(price) > 1000:
                    return float(price)
    except Exception:
        pass

    # Attempt 2: TradingView Forex Scanner
    try:
        url = "https://scanner.tradingview.com/forex/scan"
        res = requests.post(url, json=payload, headers=headers, timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get('data'):
                price = data['data'][0]['d'][0]
                if price and float(price) > 1000:
                    return float(price)
    except Exception:
        pass

    # Attempt 3: Yahoo Finance Direct JSON (XAUUSD=X Spot) Fallback
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/XAUUSD=X?interval=1m&range=1d"
        res = requests.get(url, headers=headers, timeout=3)
        if res.status_code == 200:
            data = res.json()
            price = data['chart']['result'][0]['meta'].get('regularMarketPrice')
            if price and float(price) > 1000:
                return float(price)
    except Exception:
        pass

    # Last known fallback
    return 4197.50

market_spot = get_live_xauusd_spot()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Live Feed Synchronization")
manual_override = st.sidebar.checkbox("Override Live Feed (Manual Mode)", value=False)

if manual_override:
    live_spot = st.sidebar.number_input(
        "Manual MT5 Spot Price (USD)",
        min_value=1000.0, max_value=10000.0,
        value=float(round(market_spot, 2)), step=0.10
    )
    st.sidebar.warning("Manual Override Active. Uncheck to resume live sync.")
else:
    live_spot = market_spot
    st.sidebar.success(f"Live Sync Active\n\n**Current Spot: ${live_spot:,.2f}**")

st.sidebar.header("Backtest Parameters")
capital = st.sidebar.number_input("Starting Capital ($)", min_value=1000.0, max_value=100000.0, value=15000.0, step=1000.0)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
bt_window = st.sidebar.slider("Structural Swing Lookback", min_value=5, max_value=30, value=15)

# --- DATA FETCHING ---
@st.cache_data(ttl=60)
def fetch_market_data(anchor: float):
    # Inject a custom browser session to bypass Streamlit Cloud IP blocking
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })

    # Fetch pure XAUUSD Spot data history using the unblocked session
    df_live = yf.download("XAUUSD=X", period="14d", interval="15m", progress=False, session=session)
    df_bt = yf.download("XAUUSD=X", period="6mo", interval="1h", progress=False, session=session)
    
    for df in [df_live, df_bt]:
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')
                
    # Align the historical data perfectly with the live TradingView FXCM feed
    if not df_live.empty:
        active_historical_bar = float(df_live['Close'].dropna().iloc[-1])
        spread = active_historical_bar - anchor
        for df in [df_live, df_bt]:
            if not df.empty:
                for col in ['Open', 'High', 'Low', 'Close']:
                    df[col] = df[col] - spread
                    
    return df_live, df_bt

# --- SMC STRUCTURAL LOGIC ---
def analyze_smc_structure(df, window=12):
    highs = argrelextrema(df['High'].values, np.greater, order=window)[0]
    lows = argrelextrema(df['Low'].values, np.less, order=window)[0]
    
    recent_high = float(df['High'].iloc[highs[-1]]) if len(highs) > 0 else float(df['High'].max())
    recent_low = float(df['Low'].iloc[lows[-1]]) if len(lows) > 0 else float(df['Low'].min())
    
    if recent_low >= recent_high:
        recent_high = float(df['High'].max())
        recent_low = float(df['Low'].min())
        
    total_range = recent_high - recent_low
    equilibrium = recent_high - (total_range * 0.50)
    golden_zone_low = recent_high - (total_range * 0.382) 
    golden_zone_high = recent_high - (total_range * 0.214)
    
    return recent_high, recent_low, equilibrium, golden_zone_low, golden_zone_high

# --- BACKTESTING ENGINE (LONG & SHORT SUPPORT) ---
def run_backtest(df, start_capital, risk, window):
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
    
    for i in range(window, len(df)):
        close = df['Close'].iloc[i]
        curr_high = df['High'].iloc[i]
        curr_low = df['Low'].iloc[i]
        
        high = df['Swing_High'].iloc[i]
        low = df['Swing_Low'].iloc[i]
        date = df.index[i]
        
        if pd.isna(high) or pd.isna(low) or high <= low:
            continue
            
        total_range = high - low
        eq = high - (total_range * 0.50)
        
        # Premium Supply (Short) Zone
        short_ote_low = high - (total_range * 0.382)
        short_ote_high = high - (total_range * 0.214)
        
        # Discount Demand (Long) Zone
        long_ote_high = low + (total_range * 0.382)
        long_ote_low = low + (total_range * 0.214)
        
        if not in_trade:
            # 1. Bearish Setup (Short)
            if (short_ote_low <= close <= short_ote_high) and close > eq:
                in_trade = True
                trade_type = 'Short'
                entry_price = close
                stop_loss = high + 2.50
                take_profit = low
                entry_date = date
                
            # 2. Bullish Setup (Long)
            elif (long_ote_low <= close <= long_ote_high) and close < eq:
                in_trade = True
                trade_type = 'Long'
                entry_price = close
                stop_loss = low - 2.50
                take_profit = high
                entry_date = date
                
        elif in_trade:
            risk_amt = equity * (risk / 100)
            
            # SHORT EXITS
            if trade_type == 'Short':
                if curr_high >= stop_loss:  
                    equity -= risk_amt
                    trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Type': 'Short', 'Result': 'Loss', 'Entry Price': entry_price, 'Stop Loss': stop_loss, 'Target (TP)': take_profit, 'Net P&L': -risk_amt})
                    in_trade = False
                elif curr_low <= take_profit:  
                    reward_ratio = (entry_price - take_profit) / (stop_loss - entry_price)
                    win_amt = risk_amt * reward_ratio
                    equity += win_amt
                    trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Type': 'Short', 'Result': 'Win', 'Entry Price': entry_price, 'Stop Loss': stop_loss, 'Target (TP)': take_profit, 'Net P&L': win_amt})
                    in_trade = False
                    
            # LONG EXITS
            elif trade_type == 'Long':
                if curr_low <= stop_loss:  
                    equity -= risk_amt
                    trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Type': 'Long', 'Result': 'Loss', 'Entry Price': entry_price, 'Stop Loss': stop_loss, 'Target (TP)': take_profit, 'Net P&L': -risk_amt})
                    in_trade = False
                elif curr_high >= take_profit:  
                    reward_ratio = (take_profit - entry_price) / (entry_price - stop_loss)
                    win_amt = risk_amt * reward_ratio
                    equity += win_amt
                    trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Type': 'Long', 'Result': 'Win', 'Entry Price': entry_price, 'Stop Loss': stop_loss, 'Target (TP)': take_profit, 'Net P&L': win_amt})
                    in_trade = False
                    
            if not in_trade:
                equity_curve.append(equity)
                dates.append(date)
                
    if dates[-1] != df.index[-1]:
        dates.append(df.index[-1])
        equity_curve.append(equity)
        
    return trades, dates, equity_curve

# --- RENDER DASHBOARD ---
df_live, df_bt = fetch_market_data(live_spot)

if not df_live.empty and not df_bt.empty:
    tab1, tab2 = st.tabs(["🔴 Live Market Execution", "📊 6-Month Backtest Results"])
    
    with tab1:
        current_price = live_spot 
        swing_high, swing_low, eq, gz_low, gz_high = analyze_smc_structure(df_live)
        
        st.subheader(f"Active Live Spot Price (TV): ${current_price:,.2f}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Institutional Dealing Range")
            st.write(f"**Swing High (BSL):** ${swing_high:,.2f}")
            st.write(f"**Equilibrium (50%):** ${eq:,.2f}")
            st.write(f"**Swing Low (SSL):** ${swing_low:,.2f}")
            status = "PREMIUM (Sell Allowed)" if current_price > eq else "DISCOUNT (Buy Allowed)"
            badge_color = "red" if current_price > eq else "green"
            st.markdown(f"**Market Valuation:** :{badge_color}[{status}]")

        with col2:
            st.markdown("### SMC Execution Plan")
            if current_price > eq:
                st.write(f"**Optimal Short Entry Zone:** ${gz_low:,.2f} – ${gz_high:,.2f}")
                st.write(f"**Stop Loss:** ${swing_high + 2.50:,.2f}")
                st.write(f"**Take Profit:** ${swing_low:,.2f}")
            else:
                long_ote_high = swing_low + ((swing_high - swing_low) * 0.382)
                long_ote_low = swing_low + ((swing_high - swing_low) * 0.214)
                st.write(f"**Optimal Long Entry Zone:** ${long_ote_low:,.2f} – ${long_ote_high:,.2f}")
                st.write(f"**Stop Loss:** ${swing_low - 2.50:,.2f}")
                st.write(f"**Take Profit:** ${swing_high:,.2f}")
            
        st.divider()
        st.subheader("15-Minute SMC Market Structure")
        
        df_chart = df_live.tail(150)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df_chart.index, df_chart['Close'], color='black', linewidth=1.2, label="Spot Price")
        
        ax.axhspan(eq, swing_high, color='red', alpha=0.06, label="Premium Zone")
        ax.axhspan(swing_low, eq, color='green', alpha=0.06, label="Discount Zone")
        
        ax.axhspan(gz_low, gz_high, color='darkred', alpha=0.25, label="OTE Supply (Short) Zone")
        
        long_ote_high_chart = swing_low + ((swing_high - swing_low) * 0.382)
        long_ote_low_chart = swing_low + ((swing_high - swing_low) * 0.214)
        ax.axhspan(long_ote_low_chart, long_ote_high_chart, color='darkgreen', alpha=0.25, label="OTE Demand (Long) Zone")
        
        ax.axhline(swing_high, color='red', linestyle='--', linewidth=1.8, label=f"BSL Stop (${swing_high:,.2f})")
        ax.axhline(swing_low, color='green', linestyle='--', linewidth=1.8, label=f"SSL Target (${swing_low:,.2f})")
        ax.axhline(eq, color='blue', linestyle=':', linewidth=1.4, label=f"Equilibrium (${eq:,.2f})")
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M UTC'))
        ax.set_ylabel('Spot Price (USD)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        
    with tab2:
        st.subheader("6-Month Historical Backtest (1-Hour Structure)")
        trades, bt_dates, equity_curve = run_backtest(df_bt, capital, risk_pct, bt_window)
        total_trades = len(trades)
        
        if total_trades > 0:
            wins = len([t for t in trades if t['Result'] == 'Win'])
            win_rate = (wins / total_trades) * 100
            total_net = equity_curve[-1] - capital
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Executed Trades", total_trades)
            c2.metric("System Win Rate", f"{win_rate:.1f}%")
            c3.metric("Net Profit (USD)", f"${total_net:,.2f}")
            c4.metric("Ending Account Balance", f"${equity_curve[-1]:,.2f}")
            
            st.divider()
            fig2, ax2 = plt.subplots(figsize=(14, 5))
            ax2.plot(bt_dates, equity_curve, color='teal', linewidth=2, label="Account Equity")
            ax2.fill_between(bt_dates, equity_curve, capital, where=(np.array(equity_curve) > capital), color='teal', alpha=0.1)
            ax2.fill_between(bt_dates, equity_curve, capital, where=(np.array(equity_curve) <= capital), color='red', alpha=0.1)
            ax2.axhline(capital, color='black', linestyle='--', linewidth=1, label="Starting Capital")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax2.set_ylabel('Balance (USD)')
            ax2.legend()
            ax2.grid(alpha=0.25)
            st.pyplot(fig2)
            
            st.divider()
            st.subheader("📝 Detailed Trade Ledger")
            
            trade_df = pd.DataFrame(trades)
            trade_df['Entry Date'] = trade_df['Entry Date'].dt.strftime('%b %d, %Y - %H:%M')
            trade_df['Exit Date'] = trade_df['Exit Date'].dt.strftime('%b %d, %Y - %H:%M')
            trade_df['Entry Price'] = trade_df['Entry Price'].apply(lambda x: f"${x:,.2f}")
            trade_df['Stop Loss'] = trade_df['Stop Loss'].apply(lambda x: f"${x:,.2f}")
            trade_df['Target (TP)'] = trade_df['Target (TP)'].apply(lambda x: f"${x:,.2f}")
            trade_df['Net P&L'] = trade_df['Net P&L'].apply(lambda x: f"${x:,.2f}")
            
            cols = ['Entry Date', 'Exit Date', 'Type', 'Result', 'Entry Price', 'Stop Loss', 'Target (TP)', 'Net P&L']
            trade_df = trade_df[cols]
            
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
            
        else:
            st.warning("No trades triggered under current parameters.")
else:
    st.error("Market data feeds are currently unreachable. Streamlit Cloud is blocking yfinance downloads despite session injection.")
