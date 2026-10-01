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

st.set_page_config(page_title="Multi-Timeframe SMC Engine", layout="wide")
st.title("Smart Money Concepts (SMC) — 15M & 4H Live Multi-Timeframe Desk")

# --- INITIALIZE TRADINGVIEW CONNECTION ---
@st.cache_resource
def get_tv_connection():
    try:
        return TvDatafeed()
    except Exception:
        return None

tv = get_tv_connection()

# --- REAL-TIME SPOT FETCHER ---
@st.cache_data(ttl=20)
def get_live_xauusd_spot():
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # ATTEMPT 1: TV CFD Scanner
    try:
        url = "https://scanner.tradingview.com/cfd/scan"
        payload = {"symbols": {"tickers": ["FXCM:XAUUSD", "OANDA:XAUUSD"]}, "columns": ["close"]}
        res = requests.post(url, json=payload, headers=headers, timeout=5)
        if res.status_code == 200:
            price = res.json().get('data', [{}])[0].get('d', [0])[0]
            if price > 1000: return float(price)
    except Exception: pass

    # ATTEMPT 2: MEXC PAXG
    try:
        url = "https://api.mexc.com/api/v3/ticker/price?symbol=PAXGUSDT"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            price = res.json().get('price')
            if price and float(price) > 1000: return float(price)
    except Exception: pass

    return 4150.00

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

st.sidebar.header("SMC & Risk Parameters")
capital = st.sidebar.number_input("Starting Capital ($)", min_value=1000.0, max_value=100000.0, value=15000.0, step=1000.0)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)
bt_window_15m = st.sidebar.slider("15M Swing Lookback", min_value=5, max_value=30, value=12)
bt_window_4h = st.sidebar.slider("4H Swing Lookback", min_value=3, max_value=20, value=6)

# --- STABLE HISTORICAL DATA FETCHER (15M, 1H, 4H) ---
@st.cache_data(ttl=300)
def fetch_market_data():
    """Fetches pure unshifted spot candles and resamples to 4H."""
    df_15m, df_1h = pd.DataFrame(), pd.DataFrame()

    # 1. TradingView API
    if tv is not None:
        exchanges = ['OANDA', 'FXCM', 'FOREXCOM']
        for exc in exchanges:
            try:
                df_15m = tv.get_hist(symbol='XAUUSD', exchange=exc, interval=Interval.in_15_minute, n_bars=1500)
                df_1h = tv.get_hist(symbol='XAUUSD', exchange=exc, interval=Interval.in_1_hour, n_bars=4500)
                if df_15m is not None and not df_15m.empty and df_1h is not None and not df_1h.empty:
                    break
            except Exception: continue

    # 2. Bitfinex Fallback
    if df_15m is None or df_15m.empty or df_1h is None or df_1h.empty:
        try:
            r15 = requests.get("https://api-pub.bitfinex.com/v2/candles/trade:15m:tXAUUSD/hist?limit=1500", timeout=5).json()
            d15 = pd.DataFrame(r15, columns=['time', 'open', 'close', 'high', 'low', 'volume'])
            d15['time'] = pd.to_datetime(d15['time'], unit='ms', utc=True)
            df_15m = d15[['time', 'open', 'high', 'low', 'close']].set_index('time').astype(float).sort_index()

            r1h = requests.get("https://api-pub.bitfinex.com/v2/candles/trade:1h:tXAUUSD/hist?limit=4500", timeout=5).json()
            d1h = pd.DataFrame(r1h, columns=['time', 'open', 'close', 'high', 'low', 'volume'])
            d1h['time'] = pd.to_datetime(d1h['time'], unit='ms', utc=True)
            df_1h = d1h[['time', 'open', 'high', 'low', 'close']].set_index('time').astype(float).sort_index()
        except Exception: pass

    # Clean data and synthesize 4H resampled candles
    if df_15m is not None and not df_15m.empty and df_1h is not None and not df_1h.empty:
        for df in [df_15m, df_1h]:
            df.rename(columns=lambda x: x.capitalize() if isinstance(x, str) else x, inplace=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

        df_4h = df_1h.resample('4h').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()

        return df_15m, df_1h, df_4h
        
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- SMC STRUCTURAL LOGIC (LOCKED TO CLOSED CANDLES) ---
def analyze_smc_structure(df, window=12):
    closed_df = df.iloc[:-1]
    
    highs = argrelextrema(closed_df['High'].values, np.greater, order=window)[0]
    lows = argrelextrema(closed_df['Low'].values, np.less, order=window)[0]
    
    recent_high = float(closed_df['High'].iloc[highs[-1]]) if len(highs) > 0 else float(closed_df['High'].max())
    recent_low = float(closed_df['Low'].iloc[lows[-1]]) if len(lows) > 0 else float(closed_df['Low'].min())
    
    if recent_low >= recent_high:
        recent_high = float(closed_df['High'].max())
        recent_low = float(closed_df['Low'].min())
        
    total_range = recent_high - recent_low
    equilibrium = recent_high - (total_range * 0.50)
    
    # Premium OTE (Supply) Zone: 61.8% to 78.6% retracement
    short_ote_low = recent_high - (total_range * 0.382)
    short_ote_high = recent_high - (total_range * 0.214)
    
    # Discount OTE (Demand) Zone: 61.8% to 78.6% retracement
    long_ote_low = recent_low + (total_range * 0.214)
    long_ote_high = recent_low + (total_range * 0.382)
    
    return recent_high, recent_low, equilibrium, short_ote_low, short_ote_high, long_ote_low, long_ote_high

# --- 1H BACKTESTING ENGINE ---
def run_backtest(df, start_capital, risk, window):
    df = df.copy()
    df['Swing_High'] = df['High'].rolling(window=window*2, center=False).max().ffill()
    df['Swing_Low'] = df['Low'].rolling(window=window*2, center=False).min().ffill()
    
    in_trade = False
    trade_type = None
    entry_price, stop_loss, take_profit = 0.0, 0.0, 0.0
    entry_date = None
    
    equity = start_capital
    equity_curve = [start_capital]
    dates = [df.index[0]]
    trades = []
    
    for i in range(window*2, len(df)):
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
        
        short_ote_low = high - (total_range * 0.382)
        short_ote_high = high - (total_range * 0.214)
        long_ote_low = low + (total_range * 0.214)
        long_ote_high = low + (total_range * 0.382)
        
        if not in_trade:
            # Bearish OTE trigger
            if (short_ote_low <= close <= short_ote_high) and close > eq:
                in_trade = True
                trade_type = 'Short'
                entry_price = close
                stop_loss = high + 2.50
                take_profit = low
                entry_date = date
            # Bullish OTE trigger
            elif (long_ote_low <= close <= long_ote_high) and close < eq:
                in_trade = True
                trade_type = 'Long'
                entry_price = close
                stop_loss = low - 2.50
                take_profit = high
                entry_date = date
                
        elif in_trade:
            risk_amt = equity * (risk / 100)
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
df_15m, df_1h, df_4h = fetch_market_data()

if not df_15m.empty and not df_1h.empty and not df_4h.empty:
    tab_exec, tab_backtest = st.tabs(["🔴 Live Multi-Timeframe SMC Desk", "📊 Historical Backtest Results"])
    
    with tab_exec:
        current_price = live_spot 
        st.subheader(f"Active Live Spot Price: ${current_price:,.2f}")
        
        # Calculate Structures
        h15, l15, eq15, s_ote_low15, s_ote_high15, l_ote_low15, l_ote_high15 = analyze_smc_structure(df_15m, window=bt_window_15m)
        h4, l4, eq4, s_ote_low4, s_ote_high4, l_ote_low4, l_ote_high4 = analyze_smc_structure(df_4h, window=bt_window_4h)
        
        # Multi-timeframe sub-tabs
        subtab_4h, subtab_15m = st.tabs(["🏛️ 4-Hour Macro Swing SMC", "⚡ 15-Minute Intraday SMC"])
        
        # ==================== 4-HOUR SMC SECTION ====================
        with subtab_4h:
            col1, col2 = st.columns(2)
            is_4h_premium = current_price > eq4
            val_status_4h = "PREMIUM (Sell Allowed)" if is_4h_premium else "DISCOUNT (Buy Allowed)"
            val_color_4h = "red" if is_4h_premium else "green"
            
            with col1:
                st.markdown("### 4-Hour Institutional Dealing Range")
                st.write(f"**Swing High (BSL):** ${h4:,.2f}")
                st.write(f"**Equilibrium (50%):** ${eq4:,.2f}")
                st.write(f"**Swing Low (SSL):** ${l4:,.2f}")
                st.markdown(f"**Market Valuation:** :{val_color_4h}[{val_status_4h}]")
                
            with col2:
                action_4h = "Short" if is_4h_premium else "Long"
                st.markdown(f"### 4-Hour Macro Execution Plan ({action_4h}-Biased)")
                if is_4h_premium:
                    st.write(f"**Optimal Short Entry Zone (Supply OTE):** ${s_ote_low4:,.2f} – ${s_ote_high4:,.2f}")
                    st.write(f"**Stop Loss:** ${h4 + 5.00:,.2f}")
                    st.write(f"**Take Profit:** ${l4:,.2f}")
                else:
                    st.write(f"**Optimal Long Entry Zone (Demand OTE):** ${l_ote_low4:,.2f} – ${l_ote_high4:,.2f}")
                    st.write(f"**Stop Loss:** ${l4 - 5.00:,.2f}")
                    st.write(f"**Take Profit:** ${h4:,.2f}")
                    
            st.divider()
            st.subheader("4-Hour SMC Market Structure Chart")
            
            df_chart_4h = df_4h.tail(120)
            fig_4h, ax_4h = plt.subplots(figsize=(14, 6))
            ax_4h.plot(df_chart_4h.index, df_chart_4h['Close'], color='black', linewidth=1.5, label="H4 Spot Close")
            
            # Premium & Discount shading
            ax_4h.axhspan(eq4, h4, color='red', alpha=0.06, label="H4 Premium Zone")
            ax_4h.axhspan(l4, eq4, color='green', alpha=0.06, label="H4 Discount Zone")
            
            # OTE Zones
            ax_4h.axhspan(s_ote_low4, s_ote_high4, color='darkred', alpha=0.25, label="H4 OTE Supply (Short) Zone")
            ax_4h.axhspan(l_ote_low4, l_ote_high4, color='darkgreen', alpha=0.25, label="H4 OTE Demand (Long) Zone")
            
            # Structural Lines
            ax_4h.axhline(h4, color='red', linestyle='--', linewidth=1.8, label=f"H4 BSL (${h4:,.2f})")
            ax_4h.axhline(l4, color='green', linestyle='--', linewidth=1.8, label=f"H4 SSL (${l4:,.2f})")
            ax_4h.axhline(eq4, color='blue', linestyle=':', linewidth=1.4, label=f"H4 Equilibrium (${eq4:,.2f})")
            
            ax_4h.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M UTC'))
            ax_4h.set_ylabel('Spot Price (USD)')
            ax_4h.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
            ax_4h.grid(alpha=0.25)
            st.pyplot(fig_4h)

        # ==================== 15-MINUTE SMC SECTION ====================
        with subtab_15m:
            col1, col2 = st.columns(2)
            is_15m_premium = current_price > eq15
            val_status_15m = "PREMIUM (Sell Allowed)" if is_15m_premium else "DISCOUNT (Buy Allowed)"
            val_color_15m = "red" if is_15m_premium else "green"
            
            with col1:
                st.markdown("### 15-Minute Institutional Dealing Range")
                st.write(f"**Swing High (BSL):** ${h15:,.2f}")
                st.write(f"**Equilibrium (50%):** ${eq15:,.2f}")
                st.write(f"**Swing Low (SSL):** ${l15:,.2f}")
                st.markdown(f"**Market Valuation:** :{val_color_15m}[{val_status_15m}]")
                
            with col2:
                action_15m = "Short" if is_15m_premium else "Long"
                st.markdown(f"### 15-Minute Intraday Execution Plan ({action_15m}-Biased)")
                if is_15m_premium:
                    st.write(f"**Optimal Short Entry Zone (Supply OTE):** ${s_ote_low15:,.2f} – ${s_ote_high15:,.2f}")
                    st.write(f"**Stop Loss:** ${h15 + 2.50:,.2f}")
                    st.write(f"**Take Profit:** ${l15:,.2f}")
                else:
                    st.write(f"**Optimal Long Entry Zone (Demand OTE):** ${l_ote_low15:,.2f} – ${l_ote_high15:,.2f}")
                    st.write(f"**Stop Loss:** ${l15 - 2.50:,.2f}")
                    st.write(f"**Take Profit:** ${h15:,.2f}")
                    
            st.divider()
            st.subheader("15-Minute SMC Market Structure Chart")
            
            df_chart_15m = df_15m.tail(150)
            fig_15m, ax_15m = plt.subplots(figsize=(14, 6))
            ax_15m.plot(df_chart_15m.index, df_chart_15m['Close'], color='black', linewidth=1.2, label="15m Spot Price")
            
            ax_15m.axhspan(eq15, h15, color='red', alpha=0.06, label="15m Premium Zone")
            ax_15m.axhspan(l15, eq15, color='green', alpha=0.06, label="15m Discount Zone")
            
            ax_15m.axhspan(s_ote_low15, s_ote_high15, color='darkred', alpha=0.25, label="15m OTE Supply Zone")
            ax_15m.axhspan(l_ote_low15, l_ote_high15, color='darkgreen', alpha=0.25, label="15m OTE Demand Zone")
            
            ax_15m.axhline(h15, color='red', linestyle='--', linewidth=1.8, label=f"15m BSL (${h15:,.2f})")
            ax_15m.axhline(l15, color='green', linestyle='--', linewidth=1.8, label=f"15m SSL (${l15:,.2f})")
            ax_15m.axhline(eq15, color='blue', linestyle=':', linewidth=1.4, label=f"15m Equilibrium (${eq15:,.2f})")
            
            ax_15m.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M UTC'))
            ax_15m.set_ylabel('Spot Price (USD)')
            ax_15m.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
            ax_15m.grid(alpha=0.25)
            st.pyplot(fig_15m)

    # ==================== BACKTEST SECTION ====================
    with tab_backtest:
        st.subheader("Historical Backtest Results")
        trades, bt_dates, equity_curve = run_backtest(df_1h, capital, risk_pct, bt_window_15m)
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
            
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No trades triggered under current parameters.")
else:
    st.error("All data feeds are currently unreachable. Please verify network connection or wait for IP unblock.")
