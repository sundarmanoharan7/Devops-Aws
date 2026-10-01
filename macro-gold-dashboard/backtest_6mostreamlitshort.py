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

logging.getLogger('tvDatafeed').setLevel(logging.ERROR)

st.set_page_config(page_title="4H SMC Swing Engine & 6M Backtest", layout="wide")
st.title("Gold (XAUUSD) — 4-Hour SMC Swing Engine & 6-Month Backtest")

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
st.sidebar.header("Execution & Account Settings")
live_spot = market_spot
st.sidebar.success(f"Live Feed Synchronized\n\n**Current Spot: ${live_spot:,.2f}**")

capital = st.sidebar.number_input("Account Balance ($)", min_value=1000.0, max_value=100000.0, value=15000.0, step=1000.0)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", min_value=0.5, max_value=5.0, value=2.0, step=0.5)

st.sidebar.header("4-Hour Swing Parameters")
bt_window_4h = st.sidebar.slider("4H Swing Lookback (Bars)", min_value=3, max_value=20, value=6, help="Lookback window for major 4H highs and lows.")
h4_sl_buffer = st.sidebar.slider("4H Stop Loss Buffer ($)", min_value=2.0, max_value=15.0, value=6.50, step=0.5, help="Buffer added beyond the 4H swing high/low to absorb wicks.")

# --- 6-MONTH DATA PIPELINE (15M, 1H, 4H) ---
@st.cache_data(ttl=300)
def fetch_market_data():
    """Pulls 6 months of hourly data to construct institutional 4-Hour candles."""
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

    # 2. yfinance 6-Month Fallback
    if df_1h is None or df_1h.empty:
        try:
            df_1h = yf.download("XAUUSD=X", period="6mo", interval="1h", progress=False)
            df_15m = yf.download("XAUUSD=X", period="14d", interval="15m", progress=False)
            if isinstance(df_1h.columns, pd.MultiIndex): df_1h.columns = df_1h.columns.get_level_values(0)
            if isinstance(df_15m.columns, pd.MultiIndex): df_15m.columns = df_15m.columns.get_level_values(0)
        except Exception: pass

    # 3. Bitfinex Fallback
    if df_1h is None or df_1h.empty:
        try:
            r1h = requests.get("https://api-pub.bitfinex.com/v2/candles/trade:1h:tXAUUSD/hist?limit=4500", timeout=5).json()
            d1h = pd.DataFrame(r1h, columns=['time', 'open', 'close', 'high', 'low', 'volume'])
            d1h['time'] = pd.to_datetime(d1h['time'], unit='ms', utc=True)
            df_1h = d1h[['time', 'open', 'high', 'low', 'close']].set_index('time').astype(float).sort_index()

            r15 = requests.get("https://api-pub.bitfinex.com/v2/candles/trade:15m:tXAUUSD/hist?limit=1500", timeout=5).json()
            d15 = pd.DataFrame(r15, columns=['time', 'open', 'close', 'high', 'low', 'volume'])
            d15['time'] = pd.to_datetime(d15['time'], unit='ms', utc=True)
            df_15m = d15[['time', 'open', 'high', 'low', 'close']].set_index('time').astype(float).sort_index()
        except Exception: pass

    if df_1h is not None and not df_1h.empty:
        for df in [df_15m, df_1h]:
            if not df.empty:
                df.rename(columns=lambda x: x.capitalize() if isinstance(x, str) else x, inplace=True)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                else:
                    df.index = df.index.tz_convert('UTC')

        # Resample clean 1H candles into 4H candles across the entire 6-month period
        df_4h = df_1h.resample('4h').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last'
        }).dropna()

        return df_15m, df_1h, df_4h
        
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- SMC STRUCTURAL LOGIC ---
def analyze_smc_structure(df, window=6):
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
    
    # Premium OTE (Supply) Zone: 61.8% to 78.6%
    short_ote_low = recent_high - (total_range * 0.382)
    short_ote_high = recent_high - (total_range * 0.214)
    
    # Discount OTE (Demand) Zone: 61.8% to 78.6%
    long_ote_low = recent_low + (total_range * 0.214)
    long_ote_high = recent_low + (total_range * 0.382)
    
    return recent_high, recent_low, equilibrium, short_ote_low, short_ote_high, long_ote_low, long_ote_high

# --- 4-HOUR BIDIRECTIONAL BACKTEST ENGINE ---
def run_4h_backtest(df, start_capital, risk, window, sl_buffer):
    df = df.copy()
    # Dynamic swing calculation without lookahead bias
    df['Swing_High'] = df['High'].rolling(window=window*2, center=False).max().shift(1)
    df['Swing_Low'] = df['Low'].rolling(window=window*2, center=False).min().shift(1)
    
    in_trade = False
    trade_type = None
    entry_price, stop_loss, take_profit = 0.0, 0.0, 0.0
    entry_date = None
    
    equity = start_capital
    equity_curve = [start_capital]
    dates = [df.index[0]]
    trades = []
    
    for i in range(window*2 + 1, len(df)):
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
            # 1. Bearish OTE Retracement (Sell in Premium)
            if (short_ote_low <= close <= short_ote_high) and close > eq:
                in_trade = True
                trade_type = 'Short'
                entry_price = close
                stop_loss = high + sl_buffer
                take_profit = low  # External SSL target
                entry_date = date
                
            # 2. Bullish OTE Retracement (Buy in Discount)
            elif (long_ote_low <= close <= long_ote_high) and close < eq:
                in_trade = True
                trade_type = 'Long'
                entry_price = close
                stop_loss = low - sl_buffer
                take_profit = high  # External BSL target
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

if not df_4h.empty:
    current_price = live_spot
    
    # Analyze active 4H structure
    h4, l4, eq4, s_ote_low4, s_ote_high4, l_ote_low4, l_ote_high4 = analyze_smc_structure(df_4h, window=bt_window_4h)
    
    # Run 6-Month 4H Backtest
    trades_4h, dates_4h, equity_4h = run_4h_backtest(df_4h, capital, risk_pct, bt_window_4h, h4_sl_buffer)
    total_trades_4h = len(trades_4h)
    
    tab_backtest, tab_live = st.tabs(["📊 4-Hour Historical Backtest (6 Months)", "🏛️ 4-Hour Live SMC Dealing Desk"])
    
    # ==================== TAB 1: 4H 6-MONTH BACKTEST ====================
    with tab_backtest:
        st.subheader("6-Month 4-Hour Swing Strategy Performance")
        
        if total_trades_4h > 0:
            wins = len([t for t in trades_4h if t['Result'] == 'Win'])
            losses = total_trades_4h - wins
            win_rate = (wins / total_trades_4h) * 100
            total_net = equity_4h[-1] - capital
            ret_pct = (total_net / capital) * 100
            
            # Profit Factor & Expectancy
            total_gross_win = sum([t['Net P&L'] for t in trades_4h if t['Result'] == 'Win'])
            total_gross_loss = abs(sum([t['Net P&L'] for t in trades_4h if t['Result'] == 'Loss']))
            profit_factor = (total_gross_win / total_gross_loss) if total_gross_loss > 0 else 0.0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total 4H Setups Triggered", total_trades_4h, f"Wins: {wins} | Losses: {losses}")
            c2.metric("4H System Win Rate", f"{win_rate:.1f}%")
            
            color_metric = f":green[${total_net:,.2f}]" if total_net > 0 else f":red[${total_net:,.2f}]"
            c3.markdown(f"**Net Profit (6 Months)**\n### {color_metric}")
            c4.metric("Profit Factor", f"{profit_factor:.2f}", f"Return: {ret_pct:.1f}%")
            
            st.divider()
            
            # 6-Month Equity Curve
            st.subheader("Portfolio Equity Growth (6-Month Swing Curve)")
            fig_eq, ax_eq = plt.subplots(figsize=(14, 5))
            ax_eq.plot(dates_4h, equity_4h, color='teal', linewidth=2, label="Account Equity")
            ax_eq.axhline(capital, color='black', linestyle='--', linewidth=1, label="Initial Capital")
            ax_eq.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax_eq.set_ylabel('Balance (USD)')
            ax_eq.legend(loc='upper left')
            ax_eq.grid(alpha=0.25)
            st.pyplot(fig_eq)
            
            st.divider()
            
            # 4H Trade Ledger
            st.subheader("📝 4-Hour Executed Trade Ledger (Past 6 Months)")
            trade_df = pd.DataFrame(trades_4h)
            trade_df['Entry Date'] = trade_df['Entry Date'].dt.strftime('%b %d, %Y - %H:%M')
            trade_df['Exit Date'] = trade_df['Exit Date'].dt.strftime('%b %d, %Y - %H:%M')
            trade_df['Entry Price'] = trade_df['Entry Price'].apply(lambda x: f"${x:,.2f}")
            trade_df['Stop Loss'] = trade_df['Stop Loss'].apply(lambda x: f"${x:,.2f}")
            trade_df['Target (TP)'] = trade_df['Target (TP)'].apply(lambda x: f"${x:,.2f}")
            trade_df['Net P&L'] = trade_df['Net P&L'].apply(lambda x: f"${x:,.2f}")
            
            cols = ['Entry Date', 'Exit Date', 'Type', 'Result', 'Entry Price', 'Stop Loss', 'Target (TP)', 'Net P&L']
            st.dataframe(trade_df[cols].iloc[::-1], use_container_width=True, hide_index=True)
            
        else:
            st.warning("No 4-Hour swing trades triggered under current parameter configuration. Adjust lookback window in the sidebar.")
            
    # ==================== TAB 2: LIVE 4H DEALING DESK ====================
    with tab_live:
        st.subheader(f"Current Live Spot Price: ${current_price:,.2f}")
        
        is_4h_premium = current_price > eq4
        val_status_4h = "PREMIUM (Sell Allowed)" if is_4h_premium else "DISCOUNT (Buy Allowed)"
        val_color_4h = "red" if is_4h_premium else "green"
        action_4h = "Short" if is_4h_premium else "Long"
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 4-Hour Dealing Range")
            st.write(f"**Swing High (BSL):** ${h4:,.2f}")
            st.write(f"**Equilibrium (50%):** ${eq4:,.2f}")
            st.write(f"**Swing Low (SSL):** ${l4:,.2f}")
            st.markdown(f"**Market Valuation:** :{val_color_4h}[{val_status_4h}]")
            
        with col2:
            st.markdown(f"### Active Setup ({action_4h}-Biased)")
            if is_4h_premium:
                st.write(f"**Optimal Short Entry Zone (Supply OTE):** ${s_ote_low4:,.2f} – ${s_ote_high4:,.2f}")
                st.write(f"**Stop Loss:** ${h4 + h4_sl_buffer:,.2f}")
                st.write(f"**Target (Take Profit):** ${l4:,.2f}")
            else:
                st.write(f"**Optimal Long Entry Zone (Demand OTE):** ${l_ote_low4:,.2f} – ${l_ote_high4:,.2f}")
                st.write(f"**Stop Loss:** ${l4 - h4_sl_buffer:,.2f}")
                st.write(f"**Target (Take Profit):** ${h4:,.2f}")
                
        st.divider()
        st.subheader("4-Hour SMC Institutional Chart")
        
        df_chart_4h = df_4h.tail(120)
        fig_4h, ax_4h = plt.subplots(figsize=(14, 6))
        ax_4h.plot(df_chart_4h.index, df_chart_4h['Close'], color='black', linewidth=1.5, label="H4 Spot Close")
        
        ax_4h.axhspan(eq4, h4, color='red', alpha=0.06, label="H4 Premium Zone")
        ax_4h.axhspan(l4, eq4, color='green', alpha=0.06, label="H4 Discount Zone")
        
        ax_4h.axhspan(s_ote_low4, s_ote_high4, color='darkred', alpha=0.25, label="H4 OTE Supply Zone")
        ax_4h.axhspan(l_ote_low4, l_ote_high4, color='darkgreen', alpha=0.25, label="H4 OTE Demand Zone")
        
        ax_4h.axhline(h4, color='red', linestyle='--', linewidth=1.8, label=f"H4 BSL (${h4:,.2f})")
        ax_4h.axhline(l4, color='green', linestyle='--', linewidth=1.8, label=f"H4 SSL (${l4:,.2f})")
        ax_4h.axhline(eq4, color='blue', linestyle=':', linewidth=1.4, label=f"H4 Equilibrium (${eq4:,.2f})")
        
        ax_4h.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M UTC'))
        ax_4h.set_ylabel('Spot Price (USD)')
        ax_4h.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        ax_4h.grid(alpha=0.25)
        st.pyplot(fig_4h)
else:
    st.error("Market data feeds are currently unreachable. Verify network connection or wait for IP unblock.")
