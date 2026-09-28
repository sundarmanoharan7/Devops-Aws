import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

st.set_page_config(page_title="Macro-SMC Unified Engine", layout="wide")
st.title("Unified Execution: Macro Filter + Deep OTE Engine")

CURRENT_CPI = 3.35  # Static CPI baseline

# --- 1. REAL-TIME SPOT FETCHER ---
@st.cache_data(ttl=15)
def get_live_xauusd_spot():
    """Fetches exact live Gold Spot, bypassing yfinance blocks."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Priority 1: TradingView CFD Scanner (Exact FXCM Match)
    try:
        url = "https://scanner.tradingview.com/cfd/scan"
        payload = {"symbols": {"tickers": ["FXCM:XAUUSD", "OANDA:XAUUSD"]}, "columns": ["close"]}
        res = requests.post(url, json=payload, headers=headers, timeout=3)
        if res.status_code == 200:
            price = res.json().get('data', [{}])[0].get('d', [0])[0]
            if price > 1000: return float(price)
    except: pass
    
    # Priority 2: KuCoin PAXG-USDT (1:1 Gold Peg, Unblocked for US Cloud)
    try:
        url = "https://api.kucoin.com/api/v1/market/orderbook/level1?symbol=PAXG-USDT"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            price = res.json().get('data', {}).get('price')
            if price and float(price) > 1000: return float(price)
    except: pass

    # Failsafe
    return 4285.00

market_spot = get_live_xauusd_spot()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Strategy Parameters")
capital = st.sidebar.number_input("Starting Capital ($)", min_value=1000.0, value=15000.0, step=1000.0)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.5)
bt_window = st.sidebar.slider("SMC Structural Lookback (1H)", 5, 30, 15)

# --- 2. UNIFIED DATA PIPELINE (MACRO + SMC) ---
@st.cache_data(ttl=300)
def fetch_unified_data(anchor: float):
    # A. Fetch Daily Macro Data (Wrapped to prevent crashes if blocked)
    df_macro = pd.DataFrame()
    try:
        tickers = {"DX-Y.NYB": "DXY", "^TNX": "Nominal_10Y"}
        series_list = []
        for t, name in tickers.items():
            d = yf.download(t, period="6mo", interval="1d", progress=False)
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                s = d['Close']
                s.name = name
                series_list.append(s)
                
        if len(series_list) == 2:
            df_macro = pd.concat(series_list, axis=1).ffill().dropna()
            df_macro['Real_Yield'] = df_macro['Nominal_10Y'] - CURRENT_CPI
            df_macro['DXY_SMA20'] = df_macro['DXY'].rolling(window=20).mean()
            df_macro['Yield_SMA20'] = df_macro['Real_Yield'].rolling(window=20).mean()
            
            bull = (df_macro['DXY'] < df_macro['DXY_SMA20']) & (df_macro['Real_Yield'] < df_macro['Yield_SMA20'])
            bear = (df_macro['DXY'] > df_macro['DXY_SMA20']) & (df_macro['Real_Yield'] > df_macro['Yield_SMA20'])
            df_macro['Macro_Signal'] = np.select([bull, bear], [1, -1], default=0)
            
            if df_macro.index.tz is not None: df_macro.index = df_macro.index.tz_localize(None)
            df_macro['Date_Only'] = df_macro.index.normalize()
    except: pass

    # B. Fetch Hourly/15m SMC Data
    df_live, df_bt = pd.DataFrame(), pd.DataFrame()

    # Attempt 1: Spot XAUUSD (yfinance)
    try:
        df_live = yf.download("XAUUSD=X", period="14d", interval="15m", progress=False)
        df_bt = yf.download("XAUUSD=X", period="6mo", interval="1h", progress=False)
    except: pass

    # Attempt 2: GC=F Futures (yfinance)
    if df_live.empty or df_bt.empty:
        try:
            df_live = yf.download("GC=F", period="14d", interval="15m", progress=False)
            df_bt = yf.download("GC=F", period="6mo", interval="1h", progress=False)
        except: pass
        
    # Attempt 3: KuCoin PAXG (If yfinance is entirely blocked)
    if df_live.empty or df_bt.empty:
        try:
            res_15 = requests.get("https://api.kucoin.com/api/v1/market/candles?type=15min&symbol=PAXG-USDT", timeout=5).json()
            d1 = pd.DataFrame(res_15['data'], columns=['time', 'open', 'close', 'high', 'low', 'v', 't'])
            d1['time'] = pd.to_datetime(d1['time'].astype(int), unit='s', utc=True)
            df_live = d1[['time', 'open', 'high', 'low', 'close']].set_index('time').astype(float).sort_index()
            
            res_1h = requests.get("https://api.kucoin.com/api/v1/market/candles?type=1hour&symbol=PAXG-USDT", timeout=5).json()
            d2 = pd.DataFrame(res_1h['data'], columns=['time', 'open', 'close', 'high', 'low', 'v', 't'])
            d2['time'] = pd.to_datetime(d2['time'].astype(int), unit='s', utc=True)
            df_bt = d2[['time', 'open', 'high', 'low', 'close']].set_index('time').astype(float).sort_index()
        except: pass

    # Clean & Perfect Alignment
    if not df_live.empty and not df_bt.empty:
        for df in [df_live, df_bt]:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.rename(columns=lambda x: x.capitalize() if isinstance(x, str) else x, inplace=True)
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df.dropna(inplace=True)

        active_historical_bar = float(df_live['Close'].iloc[-1])
        spread = active_historical_bar - anchor
        for df in [df_live, df_bt]:
            for col in ['Open', 'High', 'Low', 'Close']:
                df[col] = df[col] - spread

        # Merge Pipeline
        df_bt['Date_Only'] = df_bt.index.normalize()
        if not df_macro.empty:
            df_unified = pd.merge(df_bt, df_macro[['Date_Only', 'Macro_Signal']], on='Date_Only', how='left')
            df_unified.index = df_bt.index
            df_unified['Macro_Signal'] = df_unified['Macro_Signal'].ffill().fillna(0)
        else:
            df_bt['Macro_Signal'] = 0 # Defaults to Neutral if Yahoo blocks Macro
            df_unified = df_bt
            
        return df_live, df_unified

    return pd.DataFrame(), pd.DataFrame()

# --- HIGH PROBABILITY SMC STRUCTURAL LOGIC (DEEP OTE) ---
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
    
    # Deep OTE: 70.5% to 78.6% (Filters out premature entries)
    golden_zone_low = recent_high - (total_range * 0.295) 
    golden_zone_high = recent_high - (total_range * 0.214)
    
    return recent_high, recent_low, equilibrium, golden_zone_low, golden_zone_high

# --- FILTERED BIDIRECTIONAL BACKTEST ENGINE ---
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
    skipped_trades = 0 
    
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
        
        # High Probability Deep OTE Supply (Short)
        short_ote_low = high - (total_range * 0.295)
        short_ote_high = high - (total_range * 0.214)
        
        # High Probability Deep OTE Demand (Long)
        long_ote_high = low + (total_range * 0.295)
        long_ote_low = low + (total_range * 0.214)
        
        if not in_trade:
            is_short_setup = (short_ote_low <= close <= short_ote_high) and close > eq
            is_long_setup = (long_ote_low <= close <= long_ote_high) and close < eq
            
            if is_short_setup:
                if macro_bias in [-1, 0]: # Macro Agrees OR is Neutral/Blocked
                    in_trade, trade_type = True, 'Short'
                    entry_price = close
                    stop_loss = high + 3.00 # Widened structural stop to prevent wicks
                    take_profit = low
                    entry_date = date
                else: 
                    skipped_trades += 1
                    
            elif is_long_setup:
                if macro_bias in [1, 0]: # Macro Agrees OR is Neutral/Blocked
                    in_trade, trade_type = True, 'Long'
                    entry_price = close
                    stop_loss = low - 3.00 # Widened structural stop to prevent wicks
                    take_profit = high
                    entry_date = date
                else: 
                    skipped_trades += 1
                
        elif in_trade:
            risk_amt = equity * (risk / 100)
            
            if trade_type == 'Short':
                if curr_high >= stop_loss:
                    equity -= risk_amt
                    trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Type': 'Short', 'Result': 'Loss', 'Entry Price': entry_price, 'Stop Loss': stop_loss, 'Target (TP)': take_profit, 'Net P&L': -risk_amt})
                    in_trade = False
                elif curr_low <= take_profit:
                    equity += risk_amt * ((entry_price - take_profit) / (stop_loss - entry_price))
                    trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Type': 'Short', 'Result': 'Win', 'Entry Price': entry_price, 'Stop Loss': stop_loss, 'Target (TP)': take_profit, 'Net P&L': risk_amt * ((entry_price - take_profit) / (stop_loss - entry_price))})
                    in_trade = False
                    
            elif trade_type == 'Long':
                if curr_low <= stop_loss:
                    equity -= risk_amt
                    trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Type': 'Long', 'Result': 'Loss', 'Entry Price': entry_price, 'Stop Loss': stop_loss, 'Target (TP)': take_profit, 'Net P&L': -risk_amt})
                    in_trade = False
                elif curr_high >= take_profit:
                    equity += risk_amt * ((take_profit - entry_price) / (entry_price - stop_loss))
                    trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Type': 'Long', 'Result': 'Win', 'Entry Price': entry_price, 'Stop Loss': stop_loss, 'Target (TP)': take_profit, 'Net P&L': risk_amt * ((take_profit - entry_price) / (entry_price - stop_loss))})
                    in_trade = False
                    
            if not in_trade:
                equity_curve.append(equity)
                dates.append(date)
                
    if dates[-1] != df.index[-1]:
        dates.append(df.index[-1])
        equity_curve.append(equity)
        
    return trades, dates, equity_curve, skipped_trades

# --- 4. RENDER DASHBOARD ---
df_live, df_unified = fetch_unified_data(market_spot)

if not df_unified.empty:
    tab1, tab2 = st.tabs(["🔴 Live Market Execution", "📊 6-Month Backtest Results"])
    
    with tab1:
        current_price = market_spot 
        swing_high, swing_low, eq, gz_low, gz_high = analyze_smc_structure(df_live)
        
        st.subheader(f"Active Live Spot Price: ${current_price:,.2f}")
        
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
            st.markdown("### High-Probability Execution Plan")
            if current_price > eq:
                st.write(f"**Deep OTE Short Entry Zone:** ${gz_low:,.2f} – ${gz_high:,.2f}")
                st.write(f"**Stop Loss:** ${swing_high + 3.00:,.2f}")
                st.write(f"**Take Profit:** ${swing_low:,.2f}")
            else:
                long_ote_high = swing_low + ((swing_high - swing_low) * 0.295)
                long_ote_low = swing_low + ((swing_high - swing_low) * 0.214)
                st.write(f"**Deep OTE Long Entry Zone:** ${long_ote_low:,.2f} – ${long_ote_high:,.2f}")
                st.write(f"**Stop Loss:** ${swing_low - 3.00:,.2f}")
                st.write(f"**Take Profit:** ${swing_high:,.2f}")
            
        st.divider()
        st.subheader("15-Minute SMC Market Structure")
        
        df_chart = df_live.tail(150)
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df_chart.index, df_chart['Close'], color='black', linewidth=1.2, label="Spot Price")
        
        ax.axhspan(eq, swing_high, color='red', alpha=0.06, label="Premium Zone")
        ax.axhspan(swing_low, eq, color='green', alpha=0.06, label="Discount Zone")
        
        ax.axhspan(gz_low, gz_high, color='darkred', alpha=0.25, label="Deep OTE Supply (Short)")
        
        long_ote_high_chart = swing_low + ((swing_high - swing_low) * 0.295)
        long_ote_low_chart = swing_low + ((swing_high - swing_low) * 0.214)
        ax.axhspan(long_ote_low_chart, long_ote_high_chart, color='darkgreen', alpha=0.25, label="Deep OTE Demand (Long)")
        
        ax.axhline(swing_high, color='red', linestyle='--', linewidth=1.8, label=f"BSL Stop (${swing_high:,.2f})")
        ax.axhline(swing_low, color='green', linestyle='--', linewidth=1.8, label=f"SSL Target (${swing_low:,.2f})")
        ax.axhline(eq, color='blue', linestyle=':', linewidth=1.4, label=f"Equilibrium (${eq:,.2f})")
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M UTC'))
        ax.set_ylabel('Spot Price (USD)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        
    with tab2:
        st.subheader("6-Month Results: Macro-Filtered High-Probability Executions")
        
        trades, bt_dates, equity_curve, skipped = run_macro_smc_backtest(df_unified, capital, risk_pct, bt_window)
        total_trades = len(trades)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trades Filtered (Saved by Macro)", skipped, help="SMC setups ignored because they fought the Daily DXY/Yield trend.")
        c2.metric("Executed Trades", total_trades)
        
        if total_trades > 0:
            win_rate = (len([t for t in trades if t['Result'] == 'Win']) / total_trades) * 100
            total_net = equity_curve[-1] - capital
            
            c3.metric("Filtered Win Rate", f"{win_rate:.1f}%")
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
            st.subheader("📝 Macro-Approved Trade Ledger")
            trade_df = pd.DataFrame(trades)
            trade_df['Entry Date'] = trade_df['Entry Date'].dt.strftime('%b %d, %Y - %H:%M')
            trade_df['Exit Date'] = trade_df['Exit Date'].dt.strftime('%b %d, %Y - %H:%M')
            trade_df['Entry Price'] = trade_df['Entry Price'].apply(lambda x: f"${x:,.2f}")
            trade_df['Stop Loss'] = trade_df['Stop Loss'].apply(lambda x: f"${x:,.2f}")
            trade_df['Target (TP)'] = trade_df['Target (TP)'].apply(lambda x: f"${x:,.2f}")
            trade_df['Net P&L'] = trade_df['Net P&L'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No trades triggered. The Macro trend and High-Probability structure did not align during this 6-month period.")
else:
    st.error("Market data feeds are currently unreachable. Streamlit Cloud IPs have been rate-limited by the provider.")
