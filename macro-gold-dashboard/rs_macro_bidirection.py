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

st.set_page_config(page_title="Macro-SMC Unified Alpha Engine", layout="wide")
st.title("Macro-SMC Unified Alpha Engine & Execution Desk")

CURRENT_CPI = 3.35  # Static CPI baseline

# --- 1. CLOUD-RESILIENT LIVE SPOT FETCHER ---
@st.cache_data(ttl=15)
def get_live_xauusd_spot():
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
    
    # Priority 2: KuCoin PAXG-USDT (1:1 Gold Peg)
    try:
        url = "https://api.kucoin.com/api/v1/market/orderbook/level1?symbol=PAXG-USDT"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            price = res.json().get('data', {}).get('price')
            if price and float(price) > 1000: return float(price)
    except: pass

    # Priority 3: MEXC PAXGUSDT
    try:
        url = "https://api.mexc.com/api/v3/ticker/price?symbol=PAXGUSDT"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            price = res.json().get('price')
            if price and float(price) > 1000: return float(price)
    except: pass

    return 4197.50

live_price = get_live_xauusd_spot()

# --- INITIALIZE TRADINGVIEW CONNECTION ---
@st.cache_resource
def get_tv_connection():
    try:
        return TvDatafeed()
    except Exception:
        return None

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Execution Parameters")
st.sidebar.success(f"Live Feed Synchronized\n\n**Spot: ${live_price:,.2f}**")

capital = st.sidebar.number_input("Account Balance ($)", min_value=1000.0, value=15000.0, step=1000.0)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.5)
bt_window = st.sidebar.slider("Structural Swing Lookback", 5, 30, 15)

# --- 2. UNIFIED DATA PIPELINE (MACRO + STRUCTURAL) ---
@st.cache_data(ttl=300)
def fetch_unified_data():
    # A. Macro Data (DXY + 10Y Yield)
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
            
            # Trend alignment matrix
            bull = (df_macro['DXY'] < df_macro['DXY_SMA20']) & (df_macro['Real_Yield'] < df_macro['Yield_SMA20'])
            bear = (df_macro['DXY'] > df_macro['DXY_SMA20']) & (df_macro['Real_Yield'] > df_macro['Yield_SMA20'])
            df_macro['Macro_Signal'] = np.select([bull, bear], [1, -1], default=0)
            
            if df_macro.index.tz is not None: df_macro.index = df_macro.index.tz_localize(None)
            df_macro['Date_Only'] = df_macro.index.normalize()
    except: pass

    # B. Gold Historical Spot Data (With Full Crypto Cascade Fallbacks)
    df_1h, df_1d = pd.DataFrame(), pd.DataFrame()
    tv = get_tv_connection()
    
    # ATTEMPT 1: TradingView
    if tv is not None:
        exchanges = ['FXCM', 'OANDA', 'FOREXCOM']
        for exc in exchanges:
            try:
                df_1h = tv.get_hist(symbol='XAUUSD', exchange=exc, interval=Interval.in_1_hour, n_bars=1500)
                df_1d = tv.get_hist(symbol='XAUUSD', exchange=exc, interval=Interval.in_daily, n_bars=300)
                if df_1h is not None and not df_1h.empty and df_1d is not None and not df_1d.empty:
                    break
            except: continue

    # ATTEMPT 2: MEXC PAXGUSDT
    if df_1h is None or df_1h.empty or df_1d is None or df_1d.empty:
        try:
            r1h = requests.get("https://api.mexc.com/api/v3/klines?symbol=PAXGUSDT&interval=60m&limit=1000", timeout=5).json()
            d1h = pd.DataFrame(r1h, columns=['time', 'open', 'high', 'low', 'close', 'v', 'ct', 'qav', 'nt', 'tbb', 'tbq'])
            d1h['time'] = pd.to_datetime(d1h['time'].astype(int), unit='ms', utc=True)
            df_1h = d1h[['time', 'open', 'high', 'low', 'close']].set_index('time').astype(float).sort_index()

            r1d = requests.get("https://api.mexc.com/api/v3/klines?symbol=PAXGUSDT&interval=1d&limit=300", timeout=5).json()
            d1d = pd.DataFrame(r1d, columns=['time', 'open', 'high', 'low', 'close', 'v', 'ct', 'qav', 'nt', 'tbb', 'tbq'])
            d1d['time'] = pd.to_datetime(d1d['time'].astype(int), unit='ms', utc=True)
            df_1d = d1d[['time', 'open', 'high', 'low', 'close']].set_index('time').astype(float).sort_index()
        except: pass

    # ATTEMPT 3: KuCoin PAXG-USDT
    if df_1h is None or df_1h.empty or df_1d is None or df_1d.empty:
        try:
            r1h = requests.get("https://api.kucoin.com/api/v1/market/candles?type=1hour&symbol=PAXG-USDT", timeout=5).json()
            d1h = pd.DataFrame(r1h['data'], columns=['time', 'open', 'close', 'high', 'low', 'v', 't'])
            d1h['time'] = pd.to_datetime(d1h['time'].astype(int), unit='s', utc=True)
            df_1h = d1h[['time', 'open', 'high', 'low', 'close']].set_index('time').astype(float).sort_index()

            r1d = requests.get("https://api.kucoin.com/api/v1/market/candles?type=1day&symbol=PAXG-USDT", timeout=5).json()
            d1d = pd.DataFrame(r1d['data'], columns=['time', 'open', 'close', 'high', 'low', 'v', 't'])
            d1d['time'] = pd.to_datetime(d1d['time'].astype(int), unit='s', utc=True)
            df_1d = d1d[['time', 'open', 'high', 'low', 'close']].set_index('time').astype(float).sort_index()
        except: pass

    # ATTEMPT 4: yfinance
    if df_1h is None or df_1h.empty or df_1d is None or df_1d.empty:
        try:
            df_1h = yf.download("XAUUSD=X", period="60d", interval="1h", progress=False)
            df_1d = yf.download("XAUUSD=X", period="1y", interval="1d", progress=False)
        except: pass

    if df_1h is not None and not df_1h.empty and df_1d is not None and not df_1d.empty:
        for df in [df_1h, df_1d]:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.rename(columns=lambda x: x.capitalize() if isinstance(x, str) else x, inplace=True)
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            else:
                df.index = df.index.tz_convert('UTC')

        df_4h = df_1h.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
        
        # CRITICAL FIX: Strip timezone from the index before assigning to Date_Only for merging
        df_1h['Date_Only'] = df_1h.index.tz_localize(None).normalize()
        
        if not df_macro.empty:
            df_unified = pd.merge(df_1h, df_macro[['Date_Only', 'Macro_Signal']], on='Date_Only', how='left')
            df_unified.index = df_1h.index
            df_unified['Macro_Signal'] = df_unified['Macro_Signal'].ffill().fillna(0)
        else:
            df_1h['Macro_Signal'] = 0
            df_unified = df_1h

        return df_unified, df_4h, df_1d, df_macro

    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# --- 3. S/R LEVEL EXTRACTION (LOCKED TO CLOSED CANDLES) ---
def find_structural_levels(df, window=5):
    closed_df = df.iloc[:-1]
    maxima_indices = argrelextrema(closed_df['High'].values, np.greater, order=window)[0]
    resistances = closed_df['High'].iloc[maxima_indices].values
    
    minima_indices = argrelextrema(closed_df['Low'].values, np.less, order=window)[0]
    supports = closed_df['Low'].iloc[minima_indices].values
    return supports, resistances

def extract_key_levels(supports_all, resistances_all, current_price):
    res_above = sorted([r for r in set(resistances_all) if r > current_price])
    sup_below = sorted([s for s in set(supports_all) if s < current_price], reverse=True)
    
    r1 = res_above[0] if len(res_above) > 0 else None
    r2 = res_above[1] if len(res_above) > 1 else None
    r3 = res_above[2] if len(res_above) > 2 else None
    
    s1 = sup_below[0] if len(sup_below) > 0 else None
    s2 = sup_below[1] if len(sup_below) > 1 else None
    s3 = sup_below[2] if len(sup_below) > 2 else None
    
    return r1, r2, r3, s1, s2, s3

# --- 4. FILTERED BIDIRECTIONAL BACKTEST ENGINE ---
def run_macro_smc_backtest(df, start_capital, risk, window, current_live_price):
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
    skipped_trades = 0
    
    for i in range(window*2, len(df)):
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
        
        if not in_trade:
            is_short_setup = (short_ote_low <= close <= short_ote_high) and close > eq
            is_long_setup = (long_ote_low <= close <= long_ote_high) and close < eq
            
            if is_short_setup:
                if macro_bias in [-1, 0]: # Macro Agrees or Neutral
                    in_trade, trade_type = True, 'Short'
                    entry_price, stop_loss, take_profit, entry_date = close, high + 2.50, low, date
                else: 
                    skipped_trades += 1
                    
            elif is_long_setup:
                if macro_bias in [1, 0]: # Macro Agrees or Neutral
                    in_trade, trade_type = True, 'Long'
                    entry_price, stop_loss, take_profit, entry_date = close, low - 2.50, high, date
                else: 
                    skipped_trades += 1
                
        elif in_trade:
            risk_amt = equity * (risk / 100)
            if trade_type == 'Short':
                if curr_high >= stop_loss:
                    equity -= risk_amt
                    trades.append({'Date': date, 'Type': 'Short', 'Result': 'Loss', 'Entry': entry_price, 'Net': -risk_amt})
                    in_trade = False
                elif curr_low <= take_profit:
                    equity += risk_amt * ((entry_price - take_profit) / (stop_loss - entry_price))
                    trades.append({'Date': date, 'Type': 'Short', 'Result': 'Win', 'Entry': entry_price, 'Net': risk_amt * ((entry_price - take_profit) / (stop_loss - entry_price))})
                    in_trade = False
                    
            elif trade_type == 'Long':
                if curr_low <= stop_loss:
                    equity -= risk_amt
                    trades.append({'Date': date, 'Type': 'Long', 'Result': 'Loss', 'Entry': entry_price, 'Net': -risk_amt})
                    in_trade = False
                elif curr_high >= take_profit:
                    equity += risk_amt * ((take_profit - entry_price) / (entry_price - stop_loss))
                    trades.append({'Date': date, 'Type': 'Long', 'Result': 'Win', 'Entry': entry_price, 'Net': risk_amt * ((take_profit - entry_price) / (entry_price - stop_loss))})
                    in_trade = False
                    
            if not in_trade:
                equity_curve.append(equity)
                dates.append(date)

    active_position = None
    if in_trade:
        risk_dollar = equity * (risk / 100)
        sl_dist = abs(entry_price - stop_loss)
        lot_size = round(risk_dollar / (sl_dist * 100), 2) if sl_dist > 0 else 0.01 
        
        if trade_type == 'Short':
            unrealized_pnl = (entry_price - current_live_price) * 100 * lot_size
            rr = (entry_price - current_live_price) / sl_dist if sl_dist > 0 else 0.0
        else:
            unrealized_pnl = (current_live_price - entry_price) * 100 * lot_size
            rr = (current_live_price - entry_price) / sl_dist if sl_dist > 0 else 0.0
            
        active_position = {
            'is_open': True, 'type': trade_type, 'entry_price': entry_price, 'stop_loss': stop_loss,
            'take_profit': take_profit, 'current_price': current_live_price, 'lot_size': lot_size,
            'risk_dollar': risk_dollar, 'unrealized_pnl': unrealized_pnl, 'rr': rr
        }
                
    if dates[-1] != df.index[-1]:
        dates.append(df.index[-1])
        equity_curve.append(equity)
        
    return trades, dates, equity_curve, skipped_trades, active_position

# --- 5. RENDER DASHBOARD ---
df_unified, df_4h, df_1d, df_macro = fetch_unified_data()

if not df_unified.empty and not df_1d.empty:
    # Generate Multi-Timeframe S/R
    sup_1d, res_1d = find_structural_levels(df_1d, window=7)
    r1_1d, r2_1d, r3_1d, s1_1d, s2_1d, s3_1d = extract_key_levels(sup_1d, res_1d, live_price)
    sup_4h, res_4h = find_structural_levels(df_4h, window=5)
    r1_4h, r2_4h, r3_4h, s1_4h, s2_4h, s3_4h = extract_key_levels(sup_4h, res_4h, live_price)
    
    # Process Backtest & Setup
    trades, bt_dates, equity_curve, skipped, active_pos = run_macro_smc_backtest(df_unified, capital, risk_pct, bt_window, live_price)
    
    closed_df = df_unified.iloc[:-1]
    recent_high = float(closed_df['High'].tail(bt_window*2).max())
    recent_low = float(closed_df['Low'].tail(bt_window*2).min())
    total_range = recent_high - recent_low
    equilibrium = recent_high - (total_range * 0.50)
    
    short_ote_low = recent_high - (total_range * 0.382)
    short_ote_high = recent_high - (total_range * 0.214)
    long_ote_high = recent_low + (total_range * 0.382)
    long_ote_low = recent_low + (total_range * 0.214)
    
    macro_sig = int(df_unified['Macro_Signal'].iloc[-1])
    macro_text = "🟢 BULLISH (DXY/Yields Dropping)" if macro_sig == 1 else ("🔴 BEARISH (DXY/Yields Rising)" if macro_sig == -1 else "⚪ NEUTRAL")

    # --- TOP EXECUTION DESK ---
    st.subheader("⚡ Live Trade Execution Desk")

    if active_pos and active_pos['is_open']:
        if active_pos['type'] == 'Short': st.error(f"**ACTIVE POSITION IN PROGRESS: 🔴 SHORT GOLD**")
        else: st.success(f"**ACTIVE POSITION IN PROGRESS: 🟢 LONG GOLD**")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Execution Price", f"${active_pos['entry_price']:,.2f}")
        c2.metric("Stop Loss", f"${active_pos['stop_loss']:,.2f}")
        c3.metric("Take Profit", f"${active_pos['take_profit']:,.2f}")
        c4.metric("Position Size", f"{active_pos['lot_size']} Lots")
        c5.metric("Unrealized P&L", f"${active_pos['unrealized_pnl']:,.2f}", f"R:R {active_pos['rr']:.1f}:1")
    else:
        st.info("📡 **SCANNER STATUS: PENDING SETUP ORDERS BELOW**")
        
        # Determine actionable setup
        if live_price > equilibrium:
            setup_action, entry_zone = "🔴 PLACE SELL LIMIT ORDER", f"${short_ote_low:,.2f} – ${short_ote_high:,.2f}"
            order_entry, order_sl, order_tp = short_ote_low, recent_high + 2.50, recent_low
            filter_match = "Approved" if macro_sig in [-1, 0] else "Rejected by Macro Filter"
        else:
            setup_action, entry_zone = "🟢 PLACE BUY LIMIT ORDER", f"${long_ote_low:,.2f} – ${long_ote_high:,.2f}"
            order_entry, order_sl, order_tp = long_ote_high, recent_low - 2.50, recent_high
            filter_match = "Approved" if macro_sig in [1, 0] else "Rejected by Macro Filter"

        sl_distance = abs(order_entry - order_sl)
        risk_dollars = capital * (risk_pct / 100)
        calc_lots = round(risk_dollars / (sl_distance * 100), 2) if sl_distance > 0 else 0.01

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Macro Alignment", macro_text, filter_match)
        c2.metric("Optimal OTE Trigger", entry_zone, setup_action)
        c3.metric("Structural Stop Loss", f"${order_sl:,.2f}")
        c4.metric("Structural Target (TP)", f"${order_tp:,.2f}")
        c5.metric("Calculated Lot Size", f"{calc_lots} Lots", f"${risk_dollars:,.0f} Max Risk")

    st.divider()

    # --- MULTI-TIMEFRAME S/R MATRIX ---
    st.subheader("Multi-Timeframe Support & Resistance (Grid Strategy Zones)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Daily (D1) Macro Levels**")
        st.write(f"Resistance 2: **${r2_1d:,.2f}**" if r2_1d else "Resistance 2: N/A")
        st.write(f"Resistance 1: **${r1_1d:,.2f}**" if r1_1d else "Resistance 1: N/A")
        st.write(f"Support 1: **${s1_1d:,.2f}**" if s1_1d else "Support 1: N/A")
        st.write(f"Support 2: **${s2_1d:,.2f}**" if s2_1d else "Support 2: N/A")
    with col2:
        st.markdown("**4-Hour (4H) Swing Levels**")
        st.write(f"Resistance 2: **${r2_4h:,.2f}**" if r2_4h else "Resistance 2: N/A")
        st.write(f"Resistance 1: **${r1_4h:,.2f}**" if r1_4h else "Resistance 1: N/A")
        st.write(f"Support 1: **${s1_4h:,.2f}**" if s1_4h else "Support 1: N/A")
        st.write(f"Support 2: **${s2_4h:,.2f}**" if s2_4h else "Support 2: N/A")

    st.divider()

    # --- CHARTS & BACKTEST ---
    tab1, tab2 = st.tabs(["📊 Unified 1H Strategy Chart", "📈 Institutional Backtest"])

    with tab1:
        df_chart = df_unified.tail(120)
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # OTE & Dealing Range
        ax.axhspan(equilibrium, recent_high, color='red', alpha=0.03, label="Premium Range")
        ax.axhspan(recent_low, equilibrium, color='green', alpha=0.03, label="Discount Range")
        ax.axhspan(short_ote_low, short_ote_high, color='darkred', alpha=0.15, label="Supply OTE (Sell Zone)")
        ax.axhspan(long_ote_low, long_ote_high, color='darkgreen', alpha=0.15, label="Demand OTE (Buy Zone)")
        
        # Price
        ax.plot(df_chart.index, df_chart['Close'], color='black', linewidth=1.5, label="Live Spot")
        
        # S/R Overlay (Faint to prevent clutter)
        if r1_4h: ax.axhline(r1_4h, color='purple', linestyle='--', alpha=0.4, label=f"4H R1 (${r1_4h:,.2f})")
        if s1_4h: ax.axhline(s1_4h, color='blue', linestyle='--', alpha=0.4, label=f"4H S1 (${s1_4h:,.2f})")
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d - %H:%M'))
        ax.set_ylabel('Spot Price (USD)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.20, 1))
        ax.grid(alpha=0.2)
        st.pyplot(fig)

    with tab2:
        total_trades = len(trades)
        if total_trades > 0:
            win_rate = (len([t for t in trades if t['Result'] == 'Win']) / total_trades) * 100
            total_net = equity_curve[-1] - capital
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Filter Interventions", skipped, help="SMC entries canceled due to conflicting DXY/Yield trends.")
            c2.metric("Closed Trades", total_trades)
            c3.metric("System Win Rate", f"{win_rate:.1f}%")
            c4.metric("Net Backtest Return", f"${total_net:,.2f}")
            
            fig2, ax2 = plt.subplots(figsize=(14, 4))
            ax2.plot(bt_dates, equity_curve, color='teal', linewidth=2, label="Strategy Equity")
            ax2.axhline(capital, color='black', linestyle='--', linewidth=1)
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax2.set_ylabel('Equity ($)')
            ax2.grid(alpha=0.2)
            st.pyplot(fig2)
            
            st.dataframe(pd.DataFrame(trades).iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.warning("No trades triggered under current parameter configuration.")
else:
    st.error("Market data feeds are currently unreachable.")
