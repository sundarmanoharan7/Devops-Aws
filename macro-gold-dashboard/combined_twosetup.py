import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests

st.set_page_config(page_title="Macro-SMC Live Execution Engine", layout="wide")
st.title("Macro-SMC Live Execution & Position Monitor")

CURRENT_CPI = 3.35  # Static CPI baseline

# --- 1. REAL-TIME SPOT FETCHER ---
@st.cache_data(ttl=15)
def get_live_xauusd_spot():
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Priority 1: TradingView Scanner API (Direct CFD Spot)
    try:
        url = "https://scanner.tradingview.com/cfd/scan"
        payload = {"symbols": {"tickers": ["FXCM:XAUUSD", "OANDA:XAUUSD"]}, "columns": ["close"]}
        res = requests.post(url, json=payload, headers=headers, timeout=3)
        if res.status_code == 200:
            price = res.json().get('data', [{}])[0].get('d', [0])[0]
            if price > 1000: return float(price)
    except Exception: pass

    # Priority 2: Direct Yahoo v8 Endpoint
    try:
        url = "https://query1.finance.yahoo.com/v8/finance/chart/XAUUSD=X?interval=1m&range=1d"
        res = requests.get(url, headers=headers, timeout=3)
        if res.status_code == 200:
            price = res.json()['chart']['result'][0]['meta'].get('regularMarketPrice')
            if price and float(price) > 1000: return float(price)
    except Exception: pass

    return 4195.00

market_spot = get_live_xauusd_spot()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("Live Feed & Risk Parameters")
manual_override = st.sidebar.checkbox("Override Spot Price (Manual)", value=False)
if manual_override:
    live_price = st.sidebar.number_input("Broker Price (USD)", min_value=1000.0, max_value=10000.0, value=float(round(market_spot, 2)), step=0.10)
else:
    live_price = market_spot
    st.sidebar.success(f"Live Price: **${live_price:,.2f}**")

capital = st.sidebar.number_input("Account Balance ($)", min_value=1000.0, value=15000.0, step=1000.0)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.5, 5.0, 2.0, 0.5)
bt_window = st.sidebar.slider("SMC Structural Lookback (1H)", 5, 30, 15)

# --- 2. UNIFIED DATA PIPELINE (MACRO + SMC) ---
@st.cache_data(ttl=300)
def fetch_unified_data(anchor_spot: float):
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
            
            bull = (df_macro['DXY'] < df_macro['DXY_SMA20']) & (df_macro['Real_Yield'] < df_macro['Yield_SMA20'])
            bear = (df_macro['DXY'] > df_macro['DXY_SMA20']) & (df_macro['Real_Yield'] > df_macro['Yield_SMA20'])
            df_macro['Macro_Signal'] = np.select([bull, bear], [1, -1], default=0)
            
            if df_macro.index.tz is not None: df_macro.index = df_macro.index.tz_localize(None)
            df_macro['Date_Only'] = df_macro.index.normalize()
    except Exception: pass

    # B. Hourly Chart Data (Aligned to live spot)
    df_smc = pd.DataFrame()
    try:
        df_smc = yf.download("XAUUSD=X", period="6mo", interval="1h", progress=False)
    except Exception: pass

    if df_smc.empty:
        try:
            df_smc = yf.download("GC=F", period="6mo", interval="1h", progress=False)
        except Exception: pass

    if not df_smc.empty:
        if isinstance(df_smc.columns, pd.MultiIndex):
            df_smc.columns = df_smc.columns.get_level_values(0)
        df_smc.rename(columns=lambda x: x.capitalize() if isinstance(x, str) else x, inplace=True)
        if df_smc.index.tz is not None: df_smc.index = df_smc.index.tz_localize(None)
        df_smc.dropna(inplace=True)

        active_hist_bar = float(df_smc['Close'].iloc[-1])
        spread = active_hist_bar - anchor_spot
        for col in ['Open', 'High', 'Low', 'Close']:
            df_smc[col] = df_smc[col] - spread

        df_smc['Date_Only'] = df_smc.index.normalize()
        if not df_macro.empty:
            df_unified = pd.merge(df_smc, df_macro[['Date_Only', 'Macro_Signal']], on='Date_Only', how='left')
            df_unified.index = df_smc.index
            df_unified['Macro_Signal'] = df_unified['Macro_Signal'].ffill().fillna(0)
        else:
            df_smc['Macro_Signal'] = 0
            df_unified = df_smc

        return df_unified, df_macro

    return pd.DataFrame(), pd.DataFrame()

# --- 3. FILTERED BACKTEST & ACTIVE POSITION TRACKER ---
def run_macro_smc_backtest(df, start_capital, risk, window, current_live_price):
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
        
        short_ote_low = high - (total_range * 0.382)
        short_ote_high = high - (total_range * 0.214)
        long_ote_high = low + (total_range * 0.382)
        long_ote_low = low + (total_range * 0.214)
        
        if not in_trade:
            is_short_setup = (short_ote_low <= close <= short_ote_high) and close > eq
            is_long_setup = (long_ote_low <= close <= long_ote_high) and close < eq
            
            if is_short_setup:
                if macro_bias in [-1, 0]:
                    in_trade, trade_type = True, 'Short'
                    entry_price, stop_loss, take_profit, entry_date = close, high + 2.50, low, date
                else: 
                    skipped_trades += 1
                    
            elif is_long_setup:
                if macro_bias in [1, 0]:
                    in_trade, trade_type = True, 'Long'
                    entry_price, stop_loss, take_profit, entry_date = close, low - 2.50, high, date
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
                    r_multiple = (entry_price - take_profit) / (stop_loss - entry_price)
                    win_amt = risk_amt * r_multiple
                    equity += win_amt
                    trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Type': 'Short', 'Result': 'Win', 'Entry Price': entry_price, 'Stop Loss': stop_loss, 'Target (TP)': take_profit, 'Net P&L': win_amt})
                    in_trade = False
                    
            elif trade_type == 'Long':
                if curr_low <= stop_loss:
                    equity -= risk_amt
                    trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Type': 'Long', 'Result': 'Loss', 'Entry Price': entry_price, 'Stop Loss': stop_loss, 'Target (TP)': take_profit, 'Net P&L': -risk_amt})
                    in_trade = False
                elif curr_high >= take_profit:
                    r_multiple = (take_profit - entry_price) / (entry_price - stop_loss)
                    win_amt = risk_amt * r_multiple
                    equity += win_amt
                    trades.append({'Entry Date': entry_date, 'Exit Date': date, 'Type': 'Long', 'Result': 'Win', 'Entry Price': entry_price, 'Stop Loss': stop_loss, 'Target (TP)': take_profit, 'Net P&L': win_amt})
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
            'is_open': True,
            'type': trade_type,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'entry_date': entry_date,
            'current_price': current_live_price,
            'lot_size': lot_size,
            'risk_dollar': risk_dollar,
            'unrealized_pnl': unrealized_pnl,
            'rr': rr
        }
                
    if dates[-1] != df.index[-1]:
        dates.append(df.index[-1])
        equity_curve.append(equity)
        
    return trades, dates, equity_curve, skipped_trades, active_position

# --- 4. RENDER DASHBOARD ---
df_unified, df_macro = fetch_unified_data(live_price)

if not df_unified.empty:
    trades, bt_dates, equity_curve, skipped, active_pos = run_macro_smc_backtest(
        df_unified, capital, risk_pct, bt_window, live_price
    )
    
    recent_high = float(df_unified['High'].tail(bt_window*2).max())
    recent_low = float(df_unified['Low'].tail(bt_window*2).min())
    total_range = recent_high - recent_low
    equilibrium = recent_high - (total_range * 0.50)
    
    short_ote_low = recent_high - (total_range * 0.382)
    short_ote_high = recent_high - (total_range * 0.214)
    long_ote_high = recent_low + (total_range * 0.382)
    long_ote_low = recent_low + (total_range * 0.214)
    
    latest_macro_signal = int(df_unified['Macro_Signal'].iloc[-1])
    if latest_macro_signal == 1:
        macro_text = "🟢 BULLISH (Favoring Longs)"
    elif latest_macro_signal == -1:
        macro_text = "🔴 BEARISH (Favoring Shorts)"
    else:
        macro_text = "⚪ NEUTRAL (Cash / Scalp)"

    # --- TOP EXECUTION SECTION (PINNED ABOVE TABS) ---
    st.subheader("⚡ Live Trade Execution Desk")

    if active_pos and active_pos['is_open']:
        pos_badge = "🟢 LONG" if active_pos['type'] == 'Long' else "🔴 SHORT"
        
        # FIX: Explicit if/else statement to prevent Streamlit document dump
        if active_pos['type'] == 'Short':
            st.error(f"**ACTIVE POSITION IN PROGRESS: {pos_badge} GOLD**")
        else:
            st.success(f"**ACTIVE POSITION IN PROGRESS: {pos_badge} GOLD**")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Execution Price", f"${active_pos['entry_price']:,.2f}", f"Live: ${active_pos['current_price']:,.2f}")
        c2.metric("Hard Stop Loss", f"${active_pos['stop_loss']:,.2f}", f"-${abs(active_pos['entry_price'] - active_pos['stop_loss']):.2f} pts")
        c3.metric("Take Profit (Target)", f"${active_pos['take_profit']:,.2f}", f"+${abs(active_pos['take_profit'] - active_pos['entry_price']):.2f} pts")
        c4.metric("Position Size", f"{active_pos['lot_size']} Lots", f"Risk: ${active_pos['risk_dollar']:,.0f}")
        c5.metric("Unrealized P&L", f"${active_pos['unrealized_pnl']:,.2f}", f"Current R:R {active_pos['rr']:.1f}:1")
        
        st.caption(f"Entry Timestamp: {active_pos['entry_date'].strftime('%b %d, %Y - %H:%M UTC')} | Manage trade in MT5 according to structural SL/TP.")
    else:
        st.info("📡 **SCANNER STATUS: NO POSITION CURRENTLY OPEN — PENDING SETUP ORDERS BELOW**")
        
        if live_price > equilibrium:
            valuation = "PREMIUM (Sell Setup Forming)"
            setup_action = "🔴 PLACE SELL LIMIT ORDER"
            entry_zone = f"${short_ote_low:,.2f} –${short_ote_high:,.2f}"
            order_entry = short_ote_low
            order_sl = recent_high + 2.50
            order_tp = recent_low
            order_type = "Short"
        else:
            valuation = "DISCOUNT (Buy Setup Forming)"
            setup_action = "🟢 PLACE BUY LIMIT ORDER"
            entry_zone = f"${long_ote_low:,.2f} –${long_ote_high:,.2f}"
            order_entry = long_ote_high
            order_sl = recent_low - 2.50
            order_tp = recent_high
            order_type = "Long"

        sl_distance = abs(order_entry - order_sl)
        risk_dollars = capital * (risk_pct / 100)
        calc_lots = round(risk_dollars / (sl_distance * 100), 2) if sl_distance > 0 else 0.01

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Active Spot", f"${live_price:,.2f}", valuation)
        col2.metric("Optimal Entry Trigger", entry_zone, setup_action)
        col3.metric("Structural Stop Loss", f"${order_sl:,.2f}", f"Risk ${sl_distance:.2f} pts")
        col4.metric("Structural Target (TP)", f"${order_tp:,.2f}", "Range Target")
        col5.metric("Calculated Lot Size", f"{calc_lots} Lots", f"${risk_dollars:,.0f} Max Risk")

        with st.expander("📋 Click for Exact Broker / MT5 Order Parameters", expanded=True):
            st.markdown(f"""
            - **Market Bias**: {macro_text}
            - **Recommended Order**: `{setup_action}`
            - **Limit Entry Price**: **`${order_entry:,.2f}`** *(Or execute at market if inside `{entry_zone}`)*
            - **Stop Loss**: **`${order_sl:,.2f}`** *(Protects against structural invalidation)*
            - **Take Profit**: **`${order_tp:,.2f}`** *(Full dealing range target)*
            - **Contract Size**: **`{calc_lots} Standard Lots`** *(Enforces exact {risk_pct}% account risk)*
            """)

    st.divider()

    # --- 5. CHARTS & HISTORICAL LEDGER (MOVED TO TABS) ---
    tab1, tab2 = st.tabs(["📊 1-Hour SMC Structure", "📈 6-Month Macro Backtest"])

    with tab1:
        df_chart = df_unified.tail(120)
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(df_chart.index, df_chart['Close'], color='black', linewidth=1.2, label="XAUUSD Spot")
        ax.axhspan(equilibrium, recent_high, color='red', alpha=0.06, label="Premium Zone")
        ax.axhspan(recent_low, equilibrium, color='green', alpha=0.06, label="Discount Zone")
        
        ax.axhspan(short_ote_low, short_ote_high, color='darkred', alpha=0.25, label="Supply OTE (Sell Zone)")
        ax.axhspan(long_ote_low, long_ote_high, color='darkgreen', alpha=0.25, label="Demand OTE (Buy Zone)")
        
        ax.axhline(recent_high, color='red', linestyle='--', linewidth=1.5, label=f"Swing High (${recent_high:,.2f})")
        ax.axhline(recent_low, color='green', linestyle='--', linewidth=1.5, label=f"Swing Low (${recent_low:,.2f})")
        ax.axhline(equilibrium, color='blue', linestyle=':', linewidth=1.2, label=f"Equilibrium (${equilibrium:,.2f})")
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d - %H:%M'))
        ax.set_ylabel('Spot Price (USD)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1))
        ax.grid(alpha=0.25)
        st.pyplot(fig)

    with tab2:
        total_trades = len(trades)
        if total_trades > 0:
            win_count = len([t for t in trades if t['Result'] == 'Win'])
            win_rate = (win_count / total_trades) * 100
            total_net = equity_curve[-1] - capital
            
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Filter Interventions", skipped, help="Trades blocked by Macro trend filter")
            m2.metric("Closed Trades", total_trades)
            m3.metric("System Win Rate", f"{win_rate:.1f}%")
            m4.metric("Net Backtest Return", f"${total_net:,.2f}", f"${equity_curve[-1]:,.2f}")
            
            fig2, ax2 = plt.subplots(figsize=(14, 4))
            ax2.plot(bt_dates, equity_curve, color='teal', linewidth=2, label="Strategy Equity")
            ax2.axhline(capital, color='black', linestyle='--', linewidth=1, label="Base Capital")
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax2.set_ylabel('Equity ($)')
            ax2.legend()
            ax2.grid(alpha=0.25)
            st.pyplot(fig2)

            st.subheader("📝 Macro-Approved Closed Trade Ledger")
            tdf = pd.DataFrame(trades)
            tdf['Entry Date'] = tdf['Entry Date'].dt.strftime('%b %d, %Y - %H:%M')
            tdf['Exit Date'] = tdf['Exit Date'].dt.strftime('%b %d, %Y - %H:%M')
            tdf['Entry Price'] = tdf['Entry Price'].apply(lambda x: f"${x:,.2f}")
            tdf['Stop Loss'] = tdf['Stop Loss'].apply(lambda x: f"${x:,.2f}")
            tdf['Target (TP)'] = tdf['Target (TP)'].apply(lambda x: f"${x:,.2f}")
            tdf['Net P&L'] = tdf['Net P&L'].apply(lambda x: f"${x:,.2f}")
            st.dataframe(tdf.iloc[::-1], use_container_width=True, hide_index=True)
        else:
            st.warning("No closed trades logged under current lookback settings.")
else:
    st.error("Market data feeds are currently unreachable.")
