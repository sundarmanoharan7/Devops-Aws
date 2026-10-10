import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from datetime import datetime

st.set_page_config(page_title="Apex Drawdown Zero V9.20 Engine", layout="wide")
st.title("🛡️ Apex Drawdown Zero V9.20 — Multi-Asset H1 Strategy")

# --- PRESET CONFIGURATIONS ---
SYMBOL_PRESETS = {
    "XAUUSD (Gold)": {"ticker": "GC=F", "rr": 2.0, "first_hour": 8, "last_hour": 20, "contract_size": 100},
    "EURUSD":        {"ticker": "EURUSD=X", "rr": 1.5, "first_hour": 8, "last_hour": 20, "contract_size": 100000},
    "EURJPY":        {"ticker": "EURJPY=X", "rr": 1.5, "first_hour": 12, "last_hour": 20, "contract_size": 100000}
}

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("Strategy Settings (V9.20 Defaults)")
preset_choice = st.sidebar.selectbox("Symbol Preset", list(SYMBOL_PRESETS.keys()))
active_preset = SYMBOL_PRESETS[preset_choice]

capital = st.sidebar.number_input("Starting Capital ($)", min_value=100.0, value=10000.0, step=1000.0)
risk_pct = st.sidebar.slider("Risk Per Trade (%)", 0.25, 3.0, 1.0, 0.25)
max_lots = st.sidebar.number_input("Hard Max Lot Cap (0 = Off)", min_value=0.0, value=0.0, step=0.1)

with st.sidebar.expander("Advanced Range & Volatility Logic", expanded=False):
    box_candles = st.number_input("Box Candles (H1)", value=5, min_value=2, max_value=10)
    min_box_atr = st.number_input("Min Box Size (x ATR)", value=1.50, step=0.1)
    max_box_atr = st.number_input("Max Box Size (x ATR)", value=4.00, step=0.1)
    break_buffer_atr = st.number_input("Breakout Buffer (x ATR)", value=0.20, step=0.05)
    confirm_body_pct = st.slider("Min Body Size for Displacement (%)", 40, 90, 60)
    sweep_close_pct = st.slider("Sweep Re-entry Inside (%)", 10, 50, 20)
    atr_stop_mult = st.number_input("Stop Loss Multiplier (x ATR)", value=2.50, step=0.1)
    be_trigger_rr = st.slider("Break-Even Trigger (% of TP)", 0.2, 0.9, 0.60)
    ema_bias_period = st.number_input("Macro Bias EMA Period", value=50, step=5)

# --- DATA INGESTION ---
@st.cache_data(ttl=300)
def fetch_h1_data(ticker):
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    
    df = yf.download(ticker, period="6mo", interval="1h", progress=False, session=session)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    return df

raw_data = fetch_h1_data(active_preset["ticker"])

if raw_data.empty:
    st.error(f"Failed to fetch market data for {preset_choice}. Check connection or symbol ticker.")
    st.stop()

# --- TECHNICAL CALCULATIONS ---
def compute_indicators(df):
    d = df.copy()
    # ATR 14
    hl = d['High'] - d['Low']
    hpc = (d['High'] - d['Close'].shift(1)).abs()
    lpc = (d['Low'] - d['Close'].shift(1)).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    d['ATR'] = tr.rolling(14).mean().bfill()
    
    # 50 EMA Macro Trend
    d['EMA_50'] = d['Close'].ewm(span=int(ema_bias_period), adjust=False).mean()
    
    # Candle geometry
    d['CandleRange'] = d['High'] - d['Low']
    d['Body'] = (d['Close'] - d['Open']).abs()
    d['BodyPct'] = np.where(d['CandleRange'] > 0, (d['Body'] / d['CandleRange']) * 100.0, 0.0)
    
    # Broker date groupings (assumes UTC; adjust offset if aligning to GMT+2/3)
    d['Date'] = d.index.date
    d['Hour'] = d.index.hour
    return d

data = compute_indicators(raw_data)

# --- BACKTESTING LOGIC ENGINE ---
def run_apex_backtest(df, preset):
    trades = []
    equity = float(capital)
    dates = [df.index[0]]
    equity_curve = [equity]
    
    rr_ratio = preset["rr"]
    first_hour = preset["first_hour"]
    last_hour = preset["last_hour"]
    
    # Process market day by day
    for current_date, day_df in df.groupby('Date'):
        if len(day_df) < box_candles:
            continue
        
        # 1. Identify First 5 H1 Bars of the Day
        box_bars = day_df.iloc[:box_candles]
        box_high = box_bars['High'].max()
        box_low = box_bars['Low'].min()
        box_size = box_high - box_low
        atr_at_box = box_bars['ATR'].iloc[-1]
        
        if atr_at_box <= 0:
            continue
            
        box_ratio = box_size / atr_at_box
        
        # Range Filter: Skip day if range is too narrow or too wide
        if box_ratio < min_box_atr or (max_box_atr > 0 and box_ratio > max_box_atr):
            continue
        
        # Trading Window: Bars after opening box within session boundaries
        trading_bars = day_df.iloc[box_candles:]
        trade_taken_today = False
        
        for bar_idx in range(len(trading_bars)):
            if trade_taken_today:
                break
                
            c = trading_bars.iloc[bar_idx]
            
            # Check timing window
            if not (first_hour <= c['Hour'] <= last_hour):
                continue
                
            atr = c['ATR']
            close = c['Close']
            high = c['High']
            low = c['Low']
            body_pct = c['BodyPct']
            ema = c['EMA_50']
            
            signal = None
            
            # --- BUY CONDITIONS ---
            if close > ema:
                # Setup A: Displacement Breakout
                if (close > box_high + (break_buffer_atr * atr)) and (body_pct >= confirm_body_pct):
                    signal = "BUY"
                # Setup B: Liquidity Sweep of Box Low with Re-entry
                elif (low < box_low) and (close >= box_low + (sweep_close_pct / 100.0 * (box_high - box_low))):
                    signal = "BUY"
            
            # --- SELL CONDITIONS ---
            elif close < ema:
                # Setup A: Displacement Breakout
                if (close < box_low - (break_buffer_atr * atr)) and (body_pct >= confirm_body_pct):
                    signal = "SELL"
                # Setup B: Liquidity Sweep of Box High with Re-entry
                elif (high > box_high) and (close <= box_high - (sweep_close_pct / 100.0 * (box_high - box_low))):
                    signal = "SELL"
            
            if signal is not None:
                entry_price = close
                stop_dist = atr_stop_mult * atr
                tp_dist = stop_dist * rr_ratio
                
                if signal == "BUY":
                    sl_price = entry_price - stop_dist
                    tp_price = entry_price + tp_dist
                else:
                    sl_price = entry_price + stop_dist
                    tp_price = entry_price - tp_dist
                
                # Position Sizing
                risk_cash = equity * (risk_pct / 100.0)
                lot_size = risk_cash / (stop_dist * preset["contract_size"]) if stop_dist > 0 else 0.01
                lot_size = max(0.01, round(lot_size, 2))
                if max_lots > 0:
                    lot_size = min(lot_size, max_lots)
                
                trade_taken_today = True
                
                # Simulate forward path through rest of dataset
                remaining_bars = df.loc[c.name:].iloc[1:]
                be_activated = False
                exit_price = None
                exit_date = None
                trade_result = None
                
                for _, f_bar in remaining_bars.iterrows():
                    f_high = f_bar['High']
                    f_low = f_bar['Low']
                    
                    if signal == "BUY":
                        # Break-Even check
                        if not be_activated and f_high >= (entry_price + (tp_dist * be_trigger_rr)):
                            sl_price = entry_price
                            be_activated = True
                        
                        # Stop Loss hit
                        if f_low <= sl_price:
                            exit_price = sl_price
                            exit_date = f_bar.name
                            trade_result = "BE Flat" if (be_activated and sl_price == entry_price) else "Loss"
                            break
                        # Target hit
                        elif f_high >= tp_price:
                            exit_price = tp_price
                            exit_date = f_bar.name
                            trade_result = "Win"
                            break
                    else:  # SELL
                        # Break-Even check
                        if not be_activated and f_low <= (entry_price - (tp_dist * be_trigger_rr)):
                            sl_price = entry_price
                            be_activated = True
                        
                        # Stop Loss hit
                        if f_high >= sl_price:
                            exit_price = sl_price
                            exit_date = f_bar.name
                            trade_result = "BE Flat" if (be_activated and sl_price == entry_price) else "Loss"
                            break
                        # Target hit
                        elif f_low <= tp_price:
                            exit_price = tp_price
                            exit_date = f_bar.name
                            trade_result = "Win"
                            break
                
                # If trade still open at data boundary
                if exit_price is None and not remaining_bars.empty:
                    exit_price = remaining_bars.iloc[-1]['Close']
                    exit_date = remaining_bars.index[-1]
                    trade_result = "Open Exit"
                
                # PnL accounting
                if trade_result == "Win":
                    net_dollar = risk_cash * rr_ratio
                elif trade_result == "BE Flat":
                    net_dollar = 0.0
                elif trade_result == "Loss":
                    net_dollar = -risk_cash
                else:
                    price_diff = (exit_price - entry_price) if signal == "BUY" else (entry_price - exit_price)
                    net_dollar = (price_diff / stop_dist) * risk_cash
                
                equity += net_dollar
                trades.append({
                    "Entry Time": c.name,
                    "Exit Time": exit_date,
                    "Type": signal,
                    "Entry": entry_price,
                    "Stop Loss": sl_price,
                    "Take Profit": tp_price,
                    "Lots": lot_size,
                    "Result": trade_result,
                    "Net ($)": net_dollar,
                    "Balance": equity
                })
                dates.append(exit_date if exit_date else c.name)
                equity_curve.append(equity)
                break
                
    return pd.DataFrame(trades), dates, equity_curve

# --- EXECUTE ENGINE ---
trades_df, dates_list, equity_curve = run_apex_backtest(data, active_preset)

# --- DASHBOARD METRICS ---
col1, col2, col3, col4 = st.columns(4)
total_trades = len(trades_df)

if total_trades > 0:
    wins = len(trades_df[trades_df['Result'] == 'Win'])
    losses = len(trades_df[trades_df['Result'] == 'Loss'])
    be_trades = len(trades_df[trades_df['Result'] == 'BE Flat'])
    win_rate = (wins / total_trades) * 100.0
    net_pnl = equity_curve[-1] - capital
    return_pct = (net_pnl / capital) * 100.0
    
    col1.metric("Total Trades Taken", total_trades)
    col2.metric("Win Rate (Excl. BE)", f"{win_rate:.1f}%", f"{wins}W / {losses}L / {be_trades}BE")
    col3.metric("Net PnL ($)", f"${net_pnl:,.2f}", f"{return_pct:.2f}%")
    col4.metric("Ending Balance", f"${equity_curve[-1]:,.2f}")
else:
    col1.metric("Total Trades Taken", 0)
    col2.metric("Win Rate", "0.0%")
    col3.metric("Net PnL ($)", "$0.00")
    col4.metric("Ending Balance", f"${capital:,.2f}")

st.divider()

# --- EQUITY CURVE & PERFORMANCE VISUALIZATION ---
st.subheader("📈 Balance Growth & Drawdown Curve")
if total_trades > 0:
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(dates_list, equity_curve, color="#008080", linewidth=2.0, label="Account Equity")
    ax.axhline(capital, color="black", linestyle="--", linewidth=1.0, alpha=0.7, label="Initial Capital")
    ax.set_ylabel("Account Balance ($)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")
    st.pyplot(fig)

    st.subheader("📋 Completed Trade Log")
    styled_df = trades_df.copy()
    styled_df['Entry Time'] = pd.to_datetime(styled_df['Entry Time']).dt.strftime('%Y-%m-%d %H:%M')
    styled_df['Exit Time'] = pd.to_datetime(styled_df['Exit Time']).dt.strftime('%Y-%m-%d %H:%M')
    for col in ['Entry', 'Stop Loss', 'Take Profit', 'Net ($)', 'Balance']:
        styled_df[col] = styled_df[col].map(lambda x: f"${x:,.2f}")
    st.dataframe(styled_df.iloc[::-1], use_container_width=True, hide_index=True)
else:
    st.warning("No qualifying setups satisfied all criteria (Session Box + Trend + Body % + Range Filter) during this period.")
