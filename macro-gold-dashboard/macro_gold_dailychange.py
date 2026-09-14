import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- PAGE CONFIG ---
st.set_page_config(page_title="Macro Gold Strategy Engine", layout="wide")
st.title("Macro-Driven Gold Strategy (DXY & Real Yields)")

# Static CPI from fundamental analysis
CURRENT_CPI = 3.35 

# --- DATA PIPELINE (ROBUST FIX) ---
@st.cache_data(ttl=3600)
def fetch_macro_data():
    """Fetches 6 months of daily data using isolated downloads to prevent API multi-ticker crashes."""
    target_tickers = {"GC=F": "Gold", "DX-Y.NYB": "DXY", "^TNX": "Nominal_10Y"}
    series_list = []
    
    for ticker, col_name in target_tickers.items():
        try:
            # Download individually to bypass yfinance MultiIndex formatting issues
            data = yf.download(ticker, period="6mo", interval="1d", progress=False)
            
            if not data.empty:
                # Clean up columns just in case yfinance still returns a MultiIndex for single tickers
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                
                close_series = data['Close']
                close_series.name = col_name
                series_list.append(close_series)
        except Exception:
            pass
            
    # If any of the 3 required feeds failed, abort to prevent bad math
    if len(series_list) < 3:
        return pd.DataFrame()
        
    # Merge all individual series into one DataFrame based on the date index
    df = pd.concat(series_list, axis=1)
    
    # CRITICAL FIX: Forward-fill data first. Bond markets, Futures, and Forex have different holiday schedules.
    df.ffill(inplace=True) 
    df.dropna(inplace=True)
    
    # Calculate Real 10-Year Yield
    df['Real_Yield'] = df['Nominal_10Y'] - CURRENT_CPI
    
    return df

# --- STRATEGY ENGINE ---
def generate_signals(df):
    df = df.copy()
    
    # 20-Day Macro Trend Filters
    df['DXY_SMA20'] = df['DXY'].rolling(window=20).mean()
    df['Real_Yield_SMA20'] = df['Real_Yield'].rolling(window=20).mean()
    
    df.dropna(inplace=True)
    
    # Strategy Conditions
    bull_market = (df['DXY'] < df['DXY_SMA20']) & (df['Real_Yield'] < df['Real_Yield_SMA20'])
    bear_market = (df['DXY'] > df['DXY_SMA20']) & (df['Real_Yield'] > df['Real_Yield_SMA20'])
    
    # 1 = Long, -1 = Short, 0 = Cash
    df['Position'] = np.select([bull_market, bear_market], [1, -1], default=0)
    
    # Calculate Daily Returns
    df['Gold_Return'] = df['Gold'].pct_change()
    
    # Shift position by 1 day to execute on the next day's open/close (prevents look-ahead bias)
    df['Strategy_Return'] = df['Position'].shift(1) * df['Gold_Return']
    
    # Equity Curve Calculation (Base $10,000)
    df['Buy_Hold_Equity'] = (1 + df['Gold_Return'].fillna(0)).cumprod() * 10000
    df['Strategy_Equity'] = (1 + df['Strategy_Return'].fillna(0)).cumprod() * 10000
    
    return df

# --- DASHBOARD RENDER ---
df_raw = fetch_macro_data()

if not df_raw.empty:
    df_strategy = generate_signals(df_raw)
    
    latest = df_strategy.iloc[-1]
    
    st.subheader("Current Macro Environment & Signal")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Live Gold Price", f"${latest['Gold']:,.2f}")
    col2.metric("US Dollar Index (DXY)", f"{latest['DXY']:.2f}")
    col3.metric("Real 10Y Yield", f"{latest['Real_Yield']:.2f}%")
    
    # Determine text status for the active signal
    if latest['Position'] == 1:
        signal_text = "🟢 LONG GOLD"
    elif latest['Position'] == -1:
        signal_text = "🔴 SHORT GOLD"
    else:
        signal_text = "⚪ NEUTRAL (CASH)"
        
    col4.metric("Active Strategy Signal", signal_text)
    
    st.divider()
    
    # --- VISUALIZATION ---
    st.subheader("6-Month Backtest: Macro Strategy vs. Buy & Hold")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_strategy.index, df_strategy['Buy_Hold_Equity'], color='gray', linestyle='--', label='Buy & Hold Gold')
    ax.plot(df_strategy.index, df_strategy['Strategy_Equity'], color='gold', linewidth=2.5, label='Macro Trend Strategy')
    
    ax.set_ylabel('Portfolio Equity ($)')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %Y'))
    ax.legend(loc='upper left')
    ax.grid(alpha=0.2)
    
    st.pyplot(fig)
    
    st.divider()
    
    # --- DATA TABLE ---
    st.subheader("Day-by-Day Macro Ledger")
    # Clean up the dataframe for display
    display_df = df_strategy[['Gold', 'DXY', 'Real_Yield', 'Position', 'Strategy_Equity']].copy()
    display_df.index = display_df.index.strftime('%Y-%m-%d')
    display_df['Position'] = display_df['Position'].map({1: 'Long', -1: 'Short', 0: 'Neutral'})
    
    # Show the most recent 30 days automatically
    st.dataframe(display_df.tail(30), use_container_width=True)

else:
    st.error("Market data feeds are currently unreachable. Please check your internet connection or try again shortly.")
