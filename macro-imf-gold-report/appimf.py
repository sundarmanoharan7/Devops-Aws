import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from fredapi import Fred
from dbnomics import fetch_series

# --- CONFIGURATION ---
FRED_API_KEY = "YOUR_API_KEY_HERE"  # Paste your FRED API key here

st.set_page_config(page_title="Institutional Gold Dashboard", layout="wide")
st.title("Macroeconomic & Central Bank Gold Accumulation Dashboard")

# --- DATA FETCHING FUNCTIONS ---
@st.cache_data(ttl=3600)
def fetch_fred_data():
    """Fetches live CPI and 10-Year Yield data from FRED."""
    try:
        fred = Fred(api_key=FRED_API_KEY)
        
        # Fetch YoY CPI Inflation
        cpi = fred.get_series('CPIAUCSL')
        cpi_yoy = cpi.pct_change(periods=12).dropna() * 100
        
        # Fetch 10-Year Treasury Yields
        yield_10y = fred.get_series('DGS10').dropna()
        
        return cpi_yoy.tail(60), yield_10y.tail(60) # Get last 5 years of data
    except Exception as e:
        st.error(f"FRED API Error: {e}")
        return None, None

@st.cache_data(ttl=86400) # Cache for 24 hours, IMF updates monthly
def fetch_imf_gold_data():
    """
    Fetches global Central Bank Gold Reserves (in Millions of Fine Troy Ounces)
    using the free DBnomics API (which aggregates IMF IFS data).
    """
    try:
        # DBnomics code for IMF -> IFS -> Official Reserve Assets, Gold (World)
        # Note: 'W00' is the IMF country code for the entire World
        df_imf = fetch_series("IMF/IFS/M.W00.RAFAGOLDV_OZT")
        
        # Clean the dataframe to just Date and Value
        df_imf = df_imf[['period', 'value']].dropna()
        df_imf.rename(columns={'period': 'Date', 'value': 'Millions of Ounces'}, inplace=True)
        df_imf.set_index('Date', inplace=True)
        
        return df_imf.tail(60) # Get last 5 years of data
    except Exception as e:
        st.error(f"IMF Data Error: {e}")
        return None

# --- LOAD DATA ---
st.write("Fetching live institutional data from FRED and the IMF...")
cpi_data, yield_data = fetch_fred_data()
imf_gold_data = fetch_imf_gold_data()

# --- DASHBOARD UI ---
if cpi_data is not None and imf_gold_data is not None:
    
    # Create two columns for the charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Macro Headwinds: CPI vs 10-Year Yields")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=cpi_data.index, y=cpi_data.values, name="US YoY CPI (%)", line=dict(color='red')))
        fig1.add_trace(go.Scatter(x=yield_data.index, y=yield_data.values, name="10-Year Yield (%)", line=dict(color='blue')))
        fig1.update_layout(height=400, template="plotly_white", legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("When these lines spike, retail and institutional traders typically sell gold.")

    with col2:
        st.subheader("The Price Floor: Global Central Bank Gold Reserves")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=imf_gold_data.index, y=imf_gold_data['Millions of Ounces'], name="Global Reserves", marker_color='gold'))
        
        # Calculate a 3-month moving average of purchases to show trend
        imf_gold_data['Trend'] = imf_gold_data['Millions of Ounces'].rolling(window=3).mean()
        fig2.add_trace(go.Scatter(x=imf_gold_data.index, y=imf_gold_data['Trend'], name="Accumulation Trend", line=dict(color='black', width=2)))
        
        fig2.update_layout(height=400, template="plotly_white", yaxis_title="Millions of Troy Ounces")
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Central Bank accumulation acts as a 'price floor', ignoring short-term macro headwinds.")

else:
    st.warning("Data fetch failed. Check your API keys and internet connection.")
