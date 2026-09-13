import streamlit as st
import plotly.graph_objects as go
from fredapi import Fred
import pandas as pd
from dbnomics import fetch_series

# --- CONFIGURATION ---
FRED_API_KEY = st.secrets["FRED_API_KEY"]

st.set_page_config(page_title="Institutional Gold Dashboard", layout="wide")
st.title("Macroeconomic & Central Bank Gold Dashboard")

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_macro_data():
    try:
        fred = Fred(api_key=FRED_API_KEY)
        
        # Fetch CPI 
        cpi = fred.get_series('CPIAUCSL')
        latest_cpi = cpi.iloc[-1]
        year_ago_cpi = cpi.iloc[-13]
        inflation_yoy = ((latest_cpi - year_ago_cpi) / year_ago_cpi) * 100
        cpi_hist = cpi.pct_change(periods=12).dropna() * 100
        
        # Fetch 10-Year Treasury Yield
        yield_10y_series = fred.get_series('DGS10').dropna()
        yield_10y = yield_10y_series.iloc[-1]
        
        # Fetch Federal Funds Effective Rate
        fed_funds = fred.get_series('FEDFUNDS').dropna().iloc[-1]
        
        return inflation_yoy, yield_10y, fed_funds, cpi_hist.tail(60), yield_10y_series.tail(60)
    except Exception as e:
        st.error(f"Error fetching FRED data: {e}")
        return None, None, None, None, None

@st.cache_data(ttl=86400)  # Cache for 24 hours
def fetch_imf_gold_data():
    try:
        # Fetch World Gold Reserves from IMF IFS via DBnomics
        df_imf = fetch_series("IMF/IFS/M.W00.RAFAGOLDV_OZT")
        df_imf = df_imf[['period', 'value']].dropna()
        df_imf.rename(columns={'period': 'Date', 'value': 'Millions of Ounces'}, inplace=True)
        
        # Ensure Date is parsed as datetime
        df_imf['Date'] = pd.to_datetime(df_imf['Date'])
        df_imf.set_index('Date', inplace=True)
        df_imf.sort_index(inplace=True)
        
        # Calculate Month-over-Month Net Sovereign Buying/Selling
        df_imf['Net Change (M oz)'] = df_imf['Millions of Ounces'].diff()
        
        return df_imf
    except Exception as e:
        st.error(f"Error fetching IMF data: {e}")
        return None

# --- LOAD DATA ---
st.write("Fetching live institutional data from FRED and the IMF...")
inflation, yield_10y, fed_funds, cpi_hist, yield_hist = fetch_macro_data()
imf_gold_data = fetch_imf_gold_data()

if inflation is not None and imf_gold_data is not None:
    
    # --- TOP SECTION: LIVE SCORING ---
    st.subheader("Live Macro Pricing")
    col1, col2, col3 = st.columns(3)
    col1.metric("US YoY Inflation (CPI)", f"{inflation:.2f}%")
    col2.metric("10-Year Treasury Yield", f"{yield_10y:.2f}%")
    col3.metric("Fed Funds Rate", f"{fed_funds:.2f}%")
    
    # Algorithmic Logic
    inf_score = 3 if inflation > 3.0 else (1 if inflation < 2.0 else 2)
    yield_score = 3 if yield_10y > 4.0 else (1 if yield_10y < 3.5 else 2)
    fed_score = 3 if fed_funds > 4.5 else (1 if fed_funds < 3.0 else 2)
    total_score = inf_score + yield_score + fed_score
    
    if total_score >= 7:
        bias, color = "STRONG BEARISH (SHORT)", "red"
        explanation = "Inflation and yields are running hot, creating a high opportunity cost for holding zero-yield gold. Look for short setups at structural resistance."
    elif total_score <= 4:
        bias, color = "STRONG BULLISH (LONG)", "green"
        explanation = "Cooling inflation and falling yields drastically reduce the opportunity cost of holding gold. Look for long setups at structural support."
    else:
        bias, color = "NEUTRAL / RANGE-BOUND", "orange"
        explanation = "Macro indicators are mixed. Expect gold to trade level-to-level without a strong macro tailwind."
        
    st.subheader(f"System Bias: :{color}[{bias}]")
    st.write(explanation)
    
    st.divider()
    
    # --- BOTTOM SECTION: VISUAL CHARTS ---
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Macro Headwinds")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=cpi_hist.index, y=cpi_hist.values, name="YoY CPI (%)", line=dict(color='red')))
        fig1.add_trace(go.Scatter(x=yield_hist.index, y=yield_hist.values, name="10-Year Yield (%)", line=dict(color='blue')))
        fig1.update_layout(height=400, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig1, use_container_width=True)
        st.caption("When these lines spike, retail traders typically sell gold.")
        
    with chart_col2:
        st.subheader("The Institutional Price Floor")
        imf_chart_data = imf_gold_data.tail(60).copy()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=imf_chart_data.index, y=imf_chart_data['Millions of Ounces'], name="Reserves", marker_color='gold'))
        imf_chart_data['Trend'] = imf_chart_data['Millions of Ounces'].rolling(window=3).mean()
        fig2.add_trace(go.Scatter(x=imf_chart_data.index, y=imf_chart_data['Trend'], name="Trend", line=dict(color='black', width=2)))
        fig2.update_layout(height=400, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("Central Bank accumulation absorbs macro selling pressure.")

    # --- MONTHLY SOVEREIGN ACCUMULATION LEDGER (CURRENT YEAR) ---
    st.divider()
    
    # Filter for the current year dynamically
    current_year = pd.Timestamp.now().year
    current_year_ledger = imf_gold_data[imf_gold_data.index.year == current_year].copy()
    
    # Fallback to the latest available year if current year has no data yet due to IMF reporting lag
    display_year = current_year
    if current_year_ledger.empty:
        display_year = imf_gold_data.index.year.max()
        current_year_ledger = imf_gold_data[imf_gold_data.index.year == display_year].copy()
        st.info(f"Official IMF figures for {current_year} are pending publication due to reporting lag. Showing data for {display_year}.")

    st.subheader(f"Monthly Sovereign Net Accumulation Ledger ({display_year})")

    # Format values for table display
    current_year_ledger['Date (Reporting Lag)'] = current_year_ledger.index.strftime('%B %Y')
    current_year_ledger['Global Reserves (Millions of Troy Ounces)'] = current_year_ledger['Millions of Ounces'].apply(lambda x: f"{x:,.2f}")
    current_year_ledger['Net Change'] = current_year_ledger['Net Change (M oz)'].apply(
        lambda x: f"+ {x:.2f}M oz" if x > 0 else (f"- {abs(x):.2f}M oz" if x < 0 else "0.00M oz")
    )
    
    # Display Year-to-Date (YTD) Net Accumulation metric
    ytd_net = current_year_ledger['Net Change (M oz)'].sum()
    ytd_sign = "+" if ytd_net > 0 else ""
    st.metric(
        label=f"{display_year} Year-to-Date Net Accumulation",
        value=f"{ytd_sign}{ytd_net:.2f}M oz",
        delta=f"{ytd_sign}{ytd_net:.2f}M oz vs Prior Year End"
    )
    
    # Build clean output DataFrame (sorted newest month first)
    display_df = current_year_ledger[['Date (Reporting Lag)', 'Global Reserves (Millions of Troy Ounces)', 'Net Change']].iloc[::-1]
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption("Data source: International Monetary Fund (IMF IFS). New reporting points are updated monthly.")
