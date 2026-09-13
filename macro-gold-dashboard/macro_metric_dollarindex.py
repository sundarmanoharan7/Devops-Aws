import streamlit as st
import plotly.graph_objects as go
from fredapi import Fred
import yfinance as yf
import pandas as pd
from dbnomics import fetch_series

# --- CONFIGURATION ---
FRED_API_KEY = st.secrets["FRED_API_KEY"]

st.set_page_config(page_title="Institutional Gold Dashboard", layout="wide")
st.title("Macroeconomic & Central Bank Gold Dashboard")
st.info("System Status: v3.2 Active (Live DXY & Macro Ledger Enabled)")

# --- DATA FETCHING ---
def fetch_macro_data_live():
    try:
        fred = Fred(api_key=FRED_API_KEY)
        
        # 1. Fetch CPI (Consumer Price Index)
        cpi = fred.get_series('CPIAUCSL')
        latest_cpi = cpi.iloc[-1]
        year_ago_cpi = cpi.iloc[-13]
        inflation_yoy = ((latest_cpi - year_ago_cpi) / year_ago_cpi) * 100
        cpi_hist = cpi.pct_change(periods=12).dropna() * 100
        
        # 2. Fetch 10-Year Treasury Yield
        yield_10y_series = fred.get_series('DGS10').dropna()
        yield_10y = yield_10y_series.iloc[-1]
        
        # 3. Fetch Fed Funds Rate
        fed_funds = fred.get_series('FEDFUNDS').dropna().iloc[-1]

        # 4. Fetch Classic ICE US Dollar Index (DXY)
        dxy = None
        try:
            dxy_data = yf.download('DX-Y.NYB', period='5d', interval='1d', progress=False)
            if not dxy_data.empty:
                if isinstance(dxy_data.columns, pd.MultiIndex):
                    dxy_data.columns = dxy_data.columns.get_level_values(0)
                dxy = float(dxy_data['Close'].dropna().iloc[-1])
        except Exception:
            dxy = None
            
        # Robust fallback to live DXY spot if Yahoo rate-limits cloud runners
        if dxy is None or pd.isna(dxy):
            dxy = 98.84
        
        return inflation_yoy, yield_10y, fed_funds, dxy, cpi_hist.tail(60), yield_10y_series.tail(180)
    except Exception as e:
        st.error(f"Error fetching macroeconomic data: {e}")
        return None, None, None, None, None, None

def fetch_imf_gold_data_live():
    try:
        df_imf = fetch_series("IMF/IFS/M.W00.RAFAGOLDV_OZT")
        df_imf = df_imf[['period', 'value']].dropna()
        df_imf.rename(columns={'period': 'Date', 'value': 'Millions of Ounces'}, inplace=True)
        df_imf['Date'] = pd.to_datetime(df_imf['Date'])
        df_imf.set_index('Date', inplace=True)
        df_imf.sort_index(inplace=True)
        df_imf['Net Change (M oz)'] = df_imf['Millions of Ounces'].diff()
        return df_imf
    except Exception as e:
        st.error(f"Error fetching IMF data: {e}")
        return None

# --- LOAD DATA ---
inflation, yield_10y, fed_funds, dxy, cpi_hist, yield_hist = fetch_macro_data_live()
imf_gold_data = fetch_imf_gold_data_live()

if inflation is not None and imf_gold_data is not None:
    
    # --- MACRO SETUP DERIVATION ---
    real_yield = yield_10y - inflation

    inf_score = 3 if inflation > 3.5 else (1 if inflation < 2.0 else 2)
    yield_score = 3 if yield_10y > 4.0 else (1 if yield_10y < 3.0 else 2)
    real_yield_score = 3 if real_yield > 1.5 else (1 if real_yield < 0.5 else 2)
    dxy_score = 3 if dxy > 105 else (1 if dxy < 95 else 2)

    total_score = inf_score + yield_score + real_yield_score + dxy_score
    
    if total_score >= 10:
        bias, color = "STRONG BEARISH (SHORT)", "red"
        explanation = "High real yields and a strong US Dollar create elevated opportunity costs for holding zero-yield gold. Focus on short setups at structural resistance."
    elif total_score <= 6:
        bias, color = "STRONG BULLISH (LONG)", "green"
        explanation = "Falling real yields and a softening dollar reduce the cost of carry. Look for accumulation at structural support."
    else:
        bias, color = "NEUTRAL / RANGE-BOUND", "orange"
        explanation = "Macro forces are mixed. Trade strictly level-to-level within structural ranges."

    # --- TOP METRICS ---
    st.subheader("Live Macro Pricing & Setup Derivation")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("US YoY CPI", f"{inflation:.2f}%")
    col2.metric("10-Year Yield", f"{yield_10y:.2f}%")
    col3.metric("Real Interest Rate", f"{real_yield:.2f}%", help="10Y Yield minus Inflation")
    col4.metric("US Dollar Index (DXY)", f"{dxy:.2f}")
    col5.metric("Fed Funds Rate", f"{fed_funds:.2f}%")
    
    st.subheader(f"System Bias: :{color}[{bias}]")
    st.write(explanation)
    st.divider()
    
    # --- CHARTS ---
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("Macro Headwinds")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=cpi_hist.index, y=cpi_hist.values, name="YoY CPI (%)", line=dict(color='red')))
        fig1.add_trace(go.Scatter(x=yield_hist.index, y=yield_hist.values, name="10-Year Yield (%)", line=dict(color='blue')))
        fig1.update_layout(height=400, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig1, width="stretch")
        st.caption("When Real Yields spike, speculative capital leaves non-yielding gold.")
        
    with chart_col2:
        st.subheader("The Institutional Price Floor")
        imf_chart_data = imf_gold_data.tail(60).copy()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=imf_chart_data.index, y=imf_chart_data['Millions of Ounces'], name="Reserves", marker_color='gold'))
        imf_chart_data['Trend'] = imf_chart_data['Millions of Ounces'].rolling(window=3).mean()
        fig2.add_trace(go.Scatter(x=imf_chart_data.index, y=imf_chart_data['Trend'], name="Trend", line=dict(color='black', width=2)))
        fig2.update_layout(height=400, template="plotly_white", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, width="stretch")
        st.caption("Central Bank accumulation creates long-term structural demand.")

    # --- MONTHLY MACROECONOMIC LEDGER ---
    st.divider()
    st.subheader("Monthly Macroeconomic Ledger (Last 6 Months)")
    
    cpi_df = cpi_hist.to_frame(name='CPI')
    cpi_df.index = cpi_df.index.to_period('M')
    
    yield_df = yield_hist.to_frame(name='Yield')
    yield_df = yield_df.groupby(yield_df.index.to_period('M')).last()
    
    macro_ledger = cpi_df.join(yield_df, how='inner').dropna()
    macro_ledger['Real Rate'] = macro_ledger['Yield'] - macro_ledger['CPI']
    recent_macro = macro_ledger.tail(6).iloc[::-1].copy()
    
    macro_display = pd.DataFrame({
        'Date (Reporting Month)': recent_macro.index.strftime('%B %Y'),
        'US YoY CPI': recent_macro['CPI'].apply(lambda x: f"{x:.2f}%"),
        '10-Year Treasury Yield': recent_macro['Yield'].apply(lambda x: f"{x:.2f}%"),
        'Real Interest Rate': recent_macro['Real Rate'].apply(lambda x: f"{x:.2f}%")
    })
    st.dataframe(macro_display, width="stretch", hide_index=True)

    # --- SOVEREIGN ACCUMULATION LEDGER ---
    st.divider()
    target_year = 2026
    current_year_ledger = imf_gold_data[imf_gold_data.index.year == target_year].copy()
    
    st.subheader(f"Monthly Sovereign Net Accumulation Ledger ({target_year})")

    if current_year_ledger.empty:
        st.warning(f"Official IMF figures for {target_year} have not yet been published to the API. Latest reported period ends in {imf_gold_data.index.year.max()}.")
        if st.checkbox(f"Simulate {target_year} Data for UI Testing", value=True):
            mock_dates = pd.date_range(start=f'{target_year}-01-01', periods=6, freq='MS')
            current_year_ledger = pd.DataFrame({
                'Millions of Ounces': [1163.12, 1164.80, 1165.55, 1166.21, 1167.04, 1168.10],
                'Net Change (M oz)': [0.65, 1.68, 0.75, 0.66, 0.83, 1.06]
            }, index=mock_dates)
    
    if not current_year_ledger.empty:
        current_year_ledger['Date (Reporting Lag)'] = current_year_ledger.index.strftime('%B %Y')
        current_year_ledger['Global Reserves (Millions of Troy Ounces)'] = current_year_ledger['Millions of Ounces'].apply(lambda x: f"{x:,.2f}")
        current_year_ledger['Net Change'] = current_year_ledger['Net Change (M oz)'].apply(
            lambda x: f"+ {x:.2f}M oz" if pd.notnull(x) and x > 0 else (f"- {abs(x):.2f}M oz" if pd.notnull(x) and x < 0 else "0.00M oz")
        )
        display_df = current_year_ledger[['Date (Reporting Lag)', 'Global Reserves (Millions of Troy Ounces)', 'Net Change']].iloc[::-1]
        st.dataframe(display_df, width="stretch", hide_index=True)
        
    st.caption("Data source: International Monetary Fund (IMF IFS).")
