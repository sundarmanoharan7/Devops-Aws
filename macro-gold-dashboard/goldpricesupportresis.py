@st.cache_data(ttl=3600)
def fetch_gold_data(period="30d"):
    # 1. Fetch highly reliable structural data (GC=F)
    df_1h = yf.download('GC=F', period=period, interval='1h', progress=False)
    
    if df_1h.empty:
        st.error("Failed to fetch data from Yahoo Finance. The API might be down.")
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(df_1h.columns, pd.MultiIndex):
        df_1h.columns = df_1h.columns.get_level_values(0)

    # 2. Direct Mathematical Anchor to your exact Spot Price
    broker_spot_price = 4348.00  # The exact closing price you want to anchor to
    
    latest_close = df_1h['Close'].dropna().iloc[-1]
    premium_offset = latest_close - broker_spot_price
    
    # Shift all price columns down by the exact offset
    for col in ['Open', 'High', 'Low', 'Close']:
        df_1h[col] = df_1h[col] - premium_offset

    # 3. Resample to 4H Structure
    df_4h = df_1h.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    
    return df_1h, df_4h
