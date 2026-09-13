import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

st.set_page_config(page_title="XAUUSD Structural Levels", layout="wide")
st.title("Gold (XAUUSD) Macro & Execution Zones")

@st.cache_data(ttl=3600)
def fetch_gold_data(period="30d"):
    df_1h = yf.download('GC=F', period=period, interval='1h', progress=False)
    df_4h = df_1h.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    return df_1h, df_4h

def find_structural_levels(df, window):
    maxima_indices = argrelextrema(df['High'].values, np.greater, order=window)[0]
    resistances = df['High'].iloc[maxima_indices].values
    minima_indices = argrelextrema(df['Low'].values, np.less, order=window)[0]
    supports = df['Low'].iloc[minima_indices].values
    return supports, resistances

df_1h, df_4h = fetch_gold_data()
supports_4h, resistances_4h = find_structural_levels(df_4h, window=5)

st.subheader("4-Hour Macro Structure")
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df_4h.index, df_4h['Close'], label='H4 Close Price', color='black', linewidth=1.5)

for res in resistances_4h:
    ax.axhline(res, color='red', linestyle='--', alpha=0.5, linewidth=1)
for sup in supports_4h:
    ax.axhline(sup, color='green', linestyle='--', alpha=0.5, linewidth=1)

ax.set_title('XAUUSD 4-Hour Structural Support & Resistance')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend(['Closing Price', 'Resistance (Supply)', 'Support (Demand)'])
ax.grid(alpha=0.2)

st.pyplot(fig)
