#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Example: Positive earnings news → Buy signal.
#Sudden lawsuit/CEO resignation → Sell signal.

 ##Close Price vs Moving Averages

#If 20-day Moving Average (MA20) crosses above 50-day Moving Average (MA50) → Buy Signal (uptrend).

#If MA20 falls below MA50 → Sell Signal (downtrend).

##RSI (Relative Strength Index)

#RSI > 70 → Overbought → Sell Signal

#RSI < 30 → Oversold → Buy Signal

#RSI between 30–70 → Neutral / Hold

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore", category=FutureWarning)

# --- Parameters ---
MA_PERIOD = 50
RSI_PERIOD = 14

# --- Indicators ---
def calculate_MA(series, period):
    return series.rolling(window=period).mean()

def calculate_RSI(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain, index=series.index).rolling(period).mean()
    avg_loss = pd.Series(loss, index=series.index).rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Data Fetch ---
@st.cache_data
def fetch_data(ticker, period="1y", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        return data
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return None

def build_frame(df):
    out = pd.DataFrame(index=df.index)
    out["Close"] = df["Close"].astype(float)
    out[f"MA{MA_PERIOD}"] = calculate_MA(out["Close"], MA_PERIOD)
    out["RSI"] = calculate_RSI(out["Close"], RSI_PERIOD)
    return out

def generate_signals(df):
    signals = []
    for i in range(len(df)):
        close = df["Close"].iloc[i]
        ma = df[f"MA{MA_PERIOD}"].iloc[i]
        rsi = df["RSI"].iloc[i]

        if pd.isna(ma) or pd.isna(rsi):
            signals.append("HOLD ➖")
        elif close > ma and rsi < 70:
            signals.append("BUY ✅")
        elif close < ma and rsi > 30:
            signals.append("SELL ❌")
        else:
            signals.append("HOLD ➖")

    df["Signal"] = signals
    return df

# --- Streamlit App ---
st.set_page_config(page_title="Trading Signal Dashboard", layout="wide")

st.title("📈 Trading Signal Dashboard")
st.write("Get trading signals based on **Moving Average (MA50)** and **RSI (14)**.")

# Sidebar inputs
st.sidebar.header("Settings")
tickers = st.sidebar.text_area("Enter tickers (comma-separated):", 
                               "AAPL, RYA.IR, PTSB.IR, IRES.IR, A5G.IR, GVR.IR, UPR.IR, DHG.IR, GRP.IR").split(",")
tickers = [t.strip() for t in tickers if t.strip()]

period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

# Main content
for ticker in tickers:
    df = fetch_data(ticker, period=period, interval=interval)
    if df is None or df.empty:
        st.warning(f"No data for {ticker}")
        continue

    frame = build_frame(df)
    frame = generate_signals(frame)

    latest = frame.tail(1).iloc[0]

    st.subheader(f"📊 {ticker}")
    st.write(
        f"**Close:** {latest['Close']:.2f} | "
        f"**MA{MA_PERIOD}:** {latest[f'MA{MA_PERIOD}']:.2f} | "
        f"**RSI:** {latest['RSI']:.2f} | "
        f"**Signal:** {latest['Signal']}"
    )

    # --- Chart ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    # Price + MA
    ax1.plot(frame.index, frame["Close"], label="Close", color="blue")
    ax1.plot(frame.index, frame[f"MA{MA_PERIOD}"], label=f"MA{MA_PERIOD}", color="orange")
    ax1.set_ylabel("Price")
    ax1.legend()
    ax1.grid(True)

    # RSI
    ax2.plot(frame.index, frame["RSI"], label="RSI", color="purple")
    ax2.axhline(70, color="red", linestyle="--")
    ax2.axhline(30, color="green", linestyle="--")
    ax2.set_ylabel("RSI")
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig)

    # Show table
    with st.expander(f"Show historical signals for {ticker}"):
        st.dataframe(frame.tail(30))



# In[ ]:




