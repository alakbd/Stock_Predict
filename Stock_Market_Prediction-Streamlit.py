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
import requests
from bs4 import BeautifulSoup
import warnings
import matplotlib.pyplot as plt

warnings.simplefilter(action="ignore", category=FutureWarning)

# --- Config ---
MA_PERIOD = 50
RSI_PERIOD = 14

st.set_page_config(page_title="Trading Signal Dashboard", layout="wide")

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

# --- Crash / Drawdown Check ---
def detect_crash(df, threshold=15):
    rolling_max = df["Close"].cummax()
    drawdown = (df["Close"] / rolling_max - 1) * 100
    df["Drawdown"] = drawdown
    recent = drawdown.iloc[-1]
    if recent < -threshold:
        return f"⚠️ Market Crash Warning! Current Drawdown: {recent:.2f}%"
    return f"✅ Stable. Current Drawdown: {recent:.2f}%"

# --- Performance Metrics ---
def calculate_performance(df):
    if df is None or df.empty:
        return None
    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]
    total_return = (end_price / start_price - 1) * 100
    days = (df.index[-1] - df.index[0]).days
    if days > 0:
        cagr = ((end_price / start_price) ** (365 / days) - 1) * 100
    else:
        cagr = np.nan
    rolling_max = df["Close"].cummax()
    drawdown = (df["Close"] / rolling_max - 1).min() * 100
    daily_returns = df["Close"].pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100
    return {
        "Total Return (%)": total_return,
        "CAGR (%)": cagr,
        "Max Drawdown (%)": drawdown,
        "Volatility (%)": volatility,
    }

# --- Live News Scraper ---
@st.cache_data(ttl=3600)  # refresh every hour
def fetch_dynamic_news():
    news_items = []
    sources = {
        "Reuters Commodities": "https://www.reuters.com/markets/commodities/",
        "Irish Times Markets": "https://www.irishtimes.com/business/markets/",
    }
    for source, url in sources.items():
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                title = a.get_text(strip=True)
                href = a["href"]
                if title and len(title) > 40:
                    link = href if href.startswith("http") else url.rstrip("/") + "/" + href.lstrip("/")
                    news_items.append((title, link))
        except Exception as e:
            news_items.append((f"⚠️ Could not fetch {source}: {e}", url))
    return news_items[:8]

# --- Streamlit UI ---
st.title("📈 Trading Signal Dashboard with Crash Warnings")
st.write("Signals based on **MA50**, **RSI(14)**, crash detection, and live market news.")

# Sidebar
st.sidebar.header("Settings")
tickers_input = st.sidebar.text_area(
    "Tickers (comma-separated):",
    "AAPL, RYA.IR, PTSB.IR, GC=F, CL=F, BZ=F",
    key="ticker_input"
)
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]
period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

# News Section
st.sidebar.markdown("---")
st.sidebar.header("📰 Live Market News")
news = fetch_dynamic_news()
if news:
    for title, link in news:
        st.sidebar.markdown(f"- [{title}]({link})")
else:
    st.sidebar.write("No news available right now.")

# --- Main Content ---
for ticker in tickers:
    st.subheader(f"📌 {ticker}")
    df = fetch_data(ticker, period=period, interval=interval)
    if df is None or df.empty:
        st.warning(f"No data for {ticker}")
        continue

    frame = build_frame(df)
    frame = generate_signals(frame)
    crash_msg = detect_crash(frame)

    # Show latest signal
    row = frame.tail(1).iloc[0]
    st.write(
        f"Close: {row['Close']:.2f} | "
        f"MA{MA_PERIOD}: {row[f'MA{MA_PERIOD}']:.2f} | "
        f"RSI: {row['RSI']:.2f} | "
        f"Signal: {row['Signal']}"
    )
    st.info(crash_msg)

    # Performance metrics
    perf = calculate_performance(frame)
    if perf:
        cols = st.columns(4)
        cols[0].metric("Total Return (%)", f"{perf['Total Return (%)']:.2f}")
        cols[1].metric("CAGR (%)", f"{perf['CAGR (%)']:.2f}")
        cols[2].metric("Max Drawdown (%)", f"{perf['Max Drawdown (%)']:.2f}")
        cols[3].metric("Volatility (%)", f"{perf['Volatility (%)']:.2f}")

    # Chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame.index, frame["Close"], label="Close")
    ax.plot(frame.index, frame[f"MA{MA_PERIOD}"], label=f"MA{MA_PERIOD}")
    ax.set_title(f"{ticker} Price & MA")
    ax.legend()
    st.pyplot(fig)


# In[ ]:




