import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="BTC 15x Dashboard", layout="wide")

BINANCE_BASE = "https://fapi.binance.com"

@st.cache_data(ttl=300)
def fetch_price_data():
    r = requests.get(
        "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
        params={"vs_currency": "usd", "days": "4", "interval": "hourly"},
        timeout=15
    )
    r.raise_for_status()
    mc = r.json()
    prices = pd.DataFrame(mc["prices"], columns=["ts", "close"])
    volumes = pd.DataFrame(mc["total_volumes"], columns=["ts", "volume"])
    df = prices.merge(volumes, on="ts")
    df["time"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.tail(72).reset_index(drop=True)
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1) * (1 + np.random.uniform(0, 0.003, len(df)))
    df["low"] = df[["open", "close"]].min(axis=1) * (1 - np.random.uniform(0, 0.003, len(df)))
    return df

@st.cache_data(ttl=900)
def fetch_funding_rate():
    try:
        r = requests.get(
            f"{BINANCE_BASE}/fapi/v1/fundingRate",
            params={"symbol": "BTCUSDT", "limit": 10},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame(columns=["time", "fundingRate"])
        fr = pd.DataFrame(data)
        fr["time"] = pd.to_datetime(fr["fundingTime"], unit="ms")
        fr["fundingRate"] = fr["fundingRate"].astype(float) * 100
        return fr[["time", "fundingRate"]]
    except Exception:
        return pd.DataFrame(columns=["time", "fundingRate"])

@st.cache_data(ttl=900)
def fetch_open_interest():
    try:
        r = requests.get(
            f"{BINANCE_BASE}/futures/data/openInterestHist",
            params={"symbol": "BTCUSDT", "period": "1h", "limit": 30},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or 
