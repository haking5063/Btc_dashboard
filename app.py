import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="BTC 15x Dashboard", layout="wide")

BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_DAPI = "https://fapi.binance.com/futures/data"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

COINGECKO_IDS = {
    "BTCUSDT": "bitcoin",
    "ETHUSDT": "ethereum",
    "SOLUSDT": "solana",
    "XRPUSDT": "ripple"
}

@st.cache_data(ttl=60)
def fetch_klines(symbol, interval="1h", limit=100):
    try:
        r = requests.get(f"{BINANCE_FAPI}/fapi/v1/klines", params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=["time","open","high","low","close","volume","close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_klines_fallback(symbol, interval="1h", limit=100):
    df = fetch_klines(symbol, interval, limit)
    if not df.empty and "close" in df.columns:
        return df
    try:
        coin_id = COINGECKO_IDS.get(symbol, "bitcoin")
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart", params={"vs_currency": "usd", "days": "4", "interval": "hourly"}, timeout=15)
        r.raise_for_status()
        mc = r.json()
        prices = pd.DataFrame(mc.get("prices", []), columns=["ts", "close"])
        volumes = pd.DataFrame(mc.get("total_volumes", []), columns=["ts", "volume"])
        if prices.empty or volumes.empty:
            retur
