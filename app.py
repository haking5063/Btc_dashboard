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
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame(columns=["time", "oi"])
        oi = pd.DataFrame(data)
        oi["time"] = pd.to_datetime(oi["timestamp"], unit="ms")
        oi["oi"] = oi["sumOpenInterestValue"].astype(float)
        return oi[["time", "oi"]]
    except Exception:
        return pd.DataFrame(columns=["time", "oi"])

@st.cache_data(ttl=1800)
def fetch_fear_greed():
    try:
        r = requests.get(
            "https://api.alternative.me/fng/?limit=1&format=json",
            timeout=15
        )
        r.raise_for_status()
        j = r.json()
        v = int(j["data"][0]["value"])
        c = j["data"][0]["value_classification"]
        return v, c
    except Exception:
        return None, "N/A"

def add_indicators(df):
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20).mean()
    df["ema50"] = df["close"].ewm(span=50).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    bb_mid = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_mid"] = bb_mid
    df["bb_upper"] = bb_mid + 2 * bb_std
    df["bb_lower"] = bb_mid - 2 * bb_std
    return df

def find_levels(df):
    sub = df.tail(60)
    res_lvls, sup_lvls = [], []
    for i in range(2, len(sub) - 2):
        h = sub["high"].iloc[i]
        l = sub["low"].iloc[i]
        if h == sub["high"].iloc[i-2:i+3].max():
            res_lvls.append(round(h, -2))
        if l == sub["low"].iloc[i-2:i+3].min():
            sup_lvls.append(round(l, -2))
    res_lvls = sorted(set(res_lvls), reverse=True)[:3]
    sup_lvls = sorted(set(sup_lvls))[:3]
    return sup_lvls, res_lvls

def get_signal(df, sup_lvls, res_lvls, fr_df, oi_df, fng_value):
    cp = df["close"].iloc[-1]
    rsi = df["rsi"].iloc[-1]
    macd = df["macd"].iloc[-1]
    macd_sig = df["macd_signal"].iloc[-1]
    bb_low = df["bb_lower"].iloc[-1]
    bb_high = df["bb_upper"].iloc[-1]
    ema20 = df["ema20"].iloc[-1]
    ema50 = df["ema50"].iloc[-1]

    ls, ss = 0, 0
    lr, sr = [], []

    if rsi < 40:
        ls += 2
        lr.append(f"RSI 과매도 ({rsi:.1f})")
    elif rsi > 60:
        ss += 2
        sr.append(f"RSI 과매수 ({rsi:.1f})")

    if macd > macd_sig:
        ls += 1
        lr.append("MACD 상향")
    else:
        ss += 1
        sr.append("MACD 하향")

    if cp < bb_low * 1.005:
        ls += 2
        lr.append("BB 하단 근접")
    elif cp > bb_high * 0.995:
        ss += 2
        sr.append("BB 상단 근접")

    if ema20 > ema50:
        ls += 1
        lr.append("EMA20 > EMA50")
    else:
        ss += 1
        sr.append("EMA20 < EMA50")

    if sup_lvls and cp < min(sup_lvls) * 1.012:
        ls += 2
        lr.append("지지선 근처")
    if res_lvls and cp > max(res_lvls) * 0.988:
        ss += 2
        sr.append("저항선 근처")

    funding_now = None
    funding_avg = None
    funding_signal = "N/A"
    if not fr_df.empty:
        funding_now = fr_df["fundingRate"].iloc[-1]
        funding_avg = fr_df["fundingRate"].tail(5).mean()
        if funding_now > 0.01:
            ss += 2
            sr.append(f"펀딩비 과열 +{funding_now:.4f}%")
            funding_signal = "롱 과열"
        elif funding_now < -0.01:
            ls += 2
            lr.append(f"펀딩비 과냉 {funding_now:.4f}%")
            funding_signal = "숏 과열"
        else:
            funding_signal = "중립"

    oi_now = None
    oi_chg = None
    oi_signal = "N/A"
    if not oi_df.empty and len(oi_df) >= 2:
        oi_now = oi_df["oi"].iloc[-1]
        oi_prev = oi_df["oi"].iloc[-2]
        oi_chg = (oi_now - oi_prev) / oi_prev * 100 if oi_prev else 0
        if oi_chg > 0.5 and cp > df["close"].iloc[-2]:
            ls += 1
            lr.append(f"OI 증가 +{oi_chg:.2f}%")
            oi_signal = "상승 동반 OI 증가"
        elif oi_chg > 0.5 and cp < df["close"].iloc[-2]:
            ss += 1
            sr.append(f"OI 증가 +{oi_chg:.2f}%")
            oi_signal = "하락 동반 OI 증가"
        else:
            oi_signal = "변화 없음"

    if fng_value is not None:
        if fng_value <= 25:
            ls += 2
            lr.append(f"공포탐욕 극단 공포 ({fng_value})")
        elif fng_value >= 75:
            ss += 2
            sr.append(f"공포탐욕 극단 탐욕 ({fng_value})")

    liq_long = cp * (1 - 1/15.5)
    liq_short = cp * (1 + 1/14.5)

    if ls > ss:
        sig = "🟢 롱 우세"
    elif ss > ls:
        sig = "🔴 숏 우세"
    else:
        sig = "⚪ 중립"

    return {
        "cp": cp, "rsi": rsi, "ls": ls, "ss": ss,
        "lr": lr, "sr": sr,
        "liq_long": liq_long, "liq_short": liq_short,
        "signal": sig,
        "funding_now": funding_now,
        "funding_avg": funding_avg,
        "funding_signal": funding_signal,
        "oi_now": oi_now,
        "oi_chg": oi_chg,
        "oi_signal": oi_signal,
        "fng_value": fng_value
    }

def calc_tpsl(entry, side, lev, sl_pct, tp_pct):
    if side == "LONG":
        sl = entry * (1 - sl_pct / 100)
        tp = entry * (1 + tp_pct / 100)
        liq = entry * (1 - 1 / lev)
    else:
        sl = entry * (1 + sl_pct / 100)
        tp = entry * (1 - tp_pct / 100)
        liq = entry * (1 + 1 / lev)
    return sl, tp, liq

# ── 메인 UI ──────────────────────────────────────────────────────
st.title("🚀 BTC 15x 레버리지 대시보드")
st.caption("1H 기준 | CoinGecko + Binance 공개 API | 5분마다 자동 갱신")

if st.button("🔄 새로고침"):
    st.cache_data.clear()
    st.rerun()

try:
    df = add_indicators(fetch_price_data())
    fr_df = fetch_funding_rate()
    oi_df = fetch_open_interest()
    fng_value, fng_class = fetch_fear_greed()
    sup_lvls, res_lvls = find_levels(df)
    sig = get_signal(df, sup_lvls, res_lvls, fr_df, oi_df, fng_value)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("현재가", f"${sig['cp']:,.0f}")
    c2.metric("RSI", f"{sig['rsi']:.1f}",
              delta="과매도" if sig['rsi'] < 40 else ("과매수" if sig['rsi'] > 60 else "중립"))
    c3.metric("🔴 15x 롱 청산가", f"${sig['liq_long']:,.0f}")
    c4.metric("🟢 15x 숏 청산가", f"${sig['liq_short']:,.0f}")
    c5.metric("공포탐욕", f"{fng_value if fng_value is not None else 'N/A'}",
              delta=fng_class if fng_class else "")

    sig_color = "green" if sig["ls"] > sig["ss"] else ("red" if sig["ss"] > sig["ls"] else "orange")
    st.markdown(
        f"<h2 style='text-align:center;color:{sig_color}'>"
        f"{sig['signal']} &nbsp; 롱 {sig['ls']}/15 &nbsp;|&nbsp; 숏 {sig['ss']}/15</h2>",
        unsafe_allow_html=True
    )

    col_l, col_r = st.columns([3, 1])

    with col_l:
        plot_df = df.tail(60)
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.15, 0.2, 0.15],
            vertical_spacing=0.04
        )

        fig.add_trace(go.Candlestick(
            x=plot_df["time"], open=plot_df["open"],
            high=plot_df["high"], low=plot_df["low"],
            close=plot_df["close"],
            increasing_line_color="#00c878",
            decreasing_line_color="#ff4455",
            showlegend=False), row=1, col=1)

        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["ema20"],
            line=dict(color="#58a6ff", width=2), name="EMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["ema50"],
            line=dict(color="#ffd60a", width=2), name="EMA50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["bb_upper"],
            line=dict(color="rgba(100,100,200,0.5)", width=1),
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["bb_lower"],
            line=dict(color="rgba(100,100,200,0.5)", width=1),
            fill="tonexty", fillcolor="rgba(100,100,200,0.06)",
            showlegend=False), row=1, col=1)

        for s in sup_lvls:
            fig.add_hline(y=s, line=dict(color="#00c878", dash="dash", width=1.2), row=1, col=1)
        for r in res_lvls:
            fig.add_hline(y=r, line=dict(color="#ff4455", dash="dash", width=1.2), row=1, col=1)
        fig.add_hline(y=sig["liq_long"],
            line=dict(color="#ff1a40", dash="longdash", width=2), row=1, col=1)
        fig.add_hline(y=sig["liq_short"],
            line=dict(color="#00e88a", dash="longdash", width=2), row=1, col=1)

        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["rsi"],
            line=dict(color="#58a6ff", width=2), showlegend=False), row=2, col=1)
        fig.add_hline(y=70, line=dict(color="#ff4455", dash="dot", width=1), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="#00c878", dash="dot", width=1), row=2, col=1)

        hist_colors = ["#00c878" if v >= 0 else "#ff4455" for v in plot_df["macd_hist"]]
        fig.add_trace(go.Bar(x=plot_df["time"], y=plot_df["macd_hist"],
            marker_color=hist_colors, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["macd"],
            line=dict(color="#58a6ff", width=2), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["macd_signal"],
            line=dict(color="#ffd60a", width=1.5, dash="dot"), showlegend=False), row=3, col=1)

        vol_colors = [
            "#00c878" if plot_df["close"].iloc[i] >= plot_df["open"].iloc[i] else "#ff4455"
            for i in range(len(plot_df))
        ]
        fig.add_trace(go.Bar(x=plot_df["time"], y=plot_df["volume"],
            marker_color=vol_colors, showlegend=False), row=4, col=1)

        fig.update_layout(
            height=900, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=20, b=10), legend_orientation="h"
        )
        for i in range(1, 5):
            fig.update_xaxes(gridcolor="#21262d", row=i, col=1)
            fig.update_yaxes(gridcolor="#21262d", row=i, col=1)
        fig.update_yaxes(title_text="가격", tickformat="$,.0f", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="거래량", row=4, col=1)

        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("📋 진입 근거")
        st.markdown("**🟢 롱 신호**")
        for item in sig["lr"]:
            st.write(f"✅ {item}")
        st.markdown("**🔴 숏 신호**")
        for item in sig["sr"]:
            st.write(f"❌ {item}")

        st.divider()
        st.subheader("📌 펀딩비 / OI")
        if sig["funding_now"] is not None:
            st.write(f"현재 펀딩비: {sig['funding_now']:.4f}%")
            st.write(f"5회 평균: {sig['funding_avg']:.4f}%")
        else:
            st.write("현재 펀딩비: N/A")
        st.write(f"펀딩비 해석: {sig['funding_signal']}")
        if sig["oi_now"] is not None:
            st.write(f"현재 OI: ${sig['oi_now']:,.0f}")
            st.write(f"OI 변화율: {sig['oi_chg']:.2f}%")
        else:
            st.write("현재 OI: N/A")
        st.write(f"OI 해석: {sig['oi_signal']}")

        st.divider()
        st.subheader("📍 지지/저항")
        st.markdown("**저항선**")
        for r in res_lvls:
            st.write(f"🔴 ${r:,.0f}")
        st.markdown("**지지선**")
        for s in sup_lvls:
            st.write(f"🟢 ${s:,.0f}")

        st.divider()
        st.subheader("🧮 TP / SL 계산기")
        side = st.selectbox("방향", ["LONG", "SHORT"])
        entry = st.number_input("진입가 ($)", value=float(sig["cp"]), step=100.0)
        lev = st.slider("레버리지", 1, 50, 15)
        sl_pct = st.number_input("손절 %", value=3.5, step=0.1)
        tp_pct = st.number_input("익절 %", value=8.0, step=0.1)
        sl_price, tp_price, liq_price = calc_tpsl(entry, side, lev, sl_pct, tp_pct)
        st.write(f"✂️ 손절가: **${sl_price:,.0f}**")
        st.write(f"🎯 익절가: **${tp_price:,.0f}**")
        st.write(f"💥 청산가: **${liq_price:,.0f}**")

except Exception as e:
    st.error(f"데이터 로딩 실패: {e}")
    st.info("새로고침 버튼을 눌러주세요.")
