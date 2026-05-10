import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

st.set_page_config(page_title="BTC 15x Dashboard", layout="wide")

BINANCE_FAPI = "https://fapi.binance.com"
BINANCE_DAPI = "https://fapi.binance.com/futures/data"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

@st.cache_data(ttl=60)
def fetch_klines(symbol, interval="1h", limit=100):
    try:
        r = requests.get(f"{BINANCE_FAPI}/fapi/v1/klines",
                         params={"symbol": symbol, "interval": interval, "limit": limit},
                         timeout=10)
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return pd.DataFrame()
        df = pd.DataFrame(data, columns=[
            "time","open","high","low","close","volume",
            "close_time","quote_vol","trades","taker_buy_base","taker_buy_quote","ignore"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=60)
def fetch_klines_fallback(symbol, interval="1h", limit=100):
    df = fetch_klines(symbol, interval, limit)
    if not df.empty:
        return df
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart",
            params={"vs_currency": "usd", "days": "4", "interval": "hourly"},
            timeout=15
        )
        mc = r.json()
        prices = pd.DataFrame(mc["prices"], columns=["ts", "close"])
        volumes = pd.DataFrame(mc["total_volumes"], columns=["ts", "volume"])
        df2 = prices.merge(volumes, on="ts")
        df2["time"] = pd.to_datetime(df2["ts"], unit="ms")
        df2 = df2.tail(limit).reset_index(drop=True)
        df2["open"] = df2["close"].shift(1).fillna(df2["close"])
        df2["high"] = df2[["open","close"]].max(axis=1) * (1 + np.random.uniform(0, 0.003, len(df2)))
        df2["low"] = df2[["open","close"]].min(axis=1) * (1 - np.random.uniform(0, 0.003, len(df2)))
        return df2
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_funding_rate(symbol):
    try:
        r = requests.get(f"{BINANCE_FAPI}/fapi/v1/fundingRate",
                         params={"symbol": symbol, "limit": 8}, timeout=10)
        if r.status_code != 200:
            return None, None
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return None, None
        rates = [float(x["fundingRate"]) * 100 for x in data]
        return rates[-1], sum(rates) / len(rates)
    except Exception:
        return None, None

@st.cache_data(ttl=300)
def fetch_open_interest(symbol):
    try:
        r = requests.get(f"{BINANCE_FAPI}/fapi/v1/openInterest",
                         params={"symbol": symbol}, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        return float(data.get("openInterest", 0))
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_oi_hist(symbol):
    try:
        r = requests.get(f"{BINANCE_DAPI}/openInterestHist",
                         params={"symbol": symbol, "period": "1h", "limit": 10}, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            return None
        vals = [float(x["sumOpenInterestValue"]) for x in data]
        chg = (vals[-1] - vals[-2]) / vals[-2] * 100 if vals[-2] else 0
        return chg
    except Exception:
        return None

@st.cache_data(ttl=300)
def fetch_longshortratio(symbol):
    try:
        r = requests.get(f"{BINANCE_DAPI}/topLongShortPositionRatio",
                         params={"symbol": symbol, "period": "1h", "limit": 5}, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return None
        return float(data[-1]["longAccount"])
    except Exception:
        return None

@st.cache_data(ttl=1800)
def fetch_fear_greed():
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1&format=json", timeout=10)
        if r.status_code != 200:
            return None, "N/A"
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
    df["ema200"] = df["close"].ewm(span=200).mean()
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
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df

def detect_candle_patterns(df):
    patterns = []
    if len(df) < 3:
        return patterns
    c = df.iloc[-1]
    p = df.iloc[-2]
    pp = df.iloc[-3]
    body = abs(c["close"] - c["open"])
    upper_wick = c["high"] - max(c["close"], c["open"])
    lower_wick = min(c["close"], c["open"]) - c["low"]
    total_range = c["high"] - c["low"]
    if total_range == 0:
        return patterns
    if lower_wick > body * 2 and upper_wick < body * 0.5:
        patterns.append(("🔨 망치형", "LONG"))
    if upper_wick > body * 2 and lower_wick < body * 0.5:
        patterns.append(("🌠 슈팅스타", "SHORT"))
    if body < total_range * 0.1:
        patterns.append(("➕ 도지", "WAIT"))
    if (c["close"] > c["open"] and p["close"] < p["open"] and
            c["close"] > p["open"] and c["open"] < p["close"]):
        patterns.append(("🟢 강세 장악형", "LONG"))
    if (c["close"] < c["open"] and p["close"] > p["open"] and
            c["close"] < p["open"] and c["open"] > p["close"]):
        patterns.append(("🔴 약세 장악형", "SHORT"))
    if (pp["close"] > pp["open"] and p["close"] > p["open"] and
            c["close"] > c["open"] and c["close"] > p["close"] > pp["close"]):
        patterns.append(("🕯️ 쓰리 화이트 솔져", "LONG"))
    if (pp["close"] < pp["open"] and p["close"] < p["open"] and
            c["close"] < c["open"] and c["close"] < p["close"] < pp["close"]):
        patterns.append(("🕯️ 쓰리 블랙 크로우", "SHORT"))
    if lower_wick > total_range * 0.6:
        patterns.append(("📌 핀바 (아래꼬리)", "LONG"))
    if upper_wick > total_range * 0.6:
        patterns.append(("📌 핀바 (위꼬리)", "SHORT"))
    return patterns

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
    return sorted(set(sup_lvls))[:3], sorted(set(res_lvls), reverse=True)[:3]

def get_signal(df, sup_lvls, res_lvls, fr_now, fr_avg, oi_chg, ls_ratio, fng_value, patterns):
    cp = df["close"].iloc[-1]
    rsi = df["rsi"].iloc[-1]
    macd = df["macd"].iloc[-1]
    macd_sig = df["macd_signal"].iloc[-1]
    bb_low = df["bb_lower"].iloc[-1]
    bb_high = df["bb_upper"].iloc[-1]
    ema20 = df["ema20"].iloc[-1]
    ema50 = df["ema50"].iloc[-1]
    ema200 = df["ema200"].iloc[-1]

    ls, ss = 0, 0
    lr, sr = [], []

    if rsi < 40:
        ls += 2; lr.append(f"RSI 과매도 ({rsi:.1f})")
    elif rsi > 60:
        ss += 2; sr.append(f"RSI 과매수 ({rsi:.1f})")

    if macd > macd_sig:
        ls += 1; lr.append("MACD 상향")
    else:
        ss += 1; sr.append("MACD 하향")

    if cp < bb_low * 1.005:
        ls += 2; lr.append("BB 하단 근접")
    elif cp > bb_high * 0.995:
        ss += 2; sr.append("BB 상단 근접")

    if ema20 > ema50:
        ls += 1; lr.append("EMA20 > EMA50")
    else:
        ss += 1; sr.append("EMA20 < EMA50")

    if cp > ema200:
        ls += 1; lr.append("EMA200 위 (상승구조)")
    else:
        ss += 1; sr.append("EMA200 아래 (하락구조)")

    if sup_lvls and cp < min(sup_lvls) * 1.012:
        ls += 2; lr.append("지지선 근처")
    if res_lvls and cp > max(res_lvls) * 0.988:
        ss += 2; sr.append("저항선 근처")

    if fr_now is not None:
        if fr_now > 0.01:
            ss += 2; sr.append(f"펀딩비 과열 (+{fr_now:.4f}%)")
        elif fr_now < -0.01:
            ls += 2; lr.append(f"펀딩비 과냉 ({fr_now:.4f}%)")

    if oi_chg is not None:
        if oi_chg > 0.5 and cp > df["close"].iloc[-2]:
            ls += 1; lr.append(f"OI 증가 +{oi_chg:.2f}%")
        elif oi_chg > 0.5 and cp < df["close"].iloc[-2]:
            ss += 1; sr.append(f"OI 증가 하락동반 +{oi_chg:.2f}%")

    if ls_ratio is not None:
        if ls_ratio < 0.4:
            ls += 1; lr.append(f"롱숏 비율 숏 우세 ({ls_ratio:.2f})")
        elif ls_ratio > 0.6:
            ss += 1; sr.append(f"롱숏 비율 롱 과열 ({ls_ratio:.2f})")

    if fng_value is not None:
        if fng_value <= 25:
            ls += 2; lr.append(f"공포탐욕 극단공포 ({fng_value})")
        elif fng_value >= 75:
            ss += 2; sr.append(f"공포탐욕 극단탐욕 ({fng_value})")

    for name, direction in patterns:
        if direction == "LONG":
            ls += 1; lr.append(f"캔들패턴: {name}")
        elif direction == "SHORT":
            ss += 1; sr.append(f"캔들패턴: {name}")

    total = ls + ss
    liq_long = cp * (1 - 1/15.5)
    liq_short = cp * (1 + 1/14.5)

    if ls > ss:
        sig = "🟢 롱 우세"
    elif ss > ls:
        sig = "🔴 숏 우세"
    else:
        sig = "⚪ 중립"

    return {
        "cp": cp, "rsi": rsi, "ls": ls, "ss": ss, "total": total,
        "lr": lr, "sr": sr, "liq_long": liq_long, "liq_short": liq_short,
        "signal": sig, "atr": df["atr"].iloc[-1]
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

# ── UI ───────────────────────────────────────────────────────────
st.title("🚀 BTC 레버리지 대시보드 v2")
st.caption("Binance 공개 API | 캔들패턴 + 펀딩비 + OI + 롱숏비율 + 공포탐욕")

col_top1, col_top2 = st.columns([4, 1])
with col_top1:
    symbol = st.selectbox("심볼 선택", SYMBOLS, index=0)
with col_top2:
    if st.button("🔄 새로고침"):
        st.cache_data.clear()
        st.rerun()

try:
    df_1h = add_indicators(fetch_klines_fallback(symbol, "1h", 100))
    df_15m = add_indicators(fetch_klines_fallback(symbol, "15m", 100))

    if df_1h.empty:
        st.error("가격 데이터를 불러올 수 없습니다. 잠시 후 새로고침하세요.")
        st.stop()

    fr_now, fr_avg = fetch_funding_rate(symbol)
    oi_now = fetch_open_interest(symbol)
    oi_chg = fetch_oi_hist(symbol)
    ls_ratio = fetch_longshortratio(symbol)
    fng_value, fng_class = fetch_fear_greed()
    sup_lvls, res_lvls = find_levels(df_1h)
    patterns = detect_candle_patterns(df_1h)
    sig = get_signal(df_1h, sup_lvls, res_lvls, fr_now, fr_avg,
                     oi_chg, ls_ratio, fng_value, patterns)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("현재가", f"${sig['cp']:,.0f}")
    c2.metric("RSI (1H)", f"{sig['rsi']:.1f}")
    c3.metric("🔴 롱 청산가", f"${sig['liq_long']:,.0f}")
    c4.metric("🟢 숏 청산가", f"${sig['liq_short']:,.0f}")
    c5.metric("공포탐욕", f"{fng_value if fng_value else 'N/A'}", delta=fng_class)
    c6.metric("ATR", f"${sig['atr']:,.0f}" if sig['atr'] else "N/A")

    sig_color = "green" if sig["ls"] > sig["ss"] else ("red" if sig["ss"] > sig["ls"] else "orange")
    st.markdown(
        f"<h2 style='text-align:center;color:{sig_color}'>"
        f"{sig['signal']} &nbsp;&nbsp; 롱 {sig['ls']}pt &nbsp;|&nbsp; 숏 {sig['ss']}pt</h2>",
        unsafe_allow_html=True
    )

    if patterns:
        pattern_txt = " &nbsp;|&nbsp; ".join([f"{n}" for n, d in patterns])
        st.markdown(f"<div style='text-align:center;font-size:16px'>캔들 패턴: {pattern_txt}</div>",
                    unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 차트 (1H)", "📊 차트 (15M)", "🧮 TP/SL 계산기"])

    def draw_chart(df, title):
        plot_df = df.tail(60)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.15, 0.2, 0.15],
                            vertical_spacing=0.04,
                            subplot_titles=[title, "RSI", "MACD", "거래량"])

        fig.add_trace(go.Candlestick(
            x=plot_df["time"], open=plot_df["open"], high=plot_df["high"],
            low=plot_df["low"], close=plot_df["close"],
            increasing_line_color="#00c878", decreasing_line_color="#ff4455",
            showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["ema20"],
            line=dict(color="#58a6ff", width=2), name="EMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["ema50"],
            line=dict(color="#ffd60a", width=2), name="EMA50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["ema200"],
            line=dict(color="#ff9944", width=1.5, dash="dot"), name="EMA200"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["bb_upper"],
            line=dict(color="rgba(100,100,200,0.5)", width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["bb_lower"],
            line=dict(color="rgba(100,100,200,0.5)", width=1),
            fill="tonexty", fillcolor="rgba(100,100,200,0.06)", showlegend=False), row=1, col=1)

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
        fig.add_hline(y=70, line=dict(color="#ff4455", dash="dot"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="#00c878", dash="dot"), row=2, col=1)

        hist_colors = ["#00c878" if v >= 0 else "#ff4455" for v in plot_df["macd_hist"]]
        fig.add_trace(go.Bar(x=plot_df["time"], y=plot_df["macd_hist"],
            marker_color=hist_colors, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["macd"],
            line=dict(color="#58a6ff", width=2), showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["macd_signal"],
            line=dict(color="#ffd60a", width=1.5, dash="dot"), showlegend=False), row=3, col=1)

        vol_colors = ["#00c878" if plot_df["close"].iloc[i] >= plot_df["open"].iloc[i]
                      else "#ff4455" for i in range(len(plot_df))]
        fig.add_trace(go.Bar(x=plot_df["time"], y=plot_df["volume"],
            marker_color=vol_colors, showlegend=False), row=4, col=1)

        fig.update_layout(
            height=900, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font=dict(color="#e6edf3"), xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10), legend_orientation="h"
        )
        for i in range(1, 5):
            fig.update_xaxes(gridcolor="#21262d", row=i, col=1)
            fig.update_yaxes(gridcolor="#21262d", row=i, col=1)
        fig.update_yaxes(title_text="가격", tickformat="$,.0f", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="거래량", row=4, col=1)
        return fig

    with tab1:
        col_chart, col_info = st.columns([3, 1])
        with col_chart:
            st.plotly_chart(draw_chart(df_1h, f"{symbol} 1H"), use_container_width=True)
        with col_info:
            st.subheader("📋 롱 신호")
            for item in sig["lr"]:
                st.write(f"✅ {item}")
            st.subheader("📋 숏 신호")
            for item in sig["sr"]:
                st.write(f"❌ {item}")
            st.divider()
            st.subheader("📌 파생지표")
            st.write(f"펀딩비: {f'{fr_now:.4f}%' if fr_now is not None else 'N/A'}")
            st.write(f"펀딩 평균: {f'{fr_avg:.4f}%' if fr_avg is not None else 'N/A'}")
            st.write(f"OI 변화율: {f'{oi_chg:.2f}%' if oi_chg is not None else 'N/A'}")
            st.write(f"롱 비율: {f'{ls_ratio:.2f}' if ls_ratio is not None else 'N/A'}")
            st.divider()
            st.subheader("📍 지지/저항")
            for r in res_lvls:
                st.write(f"🔴 저항 ${r:,.0f}")
            for s in sup_lvls:
                st.write(f"🟢 지지 ${s:,.0f}")
            if patterns:
                st.divider()
                st.subheader("🕯️ 캔들 패턴")
                for name, direction in patterns:
                    color = "green" if direction == "LONG" else ("red" if direction == "SHORT" else "gray")
                    st.markdown(f"<span style='color:{color}'>{name} → {direction}</span>",
                                unsafe_allow_html=True)

    with tab2:
        sig_15m = get_signal(df_15m, sup_lvls, res_lvls, fr_now, fr_avg,
                             oi_chg, ls_ratio, fng_value, detect_candle_patterns(df_15m))
        st.markdown(
            f"<h3 style='color:{'green' if sig_15m['ls'] > sig_15m['ss'] else 'red'}'>"
            f"15M: {sig_15m['signal']} — 롱 {sig_15m['ls']}pt | 숏 {sig_15m['ss']}pt</h3>",
            unsafe_allow_html=True
        )
        st.plotly_chart(draw_chart(df_15m, f"{symbol} 15M"), use_container_width=True)

    with tab3:
        st.subheader("🧮 TP / SL / 청산가 계산기")
        t1, t2 = st.columns(2)
        with t1:
            side = st.selectbox("방향", ["LONG", "SHORT"])
            entry = st.number_input("진입가 ($)", value=float(sig["cp"]), step=100.0)
            lev = st.slider("레버리지", 1, 50, 15)
        with t2:
            sl_pct = st.number_input("손절 %", value=3.5, step=0.1)
            tp_pct = st.number_input("익절 %", value=8.0, step=0.1)
            capital = st.number_input("계좌 잔고 ($)", value=1000.0, step=100.0)
            risk_pct = st.number_input("허용 손실 %", value=2.0, step=0.1)

        sl_price, tp_price, liq_price = calc_tpsl(entry, side, lev, sl_pct, tp_pct)
        position_size = (capital * risk_pct / 100) / (sl_pct / 100)
        margin = position_size / lev

        r1, r2, r3 = st.columns(3)
        r1.metric("✂️ 손절가", f"${sl_price:,.0f}")
        r2.metric("🎯 익절가", f"${tp_price:,.0f}")
        r3.metric("💥 청산가", f"${liq_price:,.0f}")

        r4, r5, r6 = st.columns(3)
        r4.metric("포지션 크기", f"${position_size:,.0f}")
        r5.metric("필요 증거금", f"${margin:,.0f}")
        r6.metric("RR 비율", f"1 : {tp_pct/sl_pct:.1f}")

        atr_val = sig["atr"]
        if atr_val and atr_val > 0:
            st.info(f"💡 ATR 기반 권장 손절: ${entry - atr_val * 1.5:,.0f} (롱 기준) | "
                    f"손절거리: ${atr_val * 1.5:,.0f} ({atr_val*1.5/entry*100:.2f}%)")

except Exception as e:
    st.error(f"오류 발생: {e}")
    st.info("새로고침 버튼을 눌러주세요.")
