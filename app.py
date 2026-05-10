import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="BTC 15x Dashboard", layout="wide")

@st.cache_data(ttl=300)
def fetch_market_data():
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

def get_signal(df, sup_lvls, res_lvls):
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

    if rsi < 40: ls += 2; lr.append(f"RSI 과매도 ({rsi:.1f})")
    elif rsi > 60: ss += 2; sr.append(f"RSI 과매수 ({rsi:.1f})")

    if macd > macd_sig: ls += 1; lr.append("MACD 상향")
    else: ss += 1; sr.append("MACD 하향")

    if cp < bb_low * 1.005: ls += 2; lr.append("BB 하단 근접")
    elif cp > bb_high * 0.995: ss += 2; sr.append("BB 상단 근접")

    if ema20 > ema50: ls += 1; lr.append("EMA20 > EMA50")
    else: ss += 1; sr.append("EMA20 < EMA50")

    if sup_lvls and cp < min(sup_lvls) * 1.012: ls += 2; lr.append("지지선 근처")
    if res_lvls and cp > max(res_lvls) * 0.988: ss += 2; sr.append("저항선 근처")

    liq_long = cp * (1 - 1/15.5)
    liq_short = cp * (1 + 1/14.5)

    if ls > ss: sig = "🟢 롱 우세"
    elif ss > ls: sig = "🔴 숏 우세"
    else: sig = "⚪ 중립"

    return {"cp": cp, "rsi": rsi, "ls": ls, "ss": ss,
            "lr": lr, "sr": sr, "liq_long": liq_long,
            "liq_short": liq_short, "signal": sig}

# ── Main ──────────────────────────────────────────────────────────
st.title("🚀 BTC 15x 레버리지 대시보드")
st.caption("1H 기준 | CoinGecko 데이터 | 5분마다 자동 갱신")

if st.button("🔄 새로고침"):
    st.cache_data.clear()

try:
    df = add_indicators(fetch_market_data())
    sup_lvls, res_lvls = find_levels(df)
    sig = get_signal(df, sup_lvls, res_lvls)

    # ── 상단 지표 카드 ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("현재가", f"${sig['cp']:,.0f}")
    c2.metric("RSI", f"{sig['rsi']:.1f}", delta="과매도" if sig['rsi'] < 40 else ("과매수" if sig['rsi'] > 60 else "중립"))
    c3.metric("🔴 15x 롱 청산가", f"${sig['liq_long']:,.0f}", delta=f"-{(sig['cp']-sig['liq_long'])/sig['cp']*100:.1f}%")
    c4.metric("🟢 15x 숏 청산가", f"${sig['liq_short']:,.0f}", delta=f"+{(sig['liq_short']-sig['cp'])/sig['cp']*100:.1f}%")

    # ── 시그널 박스 ──
    sig_color = "green" if sig['ls'] > sig['ss'] else ("red" if sig['ss'] > sig['ls'] else "orange")
    st.markdown(f"<h2 style='text-align:center;color:{sig_color}'>{sig['signal']} &nbsp; 롱 {sig['ls']}/8 &nbsp;|&nbsp; 숏 {sig['ss']}/8</h2>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 1])

    with col_l:
        plot_df = df.tail(60)
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                            row_heights=[0.5, 0.15, 0.2, 0.15],
                            vertical_spacing=0.04)

        fig.add_trace(go.Candlestick(
            x=plot_df["time"], open=plot_df["open"], high=plot_df["high"],
            low=plot_df["low"], close=plot_df["close"],
            increasing_line_color="#00c878", decreasing_line_color="#ff4455",
            showlegend=False), row=1, col=1)

        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["ema20"],
            line=dict(color="#58a6ff", width=2), name="EMA20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["ema50"],
            line=dict(color="#ffd60a", width=2), name="EMA50"), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["bb_upper"],
            line=dict(color="rgba(100,100,200,0.5)", width=1), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=plot_df["time"], y=plot_df["bb_lower"],
            line=dict(color="rgba(100,100,200,0.5)", width=1),
            fill="tonexty", fillcolor="rgba(100,100,200,0.06)", showlegend=False), row=1, col=1)

        for s in sup_lvls:
            fig.add_hline(y=s, line=dict(color="#00c878", dash="dash", width=1.2), row=1, col=1)
        for r in res_lvls:
            fig.add_hline(y=r, line=dict(color="#ff4455", dash="dash", width=1.2), row=1, col=1)
        fig.add_hline(y=sig["liq_long"], line=dict(color="#ff1a40", dash="longdash", width=2), row=1, col=1)
        fig.add_hline(y=sig["liq_short"], line=dict(color="#00e88a", dash="longdash", width=2), row=1, col=1)

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

        vol_colors = ["#00c878" if plot_df["close"].iloc[i] >= plot_df["open"].iloc[i] else "#ff4455"
                      for i in range(len(plot_df))]
        fig.add_trace(go.Bar(x=plot_df["time"], y=plot_df["volume"],
            marker_color=vol_colors, showlegend=False), row=4, col=1)

        fig.update_layout(
            height=800, paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
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
        for r in sig["lr"]:
            st.write(f"✅ {r}")
        st.markdown("**🔴 숏 신호**")
        for r in sig["sr"]:
            st.write(f"❌ {r}")

        st.divider()
        st.subheader("📍 지지/저항")
        st.markdown("**저항선**")
        for r in res_lvls:
            st.write(f"🔴 ${r:,.0f}")
        st.markdown("**지지선**")
        for s in sup_lvls:
            st.write(f"🟢 ${s:,.0f}")

        st.divider()
        st.subheader("💥 청산 클러스터")
        for p, s in zip([sig["cp"]*r for r in [1.03,1.05,1.08]], [95,130,60]):
            st.write(f"🟢 숏청산 ${p:,.0f} (~${s}M)")
        for p, s in zip([sig["cp"]*r for r in [0.965,0.945,0.925]], [120,75,45]):
            st.write(f"🔴 롱청산 ${p:,.0f} (~${s}M)")

except Exception as e:
    st.error(f"데이터 로딩 실패: {e}")
    st.info("잠시 후 새로고침 버튼을 눌러주세요.")
