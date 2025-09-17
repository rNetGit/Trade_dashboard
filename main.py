
import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from ta_utils import (calculate_rsi, calculate_ema, calculate_macd, calculate_vwap, calculate_atr, fused_levels, build_trade_plan)

st.set_page_config(page_title="SPX Dashboard", page_icon="📈", layout="wide")
st.title("SPX Technical Dashboard — Enhanced")

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df_ = pd.read_csv(path)
    df_.columns = [c.strip().lower() for c in df_.columns]
    if "date" not in df_.columns: raise ValueError("CSV must include a 'date' column")
    df_["date"] = pd.to_datetime(df_["date"])
    df_ = df_.sort_values("date").reset_index(drop=True)
    for c in ["open","high","low","close"]:
        if c not in df_.columns: df_[c] = df_["close"]
    if "volume" not in df_.columns: df_["volume"] = np.nan
    return df_

# === Load data ===
df = load_data("spx_data.csv")

# === Compute indicators ===
df["rsi"] = calculate_rsi(df["close"])
df["ema20"] = calculate_ema(df["close"], 20)
df["ema50"] = calculate_ema(df["close"], 50)
df["vwap"]  = calculate_vwap(df)
macd, macd_sig = calculate_macd(df["close"])
df["macd"], df["macd_signal"] = macd, macd_sig
df["atr"]  = calculate_atr(df)

# === Build trade plan & levels (now df exists) ===
levels = fused_levels(df)
plan = build_trade_plan(df)
latest = df.iloc[-1]

# === Compact trade plan summary just under title ===
if plan['entry'] is not None:
    st.markdown(f"**Entry:** {plan['entry']:.2f}  ·  **Stop:** {plan['stop']:.2f}  ·  **Targets:** {plan['targets'][0]:.2f} / {plan['targets'][1]:.2f}")
else:
    st.markdown("**Entry:** —  ·  **Stop:** —  ·  **Targets:** range play")

# === Layout ===
col1, col2, col3 = st.columns([1,2,1])

with col1:
    st.subheader("Market Bias & Plan")
    st.markdown(f"**Bias:** {plan['bias']}")
    st.markdown(f"**Strategy:** {plan['strategy']}")
    st.markdown(f"**ATR(14):** {plan['atr']:.2f}")
    if plan["entry"] is not None:
        st.markdown(f"**Entry:** {plan['entry']:.2f}")
        st.markdown(f"**Stop:** {plan['stop']:.2f}")
        st.markdown(f"**Targets:** {plan['targets'][0]:.2f} / {plan['targets'][1]:.2f}")
    else:
        st.info("Range detected — fade extremes or use Iron Condor.")

    st.subheader("Support (score)")
    if levels["support"]:
        for lv, sc in levels["support"]:
            st.write(f"{lv:.2f}  (score {sc:.1f})")
    else:
        st.write("—")

    st.subheader("Resistance (score)")
    if levels["resistance"]:
        for lv, sc in levels["resistance"]:
            st.write(f"{lv:.2f}  (score {sc:.1f})")
    else:
        st.write("—")

with col2:
    st.subheader("Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ema20"], name="EMA20", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ema50"], name="EMA50", line=dict(dash="dot")))
    if not df["vwap"].isna().all():
        fig.add_trace(go.Scatter(x=df["date"], y=df["vwap"], name="VWAP", line=dict(dash="dash")))

    for lv,_ in levels["support"][:3]:
        fig.add_hline(y=float(lv), line_color="#17c964", opacity=0.6)
    for lv,_ in levels["resistance"][:3]:
        fig.add_hline(y=float(lv), line_color="#f31260", opacity=0.6)

    fig.update_layout(height=520, margin=dict(l=10,r=10,t=40,b=10), plot_bgcolor="#121620", paper_bgcolor="#121620", font=dict(color="#e8eef6"))
    st.plotly_chart(fig, use_container_width=True)

with col3:
    st.subheader("Latest Indicators")
    st.write(f"RSI(14): {float(latest['rsi']):.1f}")
    st.write(f"MACD: {float(latest['macd']):.2f} vs {float(latest['macd_signal']):.2f}")
    st.write(f"EMA20/50: {float(latest['ema20']):.2f} / {float(latest['ema50']):.2f}")
    vw = latest['vwap']
    st.write(f"VWAP: {'—' if np.isnan(vw) else f'{float(vw):.2f}'}")
    st.write(f"ATR(14): {float(latest['atr']):.2f}")
