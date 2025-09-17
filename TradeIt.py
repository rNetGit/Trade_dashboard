
import os, glob
import numpy as np, pandas as pd, plotly.graph_objects as go, streamlit as st
from ta_utils import (calculate_rsi, calculate_ema, calculate_macd, calculate_vwap, calculate_atr, fused_levels, build_trade_plan)

st.set_page_config(page_title="TradeIt", page_icon="📈", layout="wide")
st.title("✨ TradeIt!!")

CSS = """
<style>
.card{background:linear-gradient(120deg,#212433 60%,#26293d 100%);border-radius:16px;padding:12px;margin:8px 0;border:1px solid #2a2d3e}
.title{font-weight:700;color:#2ca4ea}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:.8rem}
.bull{background:#15be53;color:#021}.bear{background:#ff163d;color:#fee}.neu{background:#eeb100;color:#210}
@media(max-width:820px){.stColumns{display:block!important}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

folder = st.sidebar.text_input("CSV folder", ".", help="One CSV per symbol: date, open, high, low, close, volume(optional)")

@st.cache_data(show_spinner=False)
def load_csv(p):
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns: df["date"] = pd.to_datetime(df.iloc[:,0])
    else: df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for c in ["open","high","low","close"]:
        if c not in df.columns: df[c] = df["close"]
    if "volume" not in df.columns: df["volume"] = np.nan
    df["rsi"] = calculate_rsi(df["close"])
    df["ema20"] = calculate_ema(df["close"], 20)
    df["ema50"] = calculate_ema(df["close"], 50)
    df["vwap"] = calculate_vwap(df)
    macd, sig = calculate_macd(df["close"]); df["macd"]=macd; df["macd_signal"]=sig
    df["atr"] = calculate_atr(df)
    return df

# Discover CSVs
files = sorted(glob.glob(os.path.join(folder, "*.csv")))
symbols = {}
for p in files:
    sym = os.path.splitext(os.path.basename(p))[0].upper()
    try:
        symbols[sym] = load_csv(p)
    except Exception:
        pass

if not symbols:
    st.warning("No CSVs found. Point 'CSV folder' to your data directory.")
    st.stop()

if "detail" not in st.session_state: st.session_state["detail"] = None
def open_details(s): st.session_state["detail"] = s
def back(): st.session_state["detail"] = None

def bias_row(row):
    score = 0
    if row["close"] > row["ema20"] > row["ema50"]: score += 2
    if row["rsi"] >= 55: score += 1
    if row["macd"] > row["macd_signal"] and row["macd"] > 0: score += 1
    if not np.isnan(row["vwap"]) and row["close"] > row["vwap"]: score += 1
    if score >= 3: return "BULLISH"
    if score <= 1: return "BEARISH"
    return "NEUTRAL"

if st.session_state["detail"] is None:
    st.subheader("Overview")
    syms = list(symbols.keys())
    cols_per_row = 4
    for i in range(0, len(syms), cols_per_row):
        row = st.columns(min(cols_per_row, len(syms) - i))
        for j, col in enumerate(row):
            sym = syms[i + j]; df = symbols[sym]; last = df.iloc[-1]
            bias = bias_row(last)
            # Derive options structure label
            label = 'PCS' if bias=='BULLISH' else ('CCS' if bias=='BEARISH' else 'IC')
            label_badge = f'<span class="badge bull">PCS</span>' if label=='PCS' else (f'<span class="badge bear">CCS</span>' if label=='CCS' else f'<span class="badge neu">IC</span>')
            plan = build_trade_plan(df)
            with col:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<div class="title">{sym}</div>', unsafe_allow_html=True)
                st.write(f"Close: **{last['close']:.2f}**  ·  RSI: **{last['rsi']:.1f}**  ·  ATR: **{last['atr']:.2f}**")
                vwap_txt = '—' if np.isnan(last['vwap']) else f"{last['vwap']:.2f}"
                st.write(f"EMA20/50: {last['ema20']:.2f} / {last['ema50']:.2f}  ·  VWAP: {vwap_txt}")
                # Show both bias and the options structure
                bias_badge = '<span class="badge bull">BULLISH</span>' if bias=="BULLISH" else ('<span class="badge bear">BEARISH</span>' if bias=="BEARISH" else '<span class="badge neu">NEUTRAL</span>')
                st.markdown(bias_badge + ' · ' + label_badge, unsafe_allow_html=True)
                # Entry/Stop/Targets on main page
                if plan['entry'] is not None:
                    st.write(f"Entry: **{plan['entry']:.2f}**  ·  Stop: **{plan['stop']:.2f}**  ·  Targets: **{plan['targets'][0]:.2f} / {plan['targets'][1]:.2f}**")
                else:
                    st.write("Entry: —  ·  Stop: —  ·  Targets: range play")
                st.button("View details", key=f"btn_{sym}", on_click=open_details, args=(sym,))
                st.markdown('</div>', unsafe_allow_html=True)
else:
    sym = st.session_state["detail"]; df = symbols[sym]; last = df.iloc[-1]
    st.header(f"🔍 Details — {sym}")
    plan = build_trade_plan(df); levels = plan["levels"]
    supports, resistances = levels["support"], levels["resistance"]

    c1, c2 = st.columns([1,2])
    with c1:
        st.subheader("Trade Plan")
        st.write(f"**Bias:** {plan['bias']}")
        st.write(f"**Strategy:** {plan['strategy']}")
        st.write(f"**ATR(14):** {plan['atr']:.2f}")
        if plan["entry"] is not None:
            st.write(f"**Entry:** {plan['entry']:.2f}")
            st.write(f"**Stop:** {plan['stop']:.2f}")
            st.write(f"**Targets:** {plan['targets'][0]:.2f} / {plan['targets'][1]:.2f}")
        else:
            st.info("Range detected — wait for touches; consider Iron Condor.")

        st.subheader("Support")
        if supports:
            for lv, sc in supports: st.write(f"{lv:.2f}  (score {sc:.1f})")
        else: st.write("—")

        st.subheader("Resistance")
        if resistances:
            for lv, sc in resistances: st.write(f"{lv:.2f}  (score {sc:.1f})")
        else: st.write("—")

        st.button("⬅ Back", on_click=back)

    with c2:
        st.subheader("Chart")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="Price"))
        fig.add_trace(go.Scatter(x=df["date"], y=df["ema20"], name="EMA20", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df["date"], y=df["ema50"], name="EMA50", line=dict(dash="dot")))
        if not df["vwap"].isna().all():
            fig.add_trace(go.Scatter(x=df["date"], y=df["vwap"], name="VWAP", line=dict(dash="dash")))
        for lv,_ in supports[:3]:
            fig.add_hline(y=float(lv), line_color="#17c964", opacity=0.6)
        for lv,_ in resistances[:3]:
            fig.add_hline(y=float(lv), line_color="#f31260", opacity=0.6)
        fig.update_layout(height=520, margin=dict(l=10,r=10,t=40,b=10), plot_bgcolor="#121620", paper_bgcolor="#121620", font=dict(color="#e8eef6"))
        st.plotly_chart(fig, use_container_width=True)
