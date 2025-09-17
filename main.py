import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ta_utils import (
    calculate_rsi, calculate_ema, calculate_macd, calculate_vwap, calculate_atr,
    fused_levels, build_trade_plan, resample_ohlcv,
    project_day_range, project_week_range
)

st.set_page_config(page_title="SPX 4H/1D", page_icon="📈", layout="wide")
st.title("SPX — Day Trading (4H) & Swing (1D)")

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce").dt.tz_convert(None)
    else:
        df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_convert(None)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            df[c] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = np.nan
    return df

def enrich(df):
    d = df.copy()
    d["rsi"] = calculate_rsi(d["close"])
    d["ema20"] = calculate_ema(d["close"], 20)
    d["ema50"] = calculate_ema(d["close"], 50)
    d["vwap"] = calculate_vwap(d)
    macd, sig = calculate_macd(d["close"])
    d["macd"], d["macd_signal"] = macd, sig
    d["atr"] = calculate_atr(d)
    return d

src = load_data("spx_data.csv")  # put your SPX CSV next to this script

# Build 4H and 1D frames
df4h = enrich(resample_ohlcv(src, "4H"))
df1d = enrich(resample_ohlcv(src, "1D"))

# Plans & projections
plan4 = build_trade_plan(df4h)
planD = build_trade_plan(df1d)
day_proj = project_day_range(df1d)
week_proj = project_week_range(df1d)

# Compact summaries
if plan4['entry'] is not None:
    st.markdown(f"**4H Day-Trade Plan** → Entry **{plan4['entry']:.2f}** · Stop **{plan4['stop']:.2f}** · Targets **{plan4['targets'][0]:.2f}/{plan4['targets'][1]:.2f}** · Bias **{plan4['bias']}**")
else:
    st.markdown("**4H Day-Trade Plan** → Range play (IC) · Bias **NEUTRAL**")
if planD['entry'] is not None:
    st.markdown(f"**1D Swing Plan** → Entry **{planD['entry']:.2f}** · Stop **{planD['stop']:.2f}** · Targets **{planD['targets'][0]:.2f}/{planD['targets'][1]:.2f}** · Bias **{planD['bias']}**")
else:
    st.markdown("**1D Swing Plan** → Range play (IC) · Bias **NEUTRAL**")

c1, c2 = st.columns(2)

def last_n_days(df, days=5):
    d = df.copy()
    d["day"] = pd.to_datetime(d["date"]).dt.date
    last_day = d["day"].iloc[-1]
    cutoff = pd.Timestamp(last_day) - pd.Timedelta(days=days-1)
    return d[pd.to_datetime(d["day"]) >= cutoff].copy()

def add_common_overlays(fig, d, lev, day_proj=None, week_proj=None):
    fig.add_trace(go.Candlestick(
        x=d["date"], open=d["open"], high=d["high"], low=d["low"], close=d["close"], name="Price"
    ))
    fig.add_trace(go.Scatter(x=d["date"], y=d["ema20"], name="EMA20", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=d["date"], y=d["ema50"], name="EMA50", line=dict(dash="dot")))
    if not d["vwap"].isna().all():
        fig.add_trace(go.Scatter(x=d["date"], y=d["vwap"], name="VWAP", line=dict(dash="dash")))
    for lv, _ in lev["support"][:3]:
        fig.add_hline(y=float(lv), line_color="#17c964", opacity=0.6)
    for lv, _ in lev["resistance"][:3]:
        fig.add_hline(y=float(lv), line_color="#f31260", opacity=0.6)
    if day_proj:
        fig.add_hrect(y0=day_proj["proj_lo"], y1=day_proj["proj_hi"],
                      line_width=0, fillcolor="rgba(44,164,234,0.10)")
    if week_proj:
        fig.add_hrect(y0=week_proj["proj_lo"], y1=week_proj["proj_hi"],
                      line_width=0, fillcolor="rgba(255,22,61,0.06)")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10),
                      plot_bgcolor="#121620", paper_bgcolor="#121620",
                      font=dict(color="#e8eef6"))

with c1:
    st.subheader("4H — Last 5 days")
    d = last_n_days(df4h, 5)
    lev = fused_levels(d)
    fig = go.Figure()
    add_common_overlays(fig, d, lev, day_proj=day_proj, week_proj=None)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("1D — Last 5 days")
    d = last_n_days(df1d, 5)
    lev = fused_levels(d)
    fig = go.Figure()
    add_common_overlays(fig, d, lev, day_proj=day_proj, week_proj=week_proj)
    st.plotly_chart(fig, use_container_width=True)
