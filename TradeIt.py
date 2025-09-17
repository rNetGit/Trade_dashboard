import os, glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ta_utils import (
    calculate_rsi, calculate_ema, calculate_macd, calculate_vwap, calculate_atr,
    fused_levels, build_trade_plan, resample_ohlcv,
    project_day_range, project_week_range
)

# ───────────────────────────── App chrome ─────────────────────────────
st.set_page_config(page_title="TradeIt", page_icon="📈", layout="wide")
st.title("✨ TradeIt — 4H & Daily Analyzer")

st.markdown("""
<style>
:root{
  --card-bg-1:#1c2030; --card-bg-2:#23283b; --ink:#e8eef6;
  --bull:#15be53; --bear:#ff163d; --neu:#eeb100; --accent:#2ca4ea;
}
html,body,.stApp{background:#0e1118;color:var(--ink)}
.card{
  background:linear-gradient(135deg,var(--card-bg-1) 55%,var(--card-bg-2) 100%);
  border-radius:18px; padding:14px 14px 10px; margin:10px 0;
  border:1px solid #2a2d3e; box-shadow:0 6px 16px rgba(0,0,0,.25);
}
.title{font-weight:800; letter-spacing:.2px; color:#2ca4ea; font-size:1.05rem}
.subtle{opacity:.9; font-size:.92rem}
.rule{height:1px; background:linear-gradient(90deg, transparent, #30344a, transparent); margin:8px 0 10px}
.badge{display:inline-block; padding:2px 9px; border-radius:999px; font-size:.78rem; font-weight:700}
.bull{background:var(--bull); color:#021}
.bear{background:var(--bear); color:#fee}
.neu{background:var(--neu); color:#210}
.chips{display:flex; gap:8px; flex-wrap:wrap; margin-top:6px}
.chip{
  background:#121620; border:1px solid #2c3145; border-radius:999px; padding:2px 10px;
  font-size:.78rem; color:#cfd6e4
}
.kv{display:grid; grid-template-columns:auto 1fr; gap:4px 8px; margin-top:6px}
.key{opacity:.85}
.value{font-weight:700}
small.dim{opacity:.7}
@media(max-width:820px){.stColumns{display:block!important}}
</style>
""", unsafe_allow_html=True)

# ───────────────────────────── Inputs ─────────────────────────────
folder = st.sidebar.text_input(
    "CSV folder", ".",
    help="One CSV per symbol: date, open, high, low, close, volume(optional)"
)

# ───────────────────────────── Helpers ─────────────────────────────
@st.cache_data(show_spinner=False)
def _load_and_normalize_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(
        df[df.columns[0]] if "date" not in df.columns else df["date"],
        utc=True, errors="coerce"
    ).dt.tz_convert(None)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = (df[c].astype(str)
                           .str.replace(",", "", regex=False)
                           .str.replace("$", "", regex=False))
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open","high","low","close"]] > 0).all(axis=1)]
    return df.reset_index(drop=True)

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["rsi"] = calculate_rsi(d["close"])
    d["ema20"] = calculate_ema(d["close"], 20)
    d["ema50"] = calculate_ema(d["close"], 50)
    d["vwap"] = calculate_vwap(d)
    macd, sig = calculate_macd(d["close"])
    d["macd"], d["macd_signal"] = macd, sig
    d["atr"] = calculate_atr(d)
    return d

def structure_badge(bias: str) -> str:
    return ('<span class="badge bull">PCS</span>' if bias == "BULLISH"
            else '<span class="badge bear">CCS</span>' if bias == "BEARISH"
            else '<span class="badge neu">IC</span>')

def _sr_top2(levels: dict) -> tuple[list[float], list[float]]:
    s = [float(x[0]) for x in levels.get("support", [])[:2]]
    r = [float(x[0]) for x in levels.get("resistance", [])[:2]]
    while len(s) < 2: s.append(np.nan)
    while len(r) < 2: r.append(np.nan)
    return s, r

def _fmt_sr_line(s: list[float], r: list[float]) -> str:
    to2 = lambda v: ("—" if np.isnan(v) else f"{v:.2f}")
    return f"S1 {to2(s[0])} · S2 {to2(s[1])}  |  R1 {to2(r[0])} · R2 {to2(r[1])}"

def last_n_days(df: pd.DataFrame, days: int = 30) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    cutoff = d["date"].max().normalize() - pd.Timedelta(days=days-1)
    return d[d["date"] >= cutoff].copy()

def add_common_overlays(fig, d, lev, day_proj=None, week_proj=None):
    # Price + MAs + VWAP
    fig.add_trace(go.Candlestick(
        x=d["date"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        name="Price",
        increasing_line_color="#17c964", decreasing_line_color="#f31260"
    ))
    fig.add_trace(go.Scatter(x=d["date"], y=d["ema20"], name="EMA20", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=d["date"], y=d["ema50"], name="EMA50", line=dict(dash="dot")))
    if not d["vwap"].isna().all():
        fig.add_trace(go.Scatter(x=d["date"], y=d["vwap"], name="VWAP", line=dict(dash="dash")))

    # S/R lines
    for lv, _ in lev.get("support", [])[:3]:
        fig.add_hline(y=float(lv), line_color="#17c964", opacity=0.6)
    for lv, _ in lev.get("resistance", [])[:3]:
        fig.add_hline(y=float(lv), line_color="#f31260", opacity=0.6)

    # S/R labels
    for idx, (lv, _w) in enumerate(lev.get("support", [])[:2], start=1):
        fig.add_annotation(x=d["date"].iloc[-1], y=float(lv),
                           text=f"S{idx}", showarrow=False,
                           font=dict(color="#17c964", size=10),
                           bgcolor="rgba(23,201,100,0.12)", bordercolor="#17c964",
                           borderwidth=1, xanchor="right", yanchor="bottom")
    for idx, (lv, _w) in enumerate(lev.get("resistance", [])[:2], start=1):
        fig.add_annotation(x=d["date"].iloc[-1], y=float(lv),
                           text=f"R{idx}", showarrow=False,
                           font=dict(color="#f31260", size=10),
                           bgcolor="rgba(243,18,96,0.10)", bordercolor="#f31260",
                           borderwidth=1, xanchor="right", yanchor="top")

    # Projection bands
    if day_proj:
        fig.add_hrect(y0=day_proj["proj_lo"], y1=day_proj["proj_hi"], line_width=0,
                      fillcolor="rgba(44,164,234,0.10)")
    if week_proj:
        fig.add_hrect(y0=week_proj["proj_lo"], y1=week_proj["proj_hi"], line_width=0,
                      fillcolor="rgba(255,22,61,0.06)")

    # Y zoom & theme
    ymin, ymax = d["low"].min(), d["high"].max()
    pad = (ymax - ymin) * 0.03 if ymax > ymin else max(1.0, ymin * 0.01)
    fig.update_yaxes(range=[ymin - pad, ymax + pad])
    fig.update_layout(height=520, margin=dict(l=10,r=10,t=40,b=10),
                      plot_bgcolor="#121620", paper_bgcolor="#121620",
                      font=dict(color="#e8eef6"))

# ───────────────────────────── Data ─────────────────────────────
files = sorted(glob.glob(os.path.join(folder, "*.csv")))
symbols = {}
for p in files:
    sym = os.path.splitext(os.path.basename(p))[0].upper()
    try:
        symbols[sym] = _load_and_normalize_csv(p)
    except Exception as e:
        st.error(f"Failed to load {p}: {e}")

if not symbols:
    st.warning("No CSVs found. Point 'CSV folder' to your data directory.")
    st.stop()

# ───────────────────────────── Modal (Streamlit dialog) ─────────────────────────────
# Support both new and old Streamlit versions.
def open_details_dialog(sym: str):
    if hasattr(st, "dialog"):
        @st.dialog(f"🔍 {sym} — Details", width="large")
        def _dlg():
            src = symbols[sym]
            df4h = enrich(resample_ohlcv(src, "4H"))
            dfd  = enrich(resample_ohlcv(src, "1D"))
            day_proj  = project_day_range(dfd)
            week_proj = project_week_range(dfd)

            tab4, tabD = st.tabs(["4H (Day Trading)", "1D (Swing)"])
            with tab4:
                d = last_n_days(df4h, 15)
                lev = fused_levels(d)
                fig = go.Figure()
                add_common_overlays(fig, d, lev, day_proj=day_proj)
                st.plotly_chart(fig, use_container_width=True)
            with tabD:
                d = last_n_days(dfd, 30)
                lev = fused_levels(d)
                fig = go.Figure()
                add_common_overlays(fig, d, lev, day_proj=day_proj, week_proj=week_proj)
                st.plotly_chart(fig, use_container_width=True)

            if st.button("✖ Close"):
                st.rerun()
        _dlg()
    else:
        @st.experimental_dialog(f"🔍 {sym} — Details")
        def _dlg_exp():
            src = symbols[sym]
            df4h = enrich(resample_ohlcv(src, "4H"))
            dfd  = enrich(resample_ohlcv(src, "1D"))
            day_proj  = project_day_range(dfd)
            week_proj = project_week_range(dfd)

            tab4, tabD = st.tabs(["4H (Day Trading)", "1D (Swing)"])
            with tab4:
                d = last_n_days(df4h, 15)
                lev = fused_levels(d)
                fig = go.Figure()
                add_common_overlays(fig, d, lev, day_proj=day_proj)
                st.plotly_chart(fig, use_container_width=True)
            with tabD:
                d = last_n_days(dfd, 30)
                lev = fused_levels(d)
                fig = go.Figure()
                add_common_overlays(fig, d, lev, day_proj=day_proj, week_proj=week_proj)
                st.plotly_chart(fig, use_container_width=True)

            if st.button("✖ Close"):
                st.rerun()
        _dlg_exp()

# ───────────────────────────── Overview cards ─────────────────────────────
st.subheader("Overview (4H Day-Trade & 1D Swing)")

cols_per_row = 3
syms = list(symbols.keys())
for i in range(0, len(syms), cols_per_row):
    row = st.columns(min(cols_per_row, len(syms) - i))
    for j, col in enumerate(row):
        sym = syms[i + j]
        src = symbols[sym]
        df4h = enrich(resample_ohlcv(src, "4H"))
        dfd  = enrich(resample_ohlcv(src, "1D"))

        # S/R chips
        lev4 = fused_levels(df4h.tail(120))
        levD = fused_levels(dfd.tail(180))
        s4, r4 = _sr_top2(lev4)
        sD, rD = _sr_top2(levD)

        last4, lastd = df4h.iloc[-1], dfd.iloc[-1]
        plan4, pland = build_trade_plan(df4h), build_trade_plan(dfd)

        with col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="title">{sym}</div>', unsafe_allow_html=True)

            # 4H
            st.write(
                f"<div class='subtle'>4H Close <b>{last4['close']:.2f}</b> · "
                f"RSI <b>{last4['rsi']:.1f}</b> · ATR <b>{last4['atr']:.2f}</b></div>",
                unsafe_allow_html=True
            )
            st.markdown('Day trade: ' + structure_badge(plan4["bias"]), unsafe_allow_html=True)
            st.markdown(
                f"<div class='chips'><span class='chip'>{_fmt_sr_line(s4, r4)}</span></div>",
                unsafe_allow_html=True
            )
            if plan4["entry"] is not None:
                st.markdown(
                    f"<div class='kv'>"
                    f"<div class='key'>Entry</div><div class='value'>{plan4['entry']:.2f}</div>"
                    f"<div class='key'>Stop</div><div class='value'>{plan4['stop']:.2f}</div>"
                    f"<div class='key'>Targets</div>"
                    f"<div class='value'>{plan4['targets'][0]:.2f}/{plan4['targets'][1]:.2f}</div>"
                    f"</div>", unsafe_allow_html=True
                )
            else:
                st.write("<small class='dim'>Range play (IC)</small>", unsafe_allow_html=True)

            st.write("<div class='rule'></div>", unsafe_allow_html=True)

            # 1D
            st.write(
                f"<div class='subtle'>1D Close <b>{lastd['close']:.2f}</b> · "
                f"RSI <b>{lastd['rsi']:.1f}</b> · ATR <b>{lastd['atr']:.2f}</b></div>",
                unsafe_allow_html=True
            )
            st.markdown('Swing: ' + structure_badge(pland["bias"]), unsafe_allow_html=True)
            st.markdown(
                f"<div class='chips'><span class='chip'>{_fmt_sr_line(sD, rD)}</span></div>",
                unsafe_allow_html=True
            )
            if pland["entry"] is not None:
                st.markdown(
                    f"<div class='kv'>"
                    f"<div class='key'>Entry</div><div class='value'>{pland['entry']:.2f}</div>"
                    f"<div class='key'>Stop</div><div class='value'>{pland['stop']:.2f}</div>"
                    f"<div class='key'>Targets</div>"
                    f"<div class='value'>{pland['targets'][0]:.2f}/{pland['targets'][1]:.2f}</div>"
                    f"</div>", unsafe_allow_html=True
                )
            else:
                st.write("<small class='dim'>Range play (IC)</small>", unsafe_allow_html=True)

            if st.button("View details", key=f"btn_{sym}"):
                open_details_dialog(sym)

            st.markdown('</div>', unsafe_allow_html=True)
