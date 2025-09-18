import os, glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ───────────────────────────── Utilities ─────────────────────────────
def _to_index(series: pd.Series) -> pd.DatetimeIndex:
    s = pd.to_datetime(series, utc=True, errors="coerce")
    return pd.DatetimeIndex(s).tz_convert(None)

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df.copy()
    idx = _to_index(d["date"])
    mask = ~idx.isna()
    d = d.loc[mask].copy()
    idx = idx[mask]
    d.index = idx
    d = d.sort_index()
    for c in ["open","high","low","close"]:
        if c not in d.columns:
            d[c] = d["close"]
    o = d["open"].resample(rule).first()
    h = d["high"].resample(rule).max()
    l = d["low"].resample(rule).min()
    c = d["close"].resample(rule).last()
    if "volume" in d.columns:
        v = d["volume"].resample(rule).sum()
        out = pd.DataFrame({"date":o.index,"open":o.values,"high":h.values,"low":l.values,"close":c.values,"volume":v.values})
    else:
        out = pd.DataFrame({"date":o.index,"open":o.values,"high":h.values,"low":l.values,"close":c.values})
    return out.dropna().reset_index(drop=True)

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window).mean()
    roll_down = down.rolling(window).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def calculate_ema(prices: pd.Series, window: int) -> pd.Series:
    return prices.ewm(span=window, adjust=False).mean()

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns or df["volume"].isna().all():
        return pd.Series(np.nan, index=df.index)
    tp = df["close"]
    pv = tp * df["volume"]
    return pv.cumsum() / df["volume"].cumsum()

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))
    return tr.rolling(window).mean()

def calculate_bbands(prices: pd.Series, window: int = 20, mult: float = 2.0):
    ma = prices.rolling(window).mean()
    sd = prices.rolling(window).std(ddof=0)
    return ma + mult*sd, ma, ma - mult*sd

def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff(); down = -l.diff()
    plus_dm  = ((up > down) & (up > 0)).astype(float) * up
    minus_dm = ((down > up) & (down > 0)).astype(float) * down
    tr = (h.combine(c.shift(), max) - l.combine(c.shift(), min)).abs()
    tr = tr.rolling(window).sum()
    plus_di  = 100 * (plus_dm.rolling(window).sum() / tr)
    minus_di = 100 * (minus_dm.rolling(window).sum() / tr)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)) * 100
    return dx.rolling(window).mean()

def fused_levels(df: pd.DataFrame, lookbacks=(5,10,20)):
    price = float(df["close"].iloc[-1])
    levels = []
    for n in lookbacks:
        levels.append((float(df["high"].rolling(n).max().iloc[-1]), "R"))
        levels.append((float(df["low"].rolling(n).min().iloc[-1]), "S"))
    supports, resistances = [], []
    for lv, tag in levels:
        (supports if lv <= price else resistances).append((lv, 1))
    supports = sorted(supports, key=lambda x: (abs(price - x[0])))[:3]
    resistances = sorted(resistances, key=lambda x: (abs(x[0] - price)))[:3]
    return {"support": supports, "resistance": resistances}

def build_trade_plan(df: pd.DataFrame):
    price = float(df["close"].iloc[-1])
    atr = calculate_atr(df).iloc[-1]
    atr = float(atr) if not np.isnan(atr) else max(1.0, price*0.005)
    levels = fused_levels(df)
    e20, e50 = calculate_ema(df["close"],20), calculate_ema(df["close"],50)
    if price > e20.iloc[-1] > e50.iloc[-1]:
        bias = "BULLISH"
        res = levels["resistance"][0][0] if levels["resistance"] else price + 0.5*atr
        sup = levels["support"][0][0] if levels["support"] else price - 1.0*atr
        entry = max(price, res + 0.1*atr); stop = sup - 0.5*atr
        t1, t2 = entry + 1.0*atr, entry + 2.0*atr
    elif price < e20.iloc[-1] < e50.iloc[-1]:
        bias = "BEARISH"
        res = levels["resistance"][0][0] if levels["resistance"] else price + 1.0*atr
        sup = levels["support"][0][0] if levels["support"] else price - 0.5*atr
        entry = min(price, sup - 0.1*atr); stop = res + 0.5*atr
        t1, t2 = entry - 1.0*atr, entry - 2.0*atr
    else:
        bias = "NEUTRAL"
        if levels["resistance"] and levels["support"]:
            top = levels["resistance"][0][0]; bot = levels["support"][0][0]
        else:
            top, bot = price + 1.5*atr, price - 1.5*atr
        entry=None; stop=None; t1=(top+bot)/2; t2=top if abs(top-price)<abs(price-bot) else bot
    return {"bias":bias,"atr":atr,"levels":levels,"entry":entry,"stop":stop,"targets":[t1,t2]}

# ───────────────────────────── App chrome ─────────────────────────────
st.set_page_config(page_title="TradeIt Compact", page_icon="📈", layout="wide")
st.markdown("""
<style>
:root{--ink:#e8eef6;--muted:#a7b0c3;--bg:#0f121a;--panel:#121726;--accent:#2ca4ea;--bull:#17c964;--bear:#f31260;--neu:#eeb100;}
html,body,.stApp{background:var(--bg);color:var(--ink)}
.block-container{padding-top:.5rem;padding-bottom:1rem;}
header, .st-emotion-cache-18ni7ap{visibility:hidden;height:0;}
h1,h2,h3{margin:0 0 .25rem 0}
.toolbar{display:flex;gap:.75rem;flex-wrap:wrap;align-items:center;background:var(--panel);border:1px solid #252b3b;border-radius:14px;padding:.5rem .75rem;margin:.25rem 0 .5rem}
.badge{padding:.15rem .5rem;border-radius:999px;font-weight:700;font-size:.78rem}
.bull{background:var(--bull);color:#00250c}
.bear{background:var(--bear);color:#fff1f4}
.neu{background:var(--neu);color:#221b00}
.kv{display:flex;gap:10px;flex-wrap:wrap;font-size:.9rem}
.kv b{color:#fff}
.small{color:var(--muted);font-size:.85rem}
.legend {opacity:.85}
.topbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:.25rem 0 .25rem}
.topbar .label{font-size:.9rem;color:var(--muted)}
.selectbox{min-width:160px}
@media(max-width:880px){
  .toolbar{gap:.5rem;padding:.45rem .6rem}
}
</style>
""", unsafe_allow_html=True)

# Title row
col_title, col_right = st.columns([0.8,0.2])
with col_title:
    st.markdown("<h2>TradeIt — 1H / 4H / 1D</h2>", unsafe_allow_html=True)
with col_right:
    st.markdown("<div class='small' style='text-align:right;'>compact mode</div>", unsafe_allow_html=True)

# ───────────────────────────── Sidebar ─────────────────────────────
st.sidebar.markdown("### Data")
data_folder = st.sidebar.text_input("Folder with CSVs", value=".", help="Select folder with *.csv (5‑min preferred).")
show_bbands = st.sidebar.checkbox("Bollinger Bands", True)
show_adx    = st.sidebar.checkbox("ADX", True)
chart_h     = st.sidebar.slider("Chart height", 360, 620, 460, help="Lower for tighter one‑page layout.")
days_1h     = st.sidebar.slider("Days (1H)", 5, 90, 30)
days_4h     = st.sidebar.slider("Days (4H)", 5, 60, 15)
days_1d     = st.sidebar.slider("Days (1D)", 15, 365, 60)

# Discover CSVs
csv_paths = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
if not csv_paths:
    st.error("No CSVs found in folder."); st.stop()

def _sym(p): return os.path.splitext(os.path.basename(p))[0].upper()
opts = [_sym(p) for p in csv_paths]

# Choose SPX by default if present; store ONLY in session_state and do NOT pass index to widget
def _default_option(options):
    if "SPX" in options:
        return "SPX"
    for opt in options:
        if "SPX" in opt:
            return opt
    return options[0]

if "symbol" not in st.session_state or st.session_state.symbol not in opts:
    st.session_state.symbol = _default_option(opts)

# Top inline Symbol dropdown (no index passed -> avoids the warning)
st.markdown("<div class='topbar'><span class='label'>Symbol</span></div>", unsafe_allow_html=True)
st.selectbox("", opts, key="symbol", label_visibility="collapsed")
selected = st.session_state.symbol
path = csv_paths[opts.index(selected)]

# ───────────────────────────── Load ─────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(p: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    df["date"] = pd.to_datetime(df[df.columns[0]] if "date" not in df.columns else df["date"], utc=True, errors="coerce")
    df["date"] = pd.DatetimeIndex(df["date"]).tz_convert(None)
    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",","",regex=False).str.replace("$","",regex=False), errors="coerce")
    df = df.dropna(subset=["date","open","high","low","close"]).sort_values("date").reset_index(drop=True)
    return df

def enrich(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["rsi"] = calculate_rsi(d["close"])
    d["ema20"] = calculate_ema(d["close"],20)
    d["ema50"] = calculate_ema(d["close"],50)
    d["vwap"]  = calculate_vwap(d)
    d["atr"]   = calculate_atr(d)
    bb_u, bb_m, bb_l = calculate_bbands(d["close"],20,2.0)
    d["bb_upper"], d["bb_mid"], d["bb_lower"] = bb_u, bb_m, bb_l
    d["adx"] = calculate_adx(d)
    return d

def subset_days(d: pd.DataFrame, n:int) -> pd.DataFrame:
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"])
    cutoff = d["date"].max().normalize() - pd.Timedelta(days=n-1)
    return d[d["date"]>=cutoff]

src = load_csv(path)
d1h = enrich(resample_ohlcv(src,"1H"))
d4h = enrich(resample_ohlcv(src,"4H"))
d1d = enrich(resample_ohlcv(src,"1D"))

def sr_line(levels: dict) -> str:
    def to2(v): 
        try: return f"{float(v):.2f}"
        except: return "—"
    s = [to2(x[0]) for x in levels.get("support",[])[:2]]
    r = [to2(x[0]) for x in levels.get("resistance",[])[:2]]
    while len(s)<2: s.append("—")
    while len(r)<2: r.append("—")
    return f"S1 {s[0]} · S2 {s[1]} | R1 {r[0]} · R2 {r[1]}"

def bias_badge(bias: str) -> str:
    if bias=="BULLISH": return "<span class='badge bull'>PCS</span>"
    if bias=="BEARISH": return "<span class='badge bear'>CCS</span>"
    return "<span class='badge neu'>IC</span>"

def build_trade_plan_local(df: pd.DataFrame):
    return build_trade_plan(df)

def top_toolbar(d: pd.DataFrame, tf_name: str):
    plan = build_trade_plan_local(d)
    last = d.iloc[-1]
    levs = fused_levels(d)
    entry_txt = "" if plan["entry"] is None else f"<b>Entry</b> {plan['entry']:.2f} · <b>Stop</b> {plan['stop']:.2f} · <b>Targets</b> {plan['targets'][0]:.2f}/{plan['targets'][1]:.2f}"
    line = sr_line(levs)
    st.markdown(
        f"""<div class="toolbar">
            <div><b>{selected}</b> — {tf_name}</div>
            <div>{bias_badge(plan['bias'])}</div>
            <div class="kv"><span class="small">Close <b>{last['close']:.2f}</b></span>
                             <span class="small">RSI <b>{last['rsi']:.1f}</b></span>
                             <span class="small">ATR <b>{last['atr']:.2f}</b></span></div>
            <div class="small legend">{line}</div>
            <div class="small">{entry_txt}</div>
        </div>""", unsafe_allow_html=True
    )

def plot_panel(d: pd.DataFrame, tf_name: str):
    top_toolbar(d, tf_name)
    lev = fused_levels(d)
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=d["date"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        name="Price", increasing_line_color="#17c964", decreasing_line_color="#f31260"
    ))
    fig.add_trace(go.Scatter(x=d["date"], y=d["ema20"], name="EMA20", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=d["date"], y=d["ema50"], name="EMA50", line=dict(dash="dot")))
    if not d["vwap"].isna().all():
        fig.add_trace(go.Scatter(x=d["date"], y=d["vwap"], name="VWAP", line=dict(dash="dash")))
    for lv,_ in lev.get("support",[])[:3]: fig.add_hline(y=float(lv), line_color="#17c964", opacity=0.6)
    for lv,_ in lev.get("resistance",[])[:3]: fig.add_hline(y=float(lv), line_color="#f31260", opacity=0.6)
    for i,(lv,_) in enumerate(lev.get("support",[])[:2],1):
        fig.add_annotation(x=d["date"].iloc[0], y=float(lv), text=f"S{i}", showarrow=False,
                           font=dict(color="#17c964",size=10), bgcolor="rgba(23,201,100,0.12)",
                           bordercolor="#17c964", borderwidth=1, xanchor="left", yanchor="bottom")
    for i,(lv,_) in enumerate(lev.get("resistance",[])[:2],1):
        fig.add_annotation(x=d["date"].iloc[0], y=float(lv), text=f"R{i}", showarrow=False,
                           font=dict(color="#f31260",size=10), bgcolor="rgba(243,18,96,0.10)",
                           bordercolor="#f31260", borderwidth=1, xanchor="left", yanchor="top")
    if show_bbands:
        fig.add_trace(go.Scatter(x=d["date"], y=d["bb_upper"], name="BB Upper", line=dict(width=1), opacity=.7))
        fig.add_trace(go.Scatter(x=d["date"], y=d["bb_mid"],   name="BB Mid",   line=dict(width=1, dash="dot"), opacity=.6))
        fig.add_trace(go.Scatter(x=d["date"], y=d["bb_lower"], name="BB Lower", line=dict(width=1), opacity=.7))
    if show_adx and "adx" in d:
        adx_val = float(d["adx"].iloc[-1]) if not pd.isna(d["adx"].iloc[-1]) else None
        if adx_val is not None:
            fig.add_annotation(x=d["date"].iloc[-1], y=float(d["high"].tail(50).max()),
                               text=f"ADX {adx_val:.1f}", showarrow=False,
                               bgcolor="rgba(238,177,0,0.15)", bordercolor="#eeb100", font=dict(size=10),
                               xanchor="right", yanchor="bottom")
    ymin, ymax = float(d["low"].min()), float(d["high"].max())
    pad = (ymax-ymin)*0.03 if ymax>ymin else 1.0
    fig.update_yaxes(range=[ymin-pad, ymax+pad])
    fig.update_layout(height=chart_h, margin=dict(l=8,r=8,t=10,b=8),
                      plot_bgcolor="#121620", paper_bgcolor="#121620",
                      font=dict(color="#e8eef6"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

# ───────────────────────────── Tabs ─────────────────────────────
tab1, tab2, tab3 = st.tabs(["1H","4H","1D"])
with tab1:
    plot_panel(subset_days(d1h, days_1h), "1H")
with tab2:
    plot_panel(subset_days(d4h, days_4h), "4H")
with tab3:
    plot_panel(subset_days(d1d, days_1d), "1D")
