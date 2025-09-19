# iTrader_merged.py
# Streamlit dashboard for SPX/ES/etc technical analysis with credit-spread helpers.

import os, glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------------------------------------------------------------
# Streamlit config MUST be first before any st.* calls
# ------------------------------------------------------------
st.set_page_config(page_title="iTrader", page_icon="📈", layout="wide")

# =========================
# Core helpers & indicators
# =========================
def _to_naive_utc_index(series: pd.Series) -> pd.DatetimeIndex:
    s = pd.to_datetime(series, utc=True, errors="coerce")
    return pd.DatetimeIndex(s).tz_convert(None)

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df.copy()
    idx = _to_naive_utc_index(d["date"])
    d = d.loc[~idx.isna()].copy()
    d.index = idx[~idx.isna()]
    d = d.sort_index()

    for c in ["open","high","low","close"]:
        if c not in d.columns:
            d[c] = d["close"]

    o = d["open"].resample(rule).first()
    h = d["high"].resample(rule).max()
    l = d["low"].resample(rule).min()
    c = d["close"].resample(rule).last()
    out = pd.DataFrame({"date": o.index, "open": o.values, "high": h.values, "low": l.values, "close": c.values})
    if "volume" in d.columns:
        out["volume"] = d["volume"].resample(rule).sum().values
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

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    return macd, signal_line

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    if "volume" not in df.columns or df["volume"].isna().all():
        return pd.Series(np.nan, index=df.index if df.index.size else range(len(df)))
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
    return ma + mult * sd, ma, ma - mult * sd

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

# =========================
# S/R + trade plan + strikes
# =========================
def _pivot_points(df: pd.DataFrame):
    prev = df.shift(1)
    P = (prev["high"] + prev["low"] + prev["close"]) / 3.0
    R1 = 2 * P - prev["low"]; S1 = 2 * P - prev["high"]
    R2 = P + (prev["high"] - prev["low"]); S2 = P - (prev["high"] - prev["low"])
    return P, R1, S1, R2, S2

def _donchian(df: pd.DataFrame, n: int = 20):
    upper = df["high"].rolling(n).max()
    lower = df["low"].rolling(n).min()
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def _fractals(df: pd.DataFrame, left: int = 2, right: int = 2):
    h, l = df["high"], df["low"]
    sh = h[(h.shift(1).rolling(left).max() < h) & (h.shift(-1).rolling(right).max() < h)]
    sl = l[(l.shift(1).rolling(left).min() > l) & (l.shift(-1).rolling(right).min() > l)]
    return sh.dropna(), sl.dropna()

def _round_levels(price: float, steps=(5, 10, 25, 50, 100)):
    s = set()
    for step in steps:
        s.add(round(price / step) * step)
        s.add(round((price + step / 2) / step) * step)
        s.add(round((price - step / 2) / step) * step)
    return sorted(s)

def fused_levels(df: pd.DataFrame, lookbacks=(5, 10, 20)):
    price = float(df["close"].iloc[-1])
    levels = []
    for n in lookbacks:
        levels.append((float(df["high"].rolling(n).max().iloc[-1]), "R"))
        levels.append((float(df["low"].rolling(n).min().iloc[-1]), "S"))
    du, dl, _ = _donchian(df, n=max(lookbacks))
    if not np.isnan(du.iloc[-1]):
        levels.append((float(du.iloc[-1]), "R"))
        levels.append((float(dl.iloc[-1]), "S"))
    P, R1, S1, R2, S2 = _pivot_points(df)
    for v, t in [(R1.iloc[-1], "R"), (S1.iloc[-1], "S"), (R2.iloc[-1], "R"), (S2.iloc[-1], "S"), (P.iloc[-1], "M")]:
        if not np.isnan(v): levels.append((float(v), t))
    sh, sl = _fractals(df)
    levels += [(float(v), "R") for v in sh.tail(6).values]
    levels += [(float(v), "S") for v in sl.tail(6).values]

    tol = max(0.002 * price, df["close"].diff().abs().tail(50).mean() or 1.0)
    buckets = []
    for lv, tag in levels:
        placed = False
        for b in buckets:
            if abs(b["level"] - lv) <= tol:
                b["cnt"] += 1; b["tags"].add(tag)
                b["level"] = (b["level"] * b["w"] + lv) / (b["w"] + 1)
                b["w"] += 1; placed = True; break
        if not placed:
            buckets.append({"level": lv, "cnt": 1, "tags": set([tag]), "w": 1})

    supports, resistances = [], []
    for b in buckets:
        level = float(b["level"]); score = b["cnt"]
        if "S" in b["tags"] and "R" in b["tags"]: score += 1
        if any(abs(level - rn) <= tol for rn in _round_levels(price)): score += 0.5
        (supports if level <= price else resistances).append((level, score))

    supports = sorted(supports, key=lambda x: (abs(price - x[0]), -x[1]))[:5]
    resistances = sorted(resistances, key=lambda x: (abs(x[0] - price), -x[1]))[:5]
    return {"support": supports, "resistance": resistances}

def classify_trend(close: pd.Series, ema20: pd.Series, ema50: pd.Series) -> str:
    price = float(close.iloc[-1]); e20 = float(ema20.iloc[-1]); e50 = float(ema50.iloc[-1])
    if price > e20 > e50: return "up"
    if price < e20 < e50: return "down"
    return "sideways"

def build_trade_plan(df: pd.DataFrame):
    price = float(df["close"].iloc[-1])
    _atr = calculate_atr(df).iloc[-1]
    _atr = float(_atr) if not np.isnan(_atr) else max(1.0, price * 0.005)
    ema20 = calculate_ema(df["close"], 20); ema50 = calculate_ema(df["close"], 50)
    adx = calculate_adx(df)
    levels = fused_levels(df)
    trend = classify_trend(df["close"], ema20, ema50)
    strong = (adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0) > 18

    if trend == "up" and strong:
        bias, strat = "BULLISH", "Put Credit Spread (PCS)"
    elif trend == "down" and strong:
        bias, strat = "BEARISH", "Call Credit Spread (CCS)"
    else:
        bias, strat = "NEUTRAL", "Iron Condor (IC)"

    return {"bias": bias, "strategy": strat, "atr": _atr, "levels": levels,
            "adx": float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None}

def _round5(x: float) -> int:
    try: return int(round(x / 5.0) * 5)
    except Exception: return int(x)

def strike_suggestions(df: pd.DataFrame, plan: dict) -> dict:
    price = float(df["close"].iloc[-1]); atr = float(plan["atr"])
    levs = plan["levels"]
    vwap_last = float(df["vwap"].iloc[-1]) if "vwap" in df.columns and not pd.isna(df["vwap"].iloc[-1]) else price
    s1 = levs["support"][0][0] if levs["support"] else price - atr
    r1 = levs["resistance"][0][0] if levs["resistance"] else price + atr
    h5 = float(df["high"].tail(5).max()); l5 = float(df["low"].tail(5).min())
    floor = min(vwap_last, s1, l5); ceiling = max(vwap_last, r1, h5)
    pcs_short = _round5(floor - atr); pcs_long  = pcs_short - 5
    ccs_short = _round5(ceiling + atr); ccs_long  = ccs_short + 5
    put_ic  = _round5(min(pcs_short, price - atr)); call_ic = _round5(max(ccs_short, price + atr))
    return {"PCS": (pcs_short, pcs_long), "CCS": (ccs_short, ccs_long), "IC":  (put_ic, put_ic-5, call_ic, call_ic+5)}

# =========================
# Volume features & alerts
# =========================
def add_volume_features(d: pd.DataFrame, spike_z=2.0) -> pd.DataFrame:
    d = d.copy()
    if "volume" not in d.columns or d["volume"].isna().all():
        d["vol_state"] = "NA"; d["unusual"] = None; d["vol_z"] = np.nan; d["vol_ema"] = np.nan
        return d
    vol = d["volume"]
    vol_ema = vol.ewm(span=20, adjust=False).mean()
    vol_mean = vol.rolling(50).mean(); vol_std  = vol.rolling(50).std(ddof=0)
    vol_z = (vol - vol_mean) / vol_std
    state = np.where((vol_z >= spike_z) | (vol > vol_ema * 1.8), "Spiked",
             np.where((vol_z <= -1.0) | (vol < vol_ema * 0.6), "Low", "Normal"))

    rng = (d["high"] - d["low"]).fillna(0.0)
    atr_proxy = rng.rolling(14).mean()
    up_candle = d["close"] > d["open"]; dn_candle = d["close"] < d["open"]
    big_bar = rng > (0.6 * atr_proxy)
    unusual = np.where((state=="Spiked") & up_candle & big_bar, "Unusual Buy",
               np.where((state=="Spiked") & dn_candle & big_bar, "Unusual Sell", None))
    d["vol_ema"], d["vol_z"], d["vol_state"], d["unusual"] = vol_ema, vol_z, state, unusual
    return d

def tape_alert_text(d: pd.DataFrame, plan: dict) -> str | None:
    last = d.iloc[-1]
    if str(last.get("vol_state","")) == "Spiked" and last.get("unusual", None):
        rng = (last["high"] - last["low"])
        z = last["vol_z"] if not pd.isna(last["vol_z"]) else 0.0
        return f"{last['unusual']} · Spike Z≈{z:.1f} · Range {rng:.1f} · Bias {plan['bias']}"
    return None

# =========================
# Loading & enrichment
# =========================
@st.cache_data(show_spinner=False)
def load_csv(p: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]

    # detect datetime column
    date_col = None
    for cand in ["date", "datetime", "time", "timestamp"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        date_col = df.columns[0]

    df["date"] = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.tz_convert(None)

    for c in ["open","high","low","close","volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["date","open","high","low","close"]).reset_index(drop=True)

def enrich(df: pd.DataFrame, spike_z=2.0) -> pd.DataFrame:
    d = df.copy()
    d["rsi"] = calculate_rsi(d["close"])
    d["ema20"] = calculate_ema(d["close"],20)
    d["ema50"] = calculate_ema(d["close"],50)
    d["vwap"]  = calculate_vwap(d)
    d["atr"]   = calculate_atr(d)
    bb_u, bb_m, bb_l = calculate_bbands(d["close"],20,2.0)
    d["bb_upper"], d["bb_mid"], d["bb_lower"] = bb_u, bb_m, bb_l
    d["adx"] = calculate_adx(d)
    macd, macd_sig = calculate_macd(d["close"])
    d["macd"], d["macd_sig"] = macd, macd_sig
    d = add_volume_features(d, spike_z=spike_z)
    return d

# =========================
# Price Action vs opening 15m
# =========================
def opening_15_close(src_df: pd.DataFrame) -> float | None:
    if len(src_df) == 0: return None
    d15 = resample_ohlcv(src_df, "15T")
    d15["day"] = pd.to_datetime(d15["date"]).dt.date
    last_day = d15["day"].iloc[-1]
    first_bar = d15[d15["day"]==last_day].iloc[0] if (d15["day"]==last_day).any() else None
    return float(first_bar["close"]) if first_bar is not None else None

def price_action_tag(last_close: float, open15: float | None) -> str:
    if open15 is None or np.isnan(open15): return ""
    if last_close > open15 * 1.0005: status = "Above"
    elif last_close < open15 * 0.9995: status = "Below"
    else: status = "At"
    return f"Price Action: {status} (vs 15m open {open15:.2f})"

# =========================
# Plotting
# =========================
def make_chart(d: pd.DataFrame, title: str, height: int = 560,
               show_bbands: bool = True, show_vol: bool = True):
    d = d.copy()
    fig = go.Figure()

    # price
    fig.add_trace(go.Candlestick(
        x=d["date"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        increasing_line_color="#17c964", decreasing_line_color="#f31260", name="Price",
        yaxis="y1"
    ))
    fig.add_trace(go.Scatter(x=d["date"], y=d["ema20"], name="EMA20", line=dict(dash="dot"), yaxis="y1"))
    fig.add_trace(go.Scatter(x=d["date"], y=d["ema50"], name="EMA50", line=dict(dash="dot"), yaxis="y1"))
    if "vwap" in d and not d["vwap"].isna().all():
        fig.add_trace(go.Scatter(x=d["date"], y=d["vwap"], name="VWAP", line=dict(dash="dash"), yaxis="y1"))

    if show_bbands:
        fig.add_trace(go.Scatter(x=d["date"], y=d["bb_upper"], name="BB Upper", line=dict(width=1), opacity=.7, yaxis="y1"))
        fig.add_trace(go.Scatter(x=d["date"], y=d["bb_mid"],   name="BB Mid",   line=dict(width=1, dash="dot"), opacity=.6, yaxis="y1"))
        fig.add_trace(go.Scatter(x=d["date"], y=d["bb_lower"], name="BB Lower", line=dict(width=1), opacity=.7, yaxis="y1"))

    # volume pane
    if show_vol and "volume" in d.columns:
        vol_colors = np.where(d["close"]>=d["open"], "#17c964", "#f31260")
        fig.add_trace(go.Bar(x=d["date"], y=d["volume"], name="Volume", marker_color=vol_colors, opacity=0.7, yaxis="y2"))
        if "vol_state" in d.columns:
            mask = d["vol_state"]=="Spiked"
            fig.add_trace(go.Scatter(x=d["date"][mask], y=d["volume"][mask]*1.02,
                                     mode="markers", name="Vol Spike", yaxis="y2",
                                     marker=dict(size=7, symbol="triangle-up")))

    # layout
    ymin, ymax = float(d["low"].min()), float(d["high"].max())
    pad = (ymax-ymin)*0.03 if ymax>ymin else 1.0
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=8, r=8, t=40, b=8),
        plot_bgcolor="#121620", paper_bgcolor="#121620",
        font=dict(color="#e8eef6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="overlay",
        yaxis=dict(domain=[0.25,1.0], range=[ymin-pad, ymax+pad], title=None),
        yaxis2=dict(domain=[0.0,0.22], title=None)
    )
    return fig

# =========================
# Trade Levels (Entry/Stop/Targets)
# =========================
def compute_trade_levels(df: pd.DataFrame, plan: dict, rr: float = 1.5,
                         stop_mult: float = 0.8, t1_mult: float = 0.5,
                         t2_mult: float = 1.0, t3_mult: float = 1.5):
    """
    Computes simple ATR-based levels.
    - Direction chooses long for BULLISH, short for BEARISH; for NEUTRAL returns both.
    - Entry: last close
    - Stop: ATR * stop_mult away in opposite direction
    - Targets: ATR * [t1,t2,t3] in direction
    """
    last = df.iloc[-1]
    entry = float(last['close'])
    atr = float(plan['atr'])
    bias = plan['bias'] or 'NEUTRAL'

    def _dir_levels(direction: str):
        sign = 1 if direction == 'long' else -1
        stop = entry - sign * atr * stop_mult
        t1 = entry + sign * atr * t1_mult
        t2 = entry + sign * atr * t2_mult
        t3 = entry + sign * atr * t3_mult
        return {'direction': direction, 'entry': entry, 'stop': stop, 't1': t1, 't2': t2, 't3': t3}

    if bias == 'BULLISH':
        return {'primary': _dir_levels('long')}
    if bias == 'BEARISH':
        return {'primary': _dir_levels('short')}
    # Neutral → show both
    return {'long': _dir_levels('long'), 'short': _dir_levels('short')}

def add_levels_to_figure(fig: go.Figure, levels: dict, yaxis='y1'):
    """
    Draws horizontal lines for Entry/Stop/T1/T2/T3. No text annotation; we'll show
    levels inline above the chart for clarity.
    """
    def _draw_set(lv: dict, color='#31c48d'):
        fig.add_hline(y=lv['stop'], line=dict(width=1, dash='dot', color=color))
        fig.add_hline(y=lv['t1'],   line=dict(width=1, dash='solid', color=color))
        fig.add_hline(y=lv['t2'],   line=dict(width=1, dash='dash', color=color))
        fig.add_hline(y=lv['t3'],   line=dict(width=1, dash='dashdot', color=color))
        fig.add_hline(y=lv['entry'], line=dict(width=2, color=color))
    if not levels:
        return fig
    if 'primary' in levels:
        _draw_set(levels['primary'], color='#36d399')
    else:
        _draw_set(levels['long'], color='#36d399')
        _draw_set(levels['short'], color='#f87272')
    return fig
    if 'primary' in levels:
        _draw_set('Primary', levels['primary'], color='#36d399')
    else:
        _draw_set('Long', levels['long'], color='#36d399')
        _draw_set('Short', levels['short'], color='#f87272')
    return fig


# =========================
# Header formatting helpers
# =========================

def _fmt(v: float) -> str:
    # compact number formatting
    return f"{v:.1f}"

def header_levels_md(symbol: str, tf_label: str, lv_dict: dict) -> str:
    def block(label: str, lv: dict, cls: str) -> str:
        return (f"<span class='pill {cls}'>{label}</span> "
                f"E:{_fmt(lv['entry'])} · S:{_fmt(lv['stop'])} · "
                f"T1:{_fmt(lv['t1'])} · T2:{_fmt(lv['t2'])} · T3:{_fmt(lv['t3'])}")
    if not lv_dict: 
        return f"<h3>{symbol} — {tf_label}</h3>"
    if 'primary' in lv_dict:
        lbl = 'LONG' if lv_dict['primary']['direction']=='long' else 'SHORT'
        html = f"<h3>{symbol} — {tf_label} · " + block(lbl, lv_dict['primary'], 'ok') + "</h3>"
    else:
        html = (f"<h3>{symbol} — {tf_label} · " 
                + block('LONG', lv_dict['long'], 'ok') + " · "
                + block('SHORT', lv_dict['short'], 'warn') + "</h3>")
    style = """
    <style>
    h3 { margin: 0.2rem 0 0.6rem; }
    .pill { padding: 2px 8px; border-radius: 9999px; font-weight: 600; }
    .ok { background: rgba(54,211,153,0.15); border:1px solid rgba(54,211,153,0.4); }
    .warn { background: rgba(248,114,114,0.15); border:1px solid rgba(248,114,114,0.4); }
    h3, .pill { font-size: 1.1rem; }
    @media (max-width: 900px) { h3, .pill { font-size: 0.95rem; } }
    </style>
    """
    return style + html
# =========================
# Streamlit UI
# =========================
st.markdown("<h4>iTrader - Technical Analysis</h4>", unsafe_allow_html=True)

# Sidebar controls (except symbol)
data_folder = st.sidebar.text_input("Folder with CSVs", value=".", help="Folder with SPX.csv (+ ES1.csv optional)")
show_bbands = st.sidebar.checkbox("Bollinger Bands", True)
show_vol    = st.sidebar.checkbox("Volume panel", True)
show_levels = st.sidebar.checkbox("Show Entry/Stop/Targets", True)
stop_mult = st.sidebar.slider("Stop (×ATR)", 0.3, 2.0, 0.8, 0.05)
t1_mult = st.sidebar.slider("T1 (×ATR)", 0.3, 2.5, 0.5, 0.05)
t2_mult = st.sidebar.slider("T2 (×ATR)", 0.5, 3.0, 1.0, 0.05)
t3_mult = st.sidebar.slider("T3 (×ATR)", 0.5, 4.0, 1.5, 0.05)
spike_z     = st.sidebar.slider("Spike Z-score", 1.5, 4.0, 2.0, 0.1)
days_1h     = st.sidebar.slider("Days (1H)", 5, 90, 30)
days_4h     = st.sidebar.slider("Days (4H)", 5, 60, 15)
days_1d     = st.sidebar.slider("Days (1D)", 15, 365, 60)

# Discover CSVs
csv_paths = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
if not csv_paths:
    st.error("No CSVs found in folder"); st.stop()

def _sym(p): return os.path.splitext(os.path.basename(p))[0].upper()
opts = [_sym(p) for p in csv_paths]
default_sym = "SPX" if "SPX" in opts else opts[0]

# Main-area symbol dropdown (no session_state overwrite)
symbol = st.selectbox("Symbol", opts, index=opts.index(default_sym), key="symbol")

# Load selected
path = csv_paths[opts.index(symbol)]
src = load_csv(path)

# ES1 volume proxy for SPX
es1_path = None
for p in csv_paths:
    if _sym(p) == "ES1":
        es1_path = p; break
src_es1 = load_csv(es1_path) if (symbol == "SPX" and es1_path) else None

# Build resampled frames
d1h = resample_ohlcv(src, "1H")
d4h = resample_ohlcv(src, "4H")
d1d = resample_ohlcv(src, "1D")

def inject_proxy_vol(resampled: pd.DataFrame, proxy_src: pd.DataFrame, rule: str) -> pd.DataFrame:
    if proxy_src is None or "volume" not in proxy_src.columns:
        return resampled
    p = resample_ohlcv(proxy_src, rule)[["date","volume"]]
    out = resampled.merge(p, on="date", how="left", suffixes=("","_proxy"))
    out["volume"] = np.where(out["volume_proxy"].notna(), out["volume_proxy"], out.get("volume"))
    return out.drop(columns=[c for c in out.columns if c.endswith("_proxy")])

if symbol == "SPX" and src_es1 is not None:
    d1h = inject_proxy_vol(d1h, src_es1, "1H")
    d4h = inject_proxy_vol(d4h, src_es1, "4H")
    d1d = inject_proxy_vol(d1d, src_es1, "1D")

# Enrich
d1h = enrich(d1h, spike_z=spike_z)
d4h = enrich(d4h, spike_z=spike_z)
d1d = enrich(d1d, spike_z=spike_z)

# Subset windows
def _subset_days(df: pd.DataFrame, n: int) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    cutoff = d["date"].max().normalize() - pd.Timedelta(days=n-1)
    return d[d["date"] >= cutoff]

d1h_view = _subset_days(d1h, days_1h)
d4h_view = _subset_days(d4h, days_4h)
d1d_view = _subset_days(d1d, days_1d)

# Opening 15m & tags
open15 = opening_15_close(src)
plan_1h = build_trade_plan(d1h_view)
strikes_1h = strike_suggestions(d1h_view, plan_1h)
alert_1h = tape_alert_text(d1h_view, plan_1h)
last = d1h_view.iloc[-1]
pa_text = price_action_tag(float(last["close"]), open15)

# Header summary under the dropdown
st.markdown(f"### {symbol} — 1H")
st.write(f"Close {last['close']:.2f} · ATR {plan_1h['atr']:.2f} · ADX {plan_1h['adx']:.1f} · {pa_text}")
st.write(f"Bias: **{plan_1h['bias']}** · Strategy: **{plan_1h['strategy']}**")
st.write(f"Volume: **{last.get('vol_state','NA')}**{' · ' + last['unusual'] if last.get('unusual') else ''}")
st.write(f"Strikes → PCS {strikes_1h['PCS'][0]}/{strikes_1h['PCS'][1]} · "
         f"CCS {strikes_1h['CCS'][0]}/{strikes_1h['CCS'][1]} · "
         f"IC {strikes_1h['IC'][0]}/{strikes_1h['IC'][1]} · {strikes_1h['IC'][2]}/{strikes_1h['IC'][3]}")

if alert_1h:
    st.info(f"⚡ Tape Alert: {alert_1h}")

# Tabs with charts
tab1, tab2, tab3 = st.tabs(["1H", "4H", "1D"])

with tab1:
    lv1 = compute_trade_levels(d1h_view, plan_1h, stop_mult=stop_mult,
                               t1_mult=t1_mult, t2_mult=t2_mult, t3_mult=t3_mult)
    st.markdown(f"### {symbol} — 1H")
    if lv1:
        if 'primary' in lv1:
            lv = lv1['primary']
            st.caption(f"**{lv['direction'].upper()}** · E:{lv['entry']:.1f} · S:{lv['stop']:.1f} · "
                       f"T1:{lv['t1']:.1f} · T2:{lv['t2']:.1f} · T3:{lv['t3']:.1f}")
        else:
            st.caption(f"**LONG** · E:{lv1['long']['entry']:.1f} · S:{lv1['long']['stop']:.1f} · "
                       f"T1:{lv1['long']['t1']:.1f} · T2:{lv1['long']['t2']:.1f} · T3:{lv1['long']['t3']:.1f}")
            st.caption(f"**SHORT** · E:{lv1['short']['entry']:.1f} · S:{lv1['short']['stop']:.1f} · "
                       f"T1:{lv1['short']['t1']:.1f} · T2:{lv1['short']['t2']:.1f} · T3:{lv1['short']['t3']:.1f}")

    fig1 = make_chart(d1h_view, f"{symbol} — 1H", height=560,
                      show_bbands=show_bbands, show_vol=show_vol)
    if show_levels:
        fig1 = add_levels_to_figure(fig1, lv1)
    st.plotly_chart(fig1, use_container_width=True)


with tab2:
    plan_4h = build_trade_plan(d4h_view)
    lv2 = compute_trade_levels(d4h_view, plan_4h, stop_mult=stop_mult,
                               t1_mult=t1_mult, t2_mult=t2_mult, t3_mult=t3_mult)
    st.markdown(f"### {symbol} — 4H")
    if lv2:
        if 'primary' in lv2:
            lv = lv2['primary']
            st.caption(f"**{lv['direction'].upper()}** · E:{lv['entry']:.1f} · S:{lv['stop']:.1f} · "
                       f"T1:{lv['t1']:.1f} · T2:{lv['t2']:.1f} · T3:{lv['t3']:.1f}")
        else:
            st.caption(f"**LONG** · E:{lv2['long']['entry']:.1f} · S:{lv2['long']['stop']:.1f} · "
                       f"T1:{lv2['long']['t1']:.1f} · T2:{lv2['long']['t2']:.1f} · T3:{lv2['long']['t3']:.1f}")
            st.caption(f"**SHORT** · E:{lv2['short']['entry']:.1f} · S:{lv2['short']['stop']:.1f} · "
                       f"T1:{lv2['short']['t1']:.1f} · T2:{lv2['short']['t2']:.1f} · T3:{lv2['short']['t3']:.1f}")

    fig2 = make_chart(d4h_view, f"{symbol} — 4H", height=520,
                      show_bbands=show_bbands, show_vol=show_vol)
    if show_levels:
        fig2 = add_levels_to_figure(fig2, lv2)
    st.plotly_chart(fig2, use_container_width=True)


with tab3:
    plan_1d = build_trade_plan(d1d_view)
    lv3 = compute_trade_levels(d1d_view, plan_1d, stop_mult=stop_mult,
                               t1_mult=t1_mult, t2_mult=t2_mult, t3_mult=t3_mult)
    st.markdown(f"### {symbol} — 1D")
    if lv3:
        if 'primary' in lv3:
            lv = lv3['primary']
            st.caption(f"**{lv['direction'].upper()}** · E:{lv['entry']:.1f} · S:{lv['stop']:.1f} · "
                       f"T1:{lv['t1']:.1f} · T2:{lv['t2']:.1f} · T3:{lv['t3']:.1f}")
        else:
            st.caption(f"**LONG** · E:{lv3['long']['entry']:.1f} · S:{lv3['long']['stop']:.1f} · "
                       f"T1:{lv3['long']['t1']:.1f} · T2:{lv3['long']['t2']:.1f} · T3:{lv3['long']['t3']:.1f}")
            st.caption(f"**SHORT** · E:{lv3['short']['entry']:.1f} · S:{lv3['short']['stop']:.1f} · "
                       f"T1:{lv3['short']['t1']:.1f} · T2:{lv3['short']['t2']:.1f} · T3:{lv3['short']['t3']:.1f}")

    fig3 = make_chart(d1d_view, f"{symbol} — 1D", height=520,
                      show_bbands=show_bbands, show_vol=show_vol)
    if show_levels:
        fig3 = add_levels_to_figure(fig3, lv3)
    st.plotly_chart(fig3, use_container_width=True)

