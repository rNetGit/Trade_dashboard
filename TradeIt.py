
# TradeIt.py — iTrader with 1H / 4H / 1D / 1W
# Presets (AM Scalp, Lunch Chop, PM Fade), projected ranges, rail badges,
# IC/PCS/CCS rails, delta proxy, volume features, and adaptive X-axis.
# Streamlit >=1.26 (uses st.rerun).

import os, glob, math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="iTrader", page_icon="📈", layout="wide")

PALETTE = {
    "bg": "#0f1420",
    "fg": "#e6edf7",
    "green": "#17c964",
    "red": "#f31260",
    "muted": "#9aa4b2",
    "accent": "#50b4ff",
    "bull": "#22c55e",
    "bear": "#ef4444",
    "neutral": "#9ca3af",
}

# ---------------- Core helpers / indicators ----------------
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
            d[c] = d.get("close", np.nan)

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
    roll_up = up.rolling(window, min_periods=2).mean()
    roll_down = down.rolling(window, min_periods=2).mean()
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
    h, l, c = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=2).mean()

def calculate_bbands(prices: pd.Series, window: int = 20, mult: float = 2.0):
    ma = prices.rolling(window, min_periods=2).mean()
    sd = prices.rolling(window, min_periods=2).std(ddof=0)
    return ma + mult * sd, ma, ma - mult * sd

def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    # Wilder-smoothed, vectorized ADX
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    up_move = h.diff()
    down_move = -l.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat([(h - l).abs(), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)

    alpha = 1.0 / float(window)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_sm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
    minus_sm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

    plus_di = 100.0 * (plus_sm / atr.replace(0, np.nan))
    minus_di = 100.0 * (minus_sm / atr.replace(0, np.nan))

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100.0
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx

# ---------------- S/R + plan + strikes ----------------
def _pivot_points(df: pd.DataFrame):
    prev = df.shift(1)
    P = (prev["high"] + prev["low"] + prev["close"]) / 3.0
    R1 = 2 * P - prev["low"]; S1 = 2 * P - prev["high"]
    R2 = P + (prev["high"] - prev["low"]); S2 = P - (prev["high"] - prev["low"])
    return P, R1, S1, R2, S2

def _donchian(df: pd.DataFrame, n: int = 20):
    upper = df["high"].rolling(n, min_periods=2).max()
    lower = df["low"].rolling(n, min_periods=2).min()
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def _fractals(df: pd.DataFrame, left: int = 2, right: int = 2):
    h, l = df["high"], df["low"]
    sh = h[(h.shift(1).rolling(left).max() < h) & (h.shift(-1).rolling(right).max() < h)]
    sl = l[(l.shift(1).rolling(left).min() > l) & (l.shift(-1).rolling(right).min() > l)]
    return sh.dropna(), sl.dropna()

def _round5(x: float) -> int:
    try: return int(round(float(x) / 5.0) * 5)
    except Exception: return int(x)

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

# ---------------- Volume features & alerts ----------------
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

# ---------------- Delta proxy ----------------
def delta_proxy_from_dist(dist_in_atr: float, decay: float = 0.9) -> float:
    if dist_in_atr is None or not np.isfinite(dist_in_atr):
        return np.nan
    return float(max(0.01, min(0.5, 0.5 * math.exp(-decay * max(0.0, dist_in_atr)))))

# ---------------- Projected Range ----------------
def projected_range(df: pd.DataFrame, k_atr: float = 1.0, use_bbands: bool = True, use_sr: bool = True, min_periods: int = 5):
    d = df.dropna(subset=["close","high","low"]).copy()
    if "atr" not in d.columns:
        d["atr"] = calculate_atr(d)
    if d["atr"].isna().iloc[-1]:
        tr = (d["high"] - d["low"]).abs()
        win = min(14, max(min_periods, len(tr)))
        d.loc[:, "atr"] = tr.rolling(win, min_periods=min_periods).mean()
    d = d.dropna(subset=["atr"])
    if d.empty: return None, None

    last = d.iloc[-1]
    close = float(last["close"]); atr = float(last["atr"]) if not pd.isna(last["atr"]) else 0.0
    core_hi = close + k_atr * atr; core_lo = close - k_atr * atr
    hi_candidates = [core_hi]; lo_candidates = [core_lo]

    if use_bbands and {"bb_upper","bb_lower"}.issubset(d.columns):
        if not pd.isna(last["bb_upper"]): hi_candidates.append(float(last["bb_upper"]))
        if not pd.isna(last["bb_lower"]): lo_candidates.append(float(last["bb_lower"]))

    if use_sr:
        levs = fused_levels(d)
        if levs["resistance"]: hi_candidates.append(float(levs["resistance"][0][0]))
        if levs["support"]: lo_candidates.append(float(levs["support"][0][0]))

    return float(min(lo_candidates)), float(max(hi_candidates))

# ---------------- Plotting ----------------
def make_chart(d: pd.DataFrame, title: str, height: int = 560,
               show_bbands: bool = True, show_vol: bool = True, clamp_x: bool = True):
    d = d.copy()
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=d["date"], open=d["open"], high=d["high"], low=d["low"], close=d["close"],
        increasing_line_color=PALETTE["green"], decreasing_line_color=PALETTE["red"], name="Price", yaxis="y1"
    ))
    fig.add_trace(go.Scatter(x=d["date"], y=d["ema20"], name="EMA20", line=dict(dash="dot"), yaxis="y1"))
    fig.add_trace(go.Scatter(x=d["date"], y=d["ema50"], name="EMA50", line=dict(dash="dot"), yaxis="y1"))
    if "vwap" in d and not d["vwap"].isna().all():
        fig.add_trace(go.Scatter(x=d["date"], y=d["vwap"], name="VWAP", line=dict(dash="dash"), yaxis="y1"))

    if show_bbands:
        fig.add_trace(go.Scatter(x=d["date"], y=d["bb_upper"], name="BB Upper", line=dict(width=1), opacity=.7, yaxis="y1"))
        fig.add_trace(go.Scatter(x=d["date"], y=d["bb_mid"],   name="BB Mid",   line=dict(width=1, dash="dot"), opacity=.6, yaxis="y1"))
        fig.add_trace(go.Scatter(x=d["date"], y=d["bb_lower"], name="BB Lower", line=dict(width=1), opacity=.7, yaxis="y1"))

    if show_vol and "volume" in d.columns:
        vol_colors = np.where(d["close"]>=d["open"], PALETTE["green"], PALETTE["red"])
        fig.add_trace(go.Bar(x=d["date"], y=d["volume"], name="Volume", marker_color=vol_colors, opacity=0.7, yaxis="y2"))
        if "vol_state" in d.columns:
            mask = d["vol_state"]=="Spiked"
            fig.add_trace(go.Scatter(x=d["date"][mask], y=d["volume"][mask]*1.02,
                                     mode="markers", name="Vol Spike", yaxis="y2",
                                     marker=dict(size=7, symbol="triangle-up", color=PALETTE["accent"])))

    ymin, ymax = float(d["low"].min()), float(d["high"].max())
    pad = (ymax-ymin)*0.03 if ymax>ymin else 1.0
    fig.update_layout(
        title=title,
        height=height,
        margin=dict(l=8, r=8, t=40, b=8),
        plot_bgcolor=PALETTE["bg"], paper_bgcolor=PALETTE["bg"],
        font=dict(color=PALETTE["fg"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="overlay",
        yaxis=dict(domain=[0.25,1.0], range=[ymin-pad, ymax+pad], title=None, gridcolor="#1f2430"),
        yaxis2=dict(domain=[0.0,0.22], title=None, gridcolor="#1f2430"),
        xaxis=dict(rangebreaks=[dict(bounds=['sat', 'mon'])])
    )

    fig.update_xaxes(
        tickformatstops=[
            dict(dtickrange=[None, 1000*60*60*24], value="%b %d\n%H:%M"),
            dict(dtickrange=[1000*60*60*24, 1000*60*60*24*31], value="%b %d"),
            dict(dtickrange=[1000*60*60*24*31, None], value="%b %Y"),
        ]
    )
    if clamp_x and len(d) >= 2:
        fig.update_xaxes(range=[d["date"].iloc[0], d["date"].iloc[-1]])
    return fig

# ---------------- Trade level visuals ----------------
def compute_trade_levels(df: pd.DataFrame, plan: dict, rr: float = 1.5,
                         stop_mult: float = 0.8, t1_mult: float = 0.5,
                         t2_mult: float = 1.0, t3_mult: float = 1.5):
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
    return {'long': _dir_levels('long'), 'short': _dir_levels('short')}

def add_levels_to_figure(fig: go.Figure, levels: dict, yaxis='y1'):
    def _draw_set(lv: dict, color=PALETTE["bull"]):
        fig.add_hline(y=lv['stop'], line=dict(width=1, dash='dot', color=color))
        fig.add_hline(y=lv['t1'],   line=dict(width=1, dash='solid', color=color))
        fig.add_hline(y=lv['t2'],   line=dict(width=1, dash='dash', color=color))
        fig.add_hline(y=lv['t3'],   line=dict(width=1, dash='dashdot', color=color))
        fig.add_hline(y=lv['entry'], line=dict(width=2, color=color))
    if not levels:
        return fig
    if 'primary' in levels:
        _draw_set(levels['primary'], color=PALETTE["bull"])
    else:
        _draw_set(levels['long'], color=PALETTE["bull"])
        _draw_set(levels['short'], color=PALETTE["bear"])
    return fig

# ---------------- Loading & enrichment ----------------
@st.cache_data(show_spinner=False)
def load_csv(p: str) -> pd.DataFrame:
    df = pd.read_csv(p)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = None
    for cand in ["date", "datetime", "time", "timestamp"]:
        if cand in df.columns:
            date_col = cand; break
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

# ---------------- Price action tags ----------------
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

# ---------------- Rail helpers ----------------
def make_rail_texts(df, lo, hi, bias, ic_width, dir_width, decay):
    close = float(df["close"].iloc[-1])
    atr   = float(df["atr"].iloc[-1]) if "atr" in df.columns and not df["atr"].isna().all() else float(calculate_atr(df).iloc[-1])
    short_put  = _round5(lo if lo is not None else close - atr)
    short_call = _round5(hi if hi is not None else close + atr)

    ic_text = f"IC {short_put}/{short_put-ic_width} & {short_call}/{short_call+ic_width}"
    pcs_text = f"PCS {short_put}/{short_put-dir_width}"
    ccs_text = f"CCS {short_call}/{short_call+dir_width}"

    def _dp(sp, sc):
        dist_p = max(0.0, (close - sp)/atr)
        dist_c = max(0.0, (sc - close)/atr)
        dp = delta_proxy_from_dist(dist_p, decay)
        dc = delta_proxy_from_dist(dist_c, decay)
        return dp, dc

    dp_ic, dc_ic = _dp(short_put, short_call)
    dp_dir, dc_dir = _dp(short_put, short_call)

    if bias == "NEUTRAL":
        main = f"{ic_text} (Δ≈{dp_ic:.2f}/{dc_ic:.2f})"
    elif bias == "BULLISH":
        main = f"{pcs_text} (Δ≈{dp_dir:.2f})"
    else:
        main = f"{ccs_text} (Δ≈{dc_dir:.2f})"

    aux  = f"{pcs_text} · {ccs_text}"
    return main, aux

def add_rail_badge(fig, main_text, aux_text, bias_color="#50b4ff"):
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.98,
        text=f"<b>{main_text}</b><br><span style='opacity:.8'>{aux_text}</span>",
        showarrow=False, align="left",
        bgcolor="rgba(20,28,40,0.80)", bordercolor=bias_color, borderwidth=1,
        font=dict(size=12)
    )
    return fig

# ---------------- UI ----------------
st.markdown("<h4>iTrader - Technical Analysis</h4>", unsafe_allow_html=True)

# Presets
DEFAULTS = {
    "range_method": "ATR + BB + S/R",
    "days_1h": 30, "days_4h": 15, "days_1d": 60, "weeks_1w": 52,
    "dir_width": 10, "ic_width_1d": 10, "ic_width_1w": 15,
    "delta_decay": 0.9, "clamp_x": True,
}
PRESETS = {
    "AM Scalp": {
        "range_method": "ATR-only",
        "days_1h": 7, "days_4h": 10, "days_1d": 60, "weeks_1w": 52,
        "dir_width": 10, "ic_width_1d": 10, "ic_width_1w": 15,
        "delta_decay": 1.10, "clamp_x": True
    },
    "Lunch Chop": {
        "range_method": "ATR + BB + S/R",
        "days_1h": 12, "days_4h": 18, "days_1d": 60, "weeks_1w": 52,
        "dir_width": 12, "ic_width_1d": 15, "ic_width_1w": 20,
        "delta_decay": 0.85, "clamp_x": True
    },
    "PM Fade": {
        "range_method": "ATR-only",
        "days_1h": 6, "days_4h": 12, "days_1d": 45, "weeks_1w": 52,
        "dir_width": 18, "ic_width_1d": 20, "ic_width_1w": 25,
        "delta_decay": 1.25, "clamp_x": True
    },
}
def _init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init_state()

# Sidebar: presets first
preset_name = st.sidebar.selectbox("Preset", ["Custom"] + list(PRESETS.keys()))
if preset_name != "Custom" and st.sidebar.button("Apply Preset"):
    st.session_state.update(PRESETS[preset_name])
    st.rerun()

# Sidebar controls
data_folder = st.sidebar.text_input("Folder with CSVs", value=".", help="Folder with SPX.csv (+ ES1.csv optional)")
show_bbands = st.sidebar.checkbox("Bollinger Bands", True)
show_vol    = st.sidebar.checkbox("Volume panel", True)
show_levels = st.sidebar.checkbox("Show Entry/Stop/Targets", True)

days_1h  = st.sidebar.slider("Days (1H)", 3, 90, st.session_state["days_1h"], key="days_1h")
days_4h  = st.sidebar.slider("Days (4H)", 3, 60, st.session_state["days_4h"], key="days_4h")
days_1d  = st.sidebar.slider("Days (1D)", 5, 365, st.session_state["days_1d"], key="days_1d")
weeks_1w = st.sidebar.slider("Weeks (1W)", 4, 156, st.session_state["weeks_1w"], key="weeks_1w")
range_method = st.sidebar.selectbox("Range Method", ["ATR-only", "ATR + BB + S/R"], index=0 if st.session_state["range_method"]=="ATR-only" else 1, key="range_method")

stop_mult = st.sidebar.slider("Stop (×ATR)", 0.3, 2.0, 0.8, 0.05)
t1_mult   = st.sidebar.slider("T1 (×ATR)", 0.3, 2.5, 0.5, 0.05)
t2_mult   = st.sidebar.slider("T2 (×ATR)", 0.5, 3.0, 1.0, 0.05)
t3_mult   = st.sidebar.slider("T3 (×ATR)", 0.5, 4.0, 1.5, 0.05)
spike_z   = st.sidebar.slider("Spike Z-score", 1.5, 4.0, 2.0, 0.1)

# Robust width options: include 12 and 18 for presets.
options_dir = [5,10,12,15,18,20,25,30]
cur_dir = st.session_state.get("dir_width", 10)
if cur_dir not in options_dir:
    cur_dir = min(options_dir, key=lambda x: abs(x - cur_dir))
    st.session_state["dir_width"] = cur_dir

ic_width_1d = st.sidebar.select_slider("IC Width 1D (pts)", options=[5,10,15,20,25,30], value=st.session_state["ic_width_1d"], key="ic_width_1d")
ic_width_1w = st.sidebar.select_slider("IC Width 1W (pts)", options=[5,10,15,20,25,30], value=st.session_state["ic_width_1w"], key="ic_width_1w")
dir_width   = st.sidebar.select_slider("PCS/CCS Width (pts)", options=options_dir, value=cur_dir, key="dir_width")
delta_decay = st.sidebar.slider("Delta Proxy Decay (↑=more conservative)", 0.5, 1.5, st.session_state["delta_decay"], 0.05, key="delta_decay")
clamp_x     = st.sidebar.checkbox("Fit X-axis to selected window", st.session_state["clamp_x"], key="clamp_x")

# Discover CSVs
csv_paths = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
if not csv_paths:
    st.error("No CSVs found in folder"); st.stop()

def _sym(p): return os.path.splitext(os.path.basename(p))[0].upper()
opts = [_sym(p) for p in csv_paths]
default_sym = "SPX" if "SPX" in opts else opts[0]
symbol = st.selectbox("Symbol", opts, index=opts.index(default_sym), key="symbol")

# Load and prepare
path = csv_paths[opts.index(symbol)]
src = load_csv(path)


es1_path = None
for p in csv_paths:
    if _sym(p) == "ES1":
        es1_path = p; break
src_es1 = load_csv(es1_path) if (symbol == "SPX" and es1_path) else None

# --- AFTER loading paths and calling: src = load_csv(path) and (optionally) src_es1 = load_csv(es1_path) ---

# NEW: Back-date control (default = all data)
min_d = src["date"].min().date()
max_d = src["date"].max().date()

use_cap = st.sidebar.checkbox(
    "Limit data up to date",
    value=False,
    help="Backtest past sessions by hiding all bars after the selected date (EOD).",
)

cap_date = st.sidebar.date_input(
    "Data end date (EOD)",
    value=max_d,
    min_value=min_d,
    max_value=max_d,
    disabled=not use_cap,
)

if use_cap:
    # cap at end-of-day (naive UTC in this app)
    cap_dt = pd.Timestamp(cap_date) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    src = src[src["date"] <= cap_dt].copy()
    if 'src_es1' in locals() and src_es1 is not None:
        src_es1 = src_es1[src_es1["date"] <= cap_dt].copy()

    if src.empty:
        st.warning("No data on or before the selected date. Showing original dataset.")
        # fall back to full data if user picked too-early a date
        src = load_csv(path)
        if 'src_es1' in locals() and es1_path:
            src_es1 = load_csv(es1_path)

    # small banner so it’s obvious we’re not on latest data
    st.caption(f"🔒 Backtest mode: using data through **{cap_date.isoformat()}**")




d1h = resample_ohlcv(src, "1H")
d4h = resample_ohlcv(src, "4H")
d1d = resample_ohlcv(src, "1D")
d1w = resample_ohlcv(src, "1W")

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
    d1w = inject_proxy_vol(d1w, src_es1, "1W")

d1h = enrich(d1h, spike_z=spike_z)
d4h = enrich(d4h, spike_z=spike_z)
d1d = enrich(d1d, spike_z=spike_z)
d1w = enrich(d1w, spike_z=spike_z)

def _subset_days(df: pd.DataFrame, n: int) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    cutoff = d["date"].max().normalize() - pd.Timedelta(days=n-1)
    return d[d["date"] >= cutoff]

def _subset_weeks(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return _subset_days(df, n*7)

d1h_view = _subset_days(d1h, st.session_state["days_1h"])
d4h_view = _subset_days(d4h, st.session_state["days_4h"])
d1d_view = _subset_days(d1d, st.session_state["days_1d"])
d1w_view = _subset_weeks(d1w, st.session_state["weeks_1w"])

# Header panel (based on 1H)
open15 = opening_15_close(src)
plan_1h = build_trade_plan(d1h_view)
strikes_1h = strike_suggestions(d1h_view, plan_1h)
alert_1h = tape_alert_text(d1h_view, plan_1h)
last = d1h_view.iloc[-1]
pa_text = price_action_tag(float(last["close"]), open15)

use_b = (st.session_state["range_method"] == "ATR + BB + S/R")
use_s = (st.session_state["range_method"] == "ATR + BB + S/R")
lo_1h, hi_1h = projected_range(d1h_view, k_atr=0.75, use_bbands=use_b, use_sr=use_s)
lo_4h, hi_4h = projected_range(d4h_view, k_atr=1.00, use_bbands=use_b, use_sr=use_s)
lo_1d, hi_1d = projected_range(d1d_view, k_atr=1.25, use_bbands=use_b, use_sr=use_s)
lo_1w, hi_1w = projected_range(d1w_view, k_atr=1.50, use_bbands=use_b, use_sr=use_s)

def _fmt(x): return f"{x:.1f}" if x is not None else "—"

header_ranges = (
    f"Ranges → 1H {_fmt(lo_1h)}–{_fmt(hi_1h)} · "
    f"4H {_fmt(lo_4h)}–{_fmt(hi_4h)} · "
    f"1D {_fmt(lo_1d)}–{_fmt(hi_1d)} · "
    f"1W {_fmt(lo_1w)}–{_fmt(hi_1w)} "
    f"({st.session_state['range_method']})"
)

st.markdown(f"### {symbol} — 1H")
st.write(f"Close {last['close']:.2f} · ATR {plan_1h['atr']:.2f} · ADX {plan_1h['adx']:.1f} · {pa_text}")
st.write(f"Bias: **{plan_1h['bias']}** · Strategy: **{plan_1h['strategy']}**")
bias = plan_1h["bias"] or "NEUTRAL"
bias_color = {"BULLISH":PALETTE["bull"], "BEARISH":PALETTE["bear"], "NEUTRAL":PALETTE["neutral"]}.get(bias, PALETTE["neutral"])
st.markdown(f'<div style="color:{bias_color};font-weight:600;">{header_ranges}</div>', unsafe_allow_html=True)
st.write(f"Volume: **{last.get('vol_state','NA')}**{' · ' + last['unusual'] if last.get('unusual') else ''}")
st.write(f"Strikes → PCS {strikes_1h['PCS'][0]}/{strikes_1h['PCS'][1]} · "
         f"CCS {strikes_1h['CCS'][0]}/{strikes_1h['CCS'][1]} · "
         f"IC {strikes_1h['IC'][0]}/{strikes_1h['IC'][1]} · {strikes_1h['IC'][2]}/{strikes_1h['IC'][3]}")

# Quick header IC summaries (1D/1W) + delta proxies
def delta_proxy_for_frame(df_view, short_put, short_call, decay):
    if df_view is None or len(df_view)==0 or short_put is None or short_call is None:
        return np.nan, np.nan
    close = float(df_view["close"].iloc[-1])
    atr = float(df_view["atr"].iloc[-1]) if "atr" in df_view.columns else float(calculate_atr(df_view).iloc[-1])
    if not np.isfinite(atr) or atr <= 0: 
        return np.nan, np.nan
    dist_put = max(0.0, (close - short_put) / atr)
    dist_call = max(0.0, (short_call - close) / atr)
    return delta_proxy_from_dist(dist_put, decay), delta_proxy_from_dist(dist_call, decay)

def build_ic_text(lo, hi, width_pts):
    if lo is None or hi is None: return "—", None, None
    sp = _round5(lo); sc = _round5(hi)
    return f"{sp}/{sp-width_pts} & {sc}/{sc+width_pts}", sp, sc

ic_1d_text, put_1d, call_1d = build_ic_text(lo_1d, hi_1d, st.session_state["ic_width_1d"])
ic_1w_text, put_1w, call_1w = build_ic_text(lo_1w, hi_1w, st.session_state["ic_width_1w"])
p_delta_1d, c_delta_1d = delta_proxy_for_frame(d1d_view, put_1d, call_1d, st.session_state["delta_decay"])
p_delta_1w, c_delta_1w = delta_proxy_for_frame(d1w_view, put_1w, call_1w, st.session_state["delta_decay"])

st.markdown(
    f"<div style='opacity:0.95'><b>IC 1D:</b> {ic_1d_text} (Δ≈{p_delta_1d:.2f}/{c_delta_1d:.2f}) &nbsp; · "
    f"<b>IC 1W:</b> {ic_1w_text} (Δ≈{p_delta_1w:.2f}/{c_delta_1w:.2f})</div>",
    unsafe_allow_html=True
)

if alert_1h:
    st.info(f"⚡ Tape Alert: {alert_1h}")

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["1H", "4H", "1D", "1W"])

def _render_tab(df, lbl):
    plan = build_trade_plan(df)
    lv = compute_trade_levels(df, plan, stop_mult=0.8, t1_mult=0.5, t2_mult=1.0, t3_mult=1.5)
    st.markdown(f"### {symbol} — {lbl}")
    if lv:
        if 'primary' in lv:
            p = lv['primary']
            st.caption(f"**{p['direction'].upper()}** · E:{p['entry']:.1f} · S:{p['stop']:.1f} · T1:{p['t1']:.1f} · T2:{p['t2']:.1f} · T3:{p['t3']:.1f}")
        else:
            st.caption(f"**LONG** · E:{lv['long']['entry']:.1f} · S:{lv['long']['stop']:.1f} · T1:{lv['long']['t1']:.1f} · T2:{lv['long']['t2']:.1f} · T3:{lv['long']['t3']:.1f}")
            st.caption(f"**SHORT** · E:{lv['short']['entry']:.1f} · S:{lv['short']['stop']:.1f} · T1:{lv['short']['t1']:.1f} · T2:{lv['short']['t2']:.1f} · T3:{lv['short']['t3']:.1f}")

    k_map = {"1H":0.75,"4H":1.00,"1D":1.25,"1W":1.50}
    use_b = (st.session_state["range_method"] == "ATR + BB + S/R")
    use_s = (st.session_state["range_method"] == "ATR + BB + S/R")
    lo, hi = projected_range(df, k_atr=k_map[lbl], use_bbands=use_b, use_sr=use_s)
    if lo is not None and hi is not None:
        st.caption(f"Projected Range: {lo:.1f} — {hi:.1f}")

    fig = make_chart(df, f"{symbol} — {lbl}", height=520, show_bbands=show_bbands, show_vol=show_vol, clamp_x=st.session_state["clamp_x"])
    fig = add_levels_to_figure(fig, lv) if True else fig
    if lo is not None and hi is not None:
        fig.add_hrect(y0=lo, y1=hi, fillcolor="rgba(80,180,255,0.08)", line_width=0)

    # Rail badge
    ic_w = st.session_state["ic_width_1d"] if lbl=="1D" else st.session_state["ic_width_1w"] if lbl=="1W" else st.session_state["ic_width_1d"]
    dir_w = st.session_state["dir_width"]
    main, aux = make_rail_texts(df, lo, hi, plan["bias"], ic_w, dir_w, st.session_state["delta_decay"])
    badge_color = {"BULLISH":PALETTE["bull"], "BEARISH":PALETTE["bear"], "NEUTRAL":PALETTE["accent"]}.get(plan["bias"], PALETTE["accent"])
    fig = add_rail_badge(fig, main, aux, bias_color=badge_color)

    st.plotly_chart(fig, use_container_width=True)

with tab1: _render_tab(d1h_view, "1H")
with tab2: _render_tab(d4h_view, "4H")
with tab3: _render_tab(d1d_view, "1D")
with tab4: _render_tab(d1w_view, "1W")
