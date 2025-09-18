import numpy as np
import pandas as pd

# ---------------- Helpers ----------------

def _to_naive_utc_index(series: pd.Series) -> pd.DatetimeIndex:
    s = pd.to_datetime(series, utc=True, errors="coerce")
    idx = pd.DatetimeIndex(s).tz_convert(None)
    return idx

# ---------------- Core Indicators ----------------
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
        return pd.Series(np.nan, index=df.index)
    tp = df["close"]
    pv = tp * df["volume"]
    return pv.cumsum() / df["volume"].cumsum()

def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - prev_c).abs(), (l - prev_c).abs()))
    return tr.rolling(window).mean()

# ---------------- Robust Support/Resistance ----------------
def _pivot_points(df: pd.DataFrame):
    prev = df.shift(1)
    P = (prev["high"] + prev["low"] + prev["close"]) / 3.0
    R1 = 2 * P - prev["low"]
    S1 = 2 * P - prev["high"]
    R2 = P + (prev["high"] - prev["low"])
    S2 = P - (prev["high"] - prev["low"])
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

# ---------------- Trend & Trade Plan ----------------
def classify_trend(close: pd.Series, ema20: pd.Series, ema50: pd.Series) -> str:
    price = float(close.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    if price > e20 > e50: return "up"
    if price < e20 < e50: return "down"
    return "sideways"

def build_trade_plan(df: pd.DataFrame):
    price = float(df["close"].iloc[-1])
    _atr = calculate_atr(df).iloc[-1]
    _atr = float(_atr) if not np.isnan(_atr) else max(1.0, price * 0.005)
    levels = fused_levels(df)
    trend = classify_trend(df["close"], calculate_ema(df["close"], 20), calculate_ema(df["close"], 50))
    if trend == "up":
        res = levels["resistance"][0][0] if levels["resistance"] else price + 0.5 * _atr
        sup = levels["support"][0][0] if levels["support"] else price - 1.0 * _atr
        entry = max(price, res + 0.1 * _atr); stop = sup - 0.5 * _atr
        t1, t2 = entry + 1.0 * _atr, entry + 2.0 * _atr
        bias, strat = "BULLISH", "Put Credit Spread (PCS) / Long bias"
    elif trend == "down":
        res = levels["resistance"][0][0] if levels["resistance"] else price + 1.0 * _atr
        sup = levels["support"][0][0] if levels["support"] else price - 0.5 * _atr
        entry = min(price, sup - 0.1 * _atr); stop = res + 0.5 * _atr
        t1, t2 = entry - 1.0 * _atr, entry - 2.0 * _atr
        bias, strat = "BEARISH", "Call Credit Spread (CCS) / Short bias"
    else:
        if levels["resistance"] and levels["support"]:
            top = levels["resistance"][0][0]; bot = levels["support"][0][0]
        else:
            top, bot = price + 1.5 * _atr, price - 1.5 * _atr
        entry = None; stop = None
        t1 = (top + bot) / 2.0
        t2 = top if abs(top - price) < abs(price - bot) else bot
        bias, strat = "NEUTRAL", "Iron Condor / Butterflies"
    return {"bias": bias, "strategy": strat, "atr": _atr, "levels": levels, "entry": entry, "stop": stop, "targets": [t1, t2]}

# ---------------- Timeframe helpers & projections ----------------
def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    d = df.copy()
    idx = _to_naive_utc_index(d["date"])
    mask = ~idx.isna()
    d = d.loc[mask].copy()
    idx = idx[mask]
    d.index = idx
    d = d.sort_index()

    for c in ["open", "high", "low", "close"]:
        if c not in d.columns:
            d[c] = d["close"]

    o = d["open"].resample(rule).first()
    h = d["high"].resample(rule).max()
    l = d["low"].resample(rule).min()
    c = d["close"].resample(rule).last()
    if "volume" in d.columns:
        v = d["volume"].resample(rule).sum()
        out = pd.DataFrame({"date": o.index, "open": o.values, "high": h.values, "low": l.values, "close": c.values, "volume": v.values})
    else:
        out = pd.DataFrame({"date": o.index, "open": o.values, "high": h.values, "low": l.values, "close": c.values})

    return out.dropna().reset_index(drop=True)

def project_day_range(df_day: pd.DataFrame):
    if len(df_day) < 15: return None
    prev = df_day.iloc[-2]
    atr_d = calculate_atr(df_day).iloc[-1]
    return {
        "proj_hi": float(prev["close"] + (atr_d if not np.isnan(atr_d) else (prev["high"]-prev["low"]))),
        "proj_lo": float(prev["close"] - (atr_d if not np.isnan(atr_d) else (prev["high"]-prev["low"]))),
    }

def project_week_range(df_day: pd.DataFrame):
    if len(df_day) < 40: return None
    d = df_day.set_index(pd.to_datetime(df_day["date"]))
    wk = pd.DataFrame({
        "high": d["high"].resample("W").max(),
        "low": d["low"].resample("W").min(),
        "close": d["close"].resample("W").last()
    }).dropna()
    if len(wk) < 3: return None
    prev = wk.iloc[-2]
    tr = (wk["high"] - wk["low"]).rolling(14).mean()
    atr_w = tr.iloc[-1]
    est = atr_w if not np.isnan(atr_w) else (prev["high"] - prev["low"])
    return {
        "proj_hi": float(prev["close"] + est),
        "proj_lo": float(prev["close"] - est),
    }

def _sr_top2(levels: dict) -> tuple[list[float], list[float]]:
    s = [float(x[0]) for x in levels.get("support", [])[:2]]
    r = [float(x[0]) for x in levels.get("resistance", [])[:2]]
    while len(s) < 2: s.append(np.nan)
    while len(r) < 2: r.append(np.nan)
    return s, r

def _fmt_sr_line(s: list[float], r: list[float]) -> str:
    to2 = lambda v: ("—" if np.isnan(v) else f"{v:.2f}")
    return f"S1 {to2(s[0])} · S2 {to2(s[1])}  |  R1 {to2(r[0])} · R2 {to2(r[1])}"

# Extra indicators
def calculate_bbands(prices: pd.Series, window: int = 20, mult: float = 2.0):
    ma = prices.rolling(window).mean()
    sd = prices.rolling(window).std(ddof=0)
    upper = ma + mult * sd
    lower = ma - mult * sd
    return upper, ma, lower

def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    # Wilder’s ADX (simplified)
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    down = -l.diff()
    plus_dm  = ((up > down) & (up > 0)).astype(float) * up
    minus_dm = ((down > up) & (down > 0)).astype(float) * down

    tr = (h.combine(c.shift(), max) - l.combine(c.shift(), min)).abs()
    tr = tr.rolling(window).sum()

    plus_di  = 100 * (plus_dm.rolling(window).sum() / tr)
    minus_di = 100 * (minus_dm.rolling(window).sum() / tr)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)) * 100
    adx = dx.rolling(window).mean()
    return adx
