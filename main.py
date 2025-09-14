import io
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ----- Local utils
from ta_utils import (
    calculate_rsi,
    calculate_ema,
    calculate_macd,
    calculate_vwap,
    calculate_atr,
    get_levels,
)

# ---------- Page ----------
st.set_page_config(page_title="SPX Dashboard", page_icon="ðŸ“ˆ", layout="wide")
st.title("SPX Technical Dashboard")

# ---------- Data ----------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df_ = pd.read_csv(path)
    # normalize common column names
    df_.columns = [c.strip().lower() for c in df_.columns]
    if "date" not in df_.columns:
        raise ValueError("CSV must include a 'date' column (YYYY-MM-DD or ISO).")
    df_["date"] = pd.to_datetime(df_["date"])
    df_ = df_.sort_values("date").reset_index(drop=True)
    required_price_cols = {"close"}
    missing = required_price_cols - set(df_.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    # optional columns with safe defaults
    for c in ["high", "low", "open"]:
        if c not in df_.columns:
            df_[c] = df_["close"]
    return df_

df = load_data("spx_data.csv")

# ---------- Indicators ----------
df["rsi"] = calculate_rsi(df["close"])
df["ema_20"] = calculate_ema(df["close"], 20)
df["ema_50"] = calculate_ema(df["close"], 50)
df["vwap"] = calculate_vwap(df)
macd_out = calculate_macd(df["close"])
# Support both 2-tuple and 3-tuple returns from utils
if isinstance(macd_out, tuple) and len(macd_out) >= 2:
    df["macd"], df["macd_signal"] = macd_out[0], macd_out[1]
else:
    raise ValueError("calculate_macd must return at least (macd, signal).")
df["atr"] = calculate_atr(df)

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest

# ---------- Support/Resistance ----------
high_5d, low_5d, high_10d, low_10d = get_levels(df)

price = float(latest["close"])
atr = float(latest["atr"]) if not pd.isna(latest["atr"]) else np.nan

# Build candidate levels, then split into supports (<= price) and resistances (>= price)
candidates = {
    "vwap": float(latest["vwap"]) if not pd.isna(latest["vwap"]) else np.nan,
    "ema20": float(latest["ema_20"]),
    "ema50": float(latest["ema_50"]),
    "low_5d": float(low_5d),
    "low_10d": float(low_10d),
    "high_5d": float(high_5d),
    "high_10d": float(high_10d),
    "today_high": float(latest.get("high", price)),
    "today_low": float(latest.get("low", price)),
    "close+atr": float(price + (atr if not np.isnan(atr) else 50.0)),
    "close-atr": float(price - (atr if not np.isnan(atr) else 50.0)),
}

supports = sorted([v for v in candidates.values() if not np.isnan(v) and v <= price], reverse=True)[:3]
resistances = sorted([v for v in candidates.values() if not np.isnan(v) and v >= price])[:3]

def fmt3(vals):
    # pad to 3 values for display
    vals = list(vals)
    while len(vals) < 3:
        vals.append(np.nan)
    return vals

support_levels = fmt3(supports)
resistance_levels = fmt3(resistances)

# ---------- Signal logic ----------
signals = {"bullish": 0, "bearish": 0, "neutral": 0}

# RSI
rsi = float(latest["rsi"])
if rsi >= 70:
    rsi_sentiment = "Bearish (Overbought)"
    signals["bearish"] += 1
elif rsi <= 30:
    rsi_sentiment = "Bullish (Oversold)"
    signals["bullish"] += 1
elif rsi >= 55:
    rsi_sentiment = "Bullish"
    signals["bullish"] += 1
elif rsi <= 45:
    rsi_sentiment = "Bearish"
    signals["bearish"] += 1
else:
    rsi_sentiment = "Neutral"
    signals["neutral"] += 1

# Moving averages trend
price_above_ema20 = price > float(latest["ema_20"])
price_above_ema50 = price > float(latest["ema_50"])
ema20_above_ema50 = float(latest["ema_20"]) > float(latest["ema_50"])

if price_above_ema20 and price_above_ema50 and ema20_above_ema50:
    ma_sentiment = "Strong Bullish"
    signals["bullish"] += 2
elif price_above_ema20 and price_above_ema50:
    ma_sentiment = "Bullish"
    signals["bullish"] += 1
elif (not price_above_ema20) and (not price_above_ema50):
    ma_sentiment = "Bearish"
    signals["bearish"] += 1
else:
    ma_sentiment = "Mixed"
    signals["neutral"] += 1

# VWAP
if not np.isnan(latest["vwap"]) and price > float(latest["vwap"]):
    vwap_sentiment = "Bullish"
    signals["bullish"] += 1
elif not np.isnan(latest["vwap"]):
    vwap_sentiment = "Bearish"
    signals["bearish"] += 1
else:
    vwap_sentiment = "N/A"
    signals["neutral"] += 1

# MACD (prefer macd vs signal; require both > 0 for strongest)
macd = float(latest["macd"])
macd_signal = float(latest["macd_signal"])
if macd > macd_signal and macd > 0:
    macd_sentiment = "Bullish"
    signals["bullish"] += 1
elif macd < macd_signal and macd < 0:
    macd_sentiment = "Bearish"
    signals["bearish"] += 1
else:
    macd_sentiment = "Neutral"
    signals["neutral"] += 1

# Bias
if signals["bullish"] > signals["bearish"]:
    market_bias = "BULLISH"
    recommended_strategy = "Put Credit Spread (PCS)"
elif signals["bearish"] > signals["bullish"]:
    market_bias = "BEARISH"
    recommended_strategy = "Call Credit Spread (CCS)"
else:
    market_bias = "NEUTRAL"
    recommended_strategy = "Iron Condor / Butterfly"

# ---------- Layout ----------
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.subheader("Market Bias")
    st.markdown(f"**Bias:** {market_bias}")
    st.markdown(f"**Strategy:** {recommended_strategy}")
    st.markdown(f"**Bullish Signals:** {signals['bullish']}")
    st.markdown(f"**Bearish Signals:** {signals['bearish']}")
    st.markdown(f"**Neutral Signals:** {signals['neutral']}")
    st.markdown("#### Macro / Notes")
    st.info(
        "Watch employment & inflation prints, Fed speakers, and intraday breadth. "
        "Adjust credit spreads to ATR and time-to-expiry."
    )

with col2:
    st.subheader("Technical Indicators (Latest)")
    st.markdown(f"**RSI (14):** {rsi:.2f} â€” {rsi_sentiment}")
    st.markdown(f"**MACD:** {macd:.2f} vs Signal {macd_signal:.2f} â€” {macd_sentiment}")
    st.markdown(f"**EMA 20/50:** {ma_sentiment} ({latest['ema_20']:.2f} / {latest['ema_50']:.2f})")
    vwap_text = "N/A" if np.isnan(latest["vwap"]) else f"{latest['vwap']:.2f} â€” {vwap_sentiment}"
    st.markdown(f"**VWAP:** {vwap_text}")

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Close"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ema_20"], name="EMA 20", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df["date"], y=df["ema_50"], name="EMA 50", line=dict(dash="dot")))
    if not df["vwap"].isna().all():
        fig.add_trace(go.Scatter(x=df["date"], y=df["vwap"], name="VWAP", line=dict(dash="dash")))
    fig.update_layout(
        title="SPX Price & Key Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        plot_bgcolor="#1a1a1a",
        paper_bgcolor="#1a1a1a",
        font=dict(color="#FFFFFF"),
        legend=dict(bgcolor="#1a1a1a"),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Levels")
    st.markdown(
        f"**Support:** "
        f"S1: {support_levels[0]:.2f} Â· S2: {support_levels[1]:.2f} Â· S3: {support_levels[2]:.2f}"
    )
    st.markdown(
        f"**Resistance:** "
        f"R1: {resistance_levels[0]:.2f} Â· R2: {resistance_levels[1]:.2f} Â· R3: {resistance_levels[2]:.2f}"
    )
    st.markdown(f"**Current Price:** {price:.2f}")

with col3:
    st.subheader("Trade Setup")
    if market_bias == "BULLISH":
        # Use nearest support as anchor if available
        anchor = support_levels[0] if not np.isnan(support_levels[0]) else price - 20
        strike_short = int(anchor) - 20
        strike_long = strike_short - 50
        st.markdown("**Put Credit Spread (Bullish)**")
        st.markdown(f"Sell: {strike_short} Put")
        st.markdown(f"Buy: {strike_long} Put")
        st.markdown("Target Credit: $2.50â€“3.00")
        exit_ref = support_levels[1] if not np.isnan(support_levels[1]) else (price - 1.0 * (atr if not np.isnan(atr) else 50))
        st.markdown(f"Stop Loss: Exit if SPX < {exit_ref:.0f}")
    elif market_bias == "BEARISH":
        anchor = resistance_levels[0] if not np.isnan(resistance_levels[0]) else price + 20
        strike_short = int(anchor) + 20
        strike_long = strike_short + 50
        st.markdown("**Call Credit Spread (Bearish)**")
        st.markdown(f"Sell: {strike_short} Call")
        st.markdown(f"Buy: {strike_long} Call")
        st.markdown("Target Credit: $2.50â€“3.00")
        exit_ref = resistance_levels[1] if not np.isnan(resistance_levels[1]) else (price + 1.0 * (atr if not np.isnan(atr) else 50))
        st.markdown(f"Stop Loss: Exit if SPX > {exit_ref:.0f}")
    else:
        st.markdown("**Neutral**")
        st.markdown("Iron Condor / Butterfly centered near current price")

    st.subheader("Timing")
    st.markdown("**Entry Window:** 2:30â€“3:00 PM")
    st.markdown("**Exit:** Before 3:15 PM")

st.markdown("---")
st.caption(
    "This app refreshes as spx_data.csv updates. For live market data, connect to your broker or TradingView."
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Download (Code Snapshot)")
    # Package a lightweight snapshot so download_button has valid bytes
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README.txt", "Run: pip install -r requirements.txt\nThen: streamlit run main.py\n")
        # You can add more files here if needed (e.g., a sample CSV)
    mem_zip.seek(0)

    st.download_button(
        label="Download starter zip",
        data=mem_zip,
        file_name=f"spx_dashboard_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
        help="A minimal starter zip (README only). Replace with your packaged project if desired.",
        mime="application/zip",
    )

    st.markdown(
        """
**Quick Start**
- `pip install -r requirements.txt`
- `streamlit run main.py`
        """.strip()
    )