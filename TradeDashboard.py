import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ta_utils import (
    calculate_rsi,
    calculate_ema,
    calculate_macd,
    calculate_vwap,
    calculate_atr,
    get_levels,
)

# 1. Custom CSS (Color, cards, hover)
st.markdown("""
<style>
body, .main, .block-container {
  background-color: #181c25 !important;
}
.card {
  background: linear-gradient(120deg,#212433 60%, #26293d 100%);
  border-radius: 16px;
  box-shadow: 0 2px 14px rgba(44, 164, 234, 0.09);
  padding: 1.25em 1em 1em 1em;
  margin-bottom: 1.2em;
  border: 2px solid #252738;
  transition: box-shadow 0.3s, border-color 0.3s;
}
.card:hover {
  box-shadow: 0 6px 38px rgba(44,164,234, 0.16);
  border-color: #2ca4ea;
}
.card-title {
  font-size: 1.35em;
  font-family: 'Montserrat',sans-serif;
  font-weight: 700;
  color: #2ca4ea;
  margin-bottom: .36em;
  letter-spacing: 0.5px;
}
.card-kv {
  font-size: 1.03em;
  font-weight: 400;
  color: #e9ecf0;
  margin: 0.08em 0;
  font-family: 'Montserrat',sans-serif;
}
.card-kv b, .card-kv strong {
  color: #F7F7F7;
  font-weight: bold;
}
.card-bias {
  font-weight: 500;
  margin-top: 0.78em;
  font-size: 1.06em;
  font-family: 'Montserrat',sans-serif;
  letter-spacing: 0.5px;
}
.card-bias.bullish {
  color: #15be53;
}
.card-bias.bearish {
  color: #ff163d;
}
.card-bias.neutral {
  color: #eeb100;
}
.stButton>button {
  border-radius: 7px !important;
  background: linear-gradient(90deg,#2ca4ea 40%,#3794c9 70%);
  color: #fff !important;
  border: none;
  padding: 0.42em 1.25em !important;
  font-weight: 600;
  font-size: 1.01em;
  margin-top: 0.8em;
  box-shadow: 0 3px 14px rgba(44, 164, 234, 0.11);
  transition: background 0.17s;
}
.stButton>button:hover {
  background: linear-gradient(90deg,#3794c9 50%,#2ca4ea 80%);
  color: #eeb100 !important;
}
</style>
""", unsafe_allow_html=True)


# 2. Symbols
SYMBOLS = {
    'ES1': {'file': 'es1_data.csv', 'time_col': 'time'},
    'NQ1': {'file': 'nq1_data.csv', 'time_col': 'time'},
    'SPX': {'file': 'spx_data.csv', 'time_col': 'date'},
    'SPY': {'file': 'spy_data.csv', 'time_col': 'time'},
    'QQQ': {'file': 'qqq_data.csv', 'time_col': 'time'},
    'TQQQ': {'file': 'tqqq_data.csv', 'time_col': 'time'},
    'SQQQ': {'file': 'sqqq_data.csv', 'time_col': 'time'},
    'TSLA': {'file': 'tsla_data.csv', 'time_col': 'time'},
    'NVDA': {'file': 'nvda_data.csv', 'time_col': 'time'},
    'AAPL': {'file': 'aapl_data.csv', 'time_col': 'time'},
    'GOOG': {'file': 'goog_data.csv', 'time_col': 'time'},
    'MSFT': {'file': 'msft_data.csv', 'time_col': 'time'},
    'BRK.B': {'file': 'brkb_data.csv', 'time_col': 'time'}
   }

# 3. Data loading
def load_symbol(path, time_col):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col).reset_index(drop=True)
    for c in ['high','low','open']:
        if c not in df.columns:
            df[c] = df['close']
    if 'volume' not in df.columns:
        df['volume'] = np.nan
    df['rsi'] = calculate_rsi(df['close'])
    df['ema_20'] = calculate_ema(df['close'], 20)
    df['ema_50'] = calculate_ema(df['close'], 50)
    df['vwap'] = calculate_vwap(df)
    df['atr'] = calculate_atr(df)
    macd, macd_signal = calculate_macd(df['close'])
    df['macd'] = macd
    df['macd_signal'] = macd_signal
    return df

st.set_page_config(page_title="Trade Dashboard", page_icon="📈", layout="wide")

st.title("✨ Trade Technical Dashboard")

# 4. Session state
if 'selected_symbol' not in st.session_state:
    st.session_state['selected_symbol'] = None

data = {}
for sym, spec in SYMBOLS.items():
    try:
        data[sym] = load_symbol(spec['file'], spec['time_col'])
    except Exception:
        data[sym] = None

def select_symbol(symbol):
    st.session_state['selected_symbol'] = symbol


def chunk_symbols(symbols, n):
    """Yield successive n-sized chunk lists from symbols."""
    symbols = list(symbols)
    for i in range(0, len(symbols), n):
        yield symbols[i:i + n]

if st.session_state['selected_symbol'] is None:
    st.header("Markets Overview")
    cards_per_row = 5
    for symbol_chunk in chunk_symbols(SYMBOLS, cards_per_row):
        cols = st.columns(len(symbol_chunk))
        for i, sym in enumerate(symbol_chunk):
            df = data[sym]
            with cols[i]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<div class="card-title">{sym}</div>', unsafe_allow_html=True)
                if df is not None:
                    latest = df.iloc[-1]
                    st.markdown(f'<div class="card-kv">Close: <b>{latest["close"]:.2f}</b></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-kv">RSI: {latest["rsi"]:.1f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-kv">ATR: {latest["atr"]:.2f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="card-kv">Volume: {latest.get("volume", np.nan):,.0f}</div>', unsafe_allow_html=True)
                    bias = "BULLISH" if latest["rsi"] > 55 else ("BEARISH" if latest["rsi"] < 45 else "NEUTRAL")
                    bias_color = "#23954c" if bias == "BULLISH" else "#ba004e" if bias == "BEARISH" else "#767676"
                    st.markdown(f'<div class="card-bias" style="color:{bias_color}">Bias: {bias}</div>', unsafe_allow_html=True)
                    st.button(
                        "View Details", key=f"btn_{sym}",
                        on_click=select_symbol, args=(sym,)
                    )
                else:
                    st.markdown('<div class="card-kv">No data loaded.</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

else:
    sym = st.session_state['selected_symbol']
    df = data[sym]
    latest = df.iloc[-1]
    spec = SYMBOLS[sym]

    st.header(f"🔎 {sym} Details")
    st.markdown(f"#### Latest Price Data")
    st.markdown(f"**Close:** {latest['close']:.2f}")
    st.markdown(f"**RSI:** {latest['rsi']:.2f}")
    st.markdown(f"**MACD:** {latest['macd']:.2f} vs Signal {latest['macd_signal']:.2f}")
    st.markdown(f"**EMA 20/50:** {latest['ema_20']:.2f} / {latest['ema_50']:.2f}")
    st.markdown(f"**VWAP:** {latest['vwap']:.2f}")
    st.markdown(f"**ATR:** {latest['atr']:.2f}")
    st.markdown(f"**Volume:** {latest.get('volume', np.nan):,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[spec['time_col']], y=df["close"], name="Close"))
    fig.add_trace(go.Scatter(x=df[spec['time_col']], y=df["ema_20"], name="EMA 20", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=df[spec['time_col']], y=df["ema_50"], name="EMA 50", line=dict(dash="dot")))
    if not df["vwap"].isna().all():
        fig.add_trace(go.Scatter(x=df[spec['time_col']], y=df["vwap"], name="VWAP", line=dict(dash="dash")))
    fig.update_layout(
        title=f"{sym} Price & Key Moving Averages",
        xaxis_title="Date", yaxis_title="Price",
        plot_bgcolor="#f9fbff",
        paper_bgcolor="#f9fbff",
        font=dict(color="#203040"),
        legend=dict(bgcolor="#f9fbff"),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Levels")
    high_5d, low_5d, high_10d, low_10d = get_levels(df)
    st.markdown(f"**Support:** S1: {low_5d:.2f} — S2: {low_10d:.2f}")
    st.markdown(f"**Resistance:** R1: {high_5d:.2f} — R2: {high_10d:.2f}")

    st.button("⬅ Back to Dashboard", key=f"back_{sym}", on_click=lambda: select_symbol(None))

