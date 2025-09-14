import pandas as pd
import numpy as np

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(prices, window):
    return prices.ewm(span=window).mean()

def calculate_macd(prices, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd = ema_fast - ema_slow
    signal_line = calculate_ema(macd, signal)
    return macd, signal_line

def calculate_vwap(df):
    return (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

def calculate_atr(df, window=14):
    df['tr'] = np.maximum(df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)),
                   abs(df['low'] - df['close'].shift(1))))
    atr = df['tr'].rolling(window=window).mean()
    return atr

def get_levels(df):
    high_5d = df.tail(5)['high'].max()
    low_5d = df.tail(5)['low'].min()
    high_10d = df.tail(10)['high'].max()
    low_10d = df.tail(10)['low'].min()
    return high_5d, low_5d, high_10d, low_10d
