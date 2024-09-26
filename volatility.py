import numpy as np
import pandas as pd

# 1. Average True Range (ATR)
def average_true_range(data, window=14):
    high = data['High']
    low = data['Low']
    close = data['Close'].shift(1)
    
    true_range = np.maximum(high - low, np.maximum(np.abs(high - close), np.abs(low - close)))
    atr = pd.Series(true_range).rolling(window=window).mean().to_numpy()
    
    return atr

# 2. Bollinger Bands
def bollinger_bands(data, window=20, multiplier=2):
    close = data['Close']
    sma = close.rolling(window=window).mean()
    stddev = close.rolling(window=window).std()
    
    upper_band = (sma + (multiplier * stddev)).to_numpy()
    lower_band = (sma - (multiplier * stddev)).to_numpy()
    
    return upper_band, lower_band

# 3. Donchian Channel
def donchian_channel(data, window=20):
    high = data['High']
    low = data['Low']
    
    upper_band = high.rolling(window=window).max().to_numpy()
    lower_band = low.rolling(window=window).min().to_numpy()
    
    return upper_band, lower_band

# 4. Keltner Channel
def keltner_channel(data, window=20, multiplier=2, atr_window=10):
    high = data['High']
    low = data['Low']
    close = data['Close']
    
    typical_price = (high + low + close) / 3
    ema = typical_price.ewm(span=window, adjust=False).mean()
    
    true_range = np.maximum(high - low, np.maximum(np.abs(high - close.shift(1)), np.abs(low - close.shift(1))))
    atr = pd.Series(true_range).rolling(window=atr_window).mean()
    
    upper_band = (ema + multiplier * atr).to_numpy()
    lower_band = (ema - multiplier * atr).to_numpy()
    
    return upper_band, lower_band

# 5. Ulcer Index (UI)
def ulcer_index(data, window=14):
    close = data['Close']
    max_close = close.rolling(window=window).max()
    
    drawdown = ((close - max_close) / max_close) * 100
    drawdown_squared = drawdown ** 2
    ulcer_idx = np.sqrt(drawdown_squared.rolling(window=window).mean()).to_numpy()
    
    return ulcer_idx

