import numpy as np
import pandas as pd

def awesome_oscillator(data, window1, window2):
    high = data['High']
    low = data['Low']
    median_price = (high + low) / 2

    # Tính SMA cho 5 và 34 kỳ
    sma_5 = pd.Series(median_price).rolling(window=window1).mean()
    sma_34 = pd.Series(median_price).rolling(window=window2).mean()

    # AO = SMA(5) - SMA(34)
    ao = sma_5 - sma_34
    return np.array(ao)

# Hàm tính Kaufman's Adaptive Moving Average (KAMA), trả về dạng numpy array
def kaufman_adaptive_moving_average(data, period=10, fast_period=2, slow_period=30):
    close = pd.Series(data['Close'])
    
    change = close.diff(period)
    volatility = close.diff(1).abs().rolling(window=period).sum()

    efficiency_ratio = (change.abs() / volatility).to_numpy()
    smoothing_constant = (efficiency_ratio * (2 / (fast_period + 1) - 2 / (slow_period + 1)) + 2 / (slow_period + 1)) ** 2

    kama = [close.iloc[0]]  # Khởi tạo KAMA với giá trị đầu tiên
    for i in range(1, len(close)):
        kama.append(kama[-1] + smoothing_constant[i] * (close.iloc[i] - kama[-1]))

    return np.array(kama)

