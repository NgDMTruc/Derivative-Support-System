import numpy as np
import pandas as pd

def awesome_oscillator(data, window1=5, window2=34):
    high = data['High']
    low = data['Low']
    median_price = (high + low) / 2

    # Tính SMA cho 5 và 34 kỳ
    sma_5 = pd.Series(median_price).rolling(window=window1).mean()
    sma_34 = pd.Series(median_price).rolling(window=window2).mean()

    # AO = SMA(5) - SMA(34)
    ao = sma_5 - sma_34
    return np.array(ao)

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

def percentage_price_oscillator(data, fast_period=12, slow_period=26):
    close = pd.Series(data['Close'])
    ppo = ((close.ewm(span=fast_period).mean() - close.ewm(span=slow_period).mean()) / close.ewm(span=slow_period).mean()) * 100
    return ppo.to_numpy()

def percentage_volume_oscillator(data, fast_period=14, slow_period=28):
    volume = pd.Series(data['Volume'])
    pvo = ((volume.ewm(span=fast_period).mean() - volume.ewm(span=slow_period).mean()) / volume.ewm(span=slow_period).mean()) * 100
    return pvo.to_numpy()

def rate_of_change(data, period=14):
    close = pd.Series(data['Close'])
    roc = ((close.diff(period) / close.shift(period)) * 100)
    return roc.to_numpy()

def relative_strength_index(data, period=14):
    close = pd.Series(data['Close'])
    delta = close.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.to_numpy()

def stochastic_oscillator(data, k_period=14, d_period=3):
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])
    close = pd.Series(data['Close'])
    
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k = 100 * ((close - lowest_low) / (highest_high - lowest_low)).fillna(0)
    d = k.rolling(window=d_period).mean()

    return k.to_numpy(), d.to_numpy()  # trả về cả %K và %D

def true_strength_index(data, long_period=25, short_period=13):
    close = pd.Series(data['Close'])
    delta = close.diff()

    # Tính EMA cho các độ biến đổi
    ema1 = delta.ewm(span=long_period).mean()
    double_smoothed_pc  = ema1.ewm(span=short_period).mean()
    
    ema2 = abs(delta).ewm(span=long_period).mean()
    double_smoothed_abs_pc  = ema2.ewm(span=short_period).mean()

    # TSI
    tsi = 100 * (double_smoothed_pc / double_smoothed_abs_pc)
    
    return tsi.to_numpy()

def ultimate_oscillator(data, short_period=7, medium_period=14, long_period=28):
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])
    close = pd.Series(data['Close'])

    bp = close - np.minimum(low, close.shift(1))  # Buying Pressure
    tr = np.maximum(high, close.shift(1)) - np.minimum(low, close.shift(1))  # True Range
    
    avg_short = bp.rolling(short_period).sum() / tr.rolling(short_period).sum()
    avg_medium = bp.rolling(medium_period).sum() / tr.rolling(medium_period).sum()
    avg_long = bp.rolling(long_period).sum() / tr.rolling(long_period).sum()

    ultimate_osc = (4 * avg_short + 2 * avg_medium + avg_long) / (4 + 2 + 1) * 100
    return ultimate_osc.to_numpy()

def williams_r(data, period=14):
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])
    close = pd.Series(data['Close'])

    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    wr = -100 * ((highest_high - close) / (highest_high - lowest_low)).fillna(0)
    
    return wr.to_numpy()

def stochastic_oscillator_signal(data, k_period=14, d_period=3):
    k, d = stochastic_oscillator(data, k_period, d_period)  # Sử dụng hàm stochastic_oscillator
    signal = np.where(k > d, 1, 0)  # Tín hiệu mua khi %K cắt lên %D
    return signal

def stochastic_rsi(data, rsi_period=14, stoch_period=14):

    rsi = relative_strength_index(data, period=rsi_period)  # Sử dụng    close = pd.Series(data['Close']) hàm RSI đã có

    # Tính toán %K cho Stochastic RSI
    rsi_high = pd.Series(rsi).rolling(window=stoch_period).max()
    rsi_low = pd.Series(rsi).rolling(window=stoch_period).min()

    stoch_rsi = 100 * (rsi - rsi_low) / (rsi_high - rsi_low)
    
    return stoch_rsi.to_numpy()

def stochastic_rsi_d(data, rsi_period=14, stoch_period=14, d_period=3):
    stoch_rsi_values = stochastic_rsi(data, rsi_period, stoch_period)
    stoch_rsi_d = pd.Series(stoch_rsi_values).rolling(window=d_period).mean()
    
    return stoch_rsi_d.to_numpy()
