import pandas as pd
import numpy as np

# 1. Average Directional Index (ADX)
def adx(data, window=14):
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])
    close = pd.Series(data['Close'])

    # Calculate UpMove and DownMove
    up_move = high.diff()
    down_move = low.diff().abs()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Calculate True Range (TR)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=window).mean()

    # Calculate Directional Indicators
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=window).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=window).mean() / atr)

    # Calculate ADX
    adx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di)).rolling(window=window).mean()

    return adx.values

# 2. Aroon Indicator
def aroon(data, window=25):
    close = pd.Series(data['Close'])
    
    aroon_up = 100 * (window - close.rolling(window).apply(lambda x: x.argmax(), raw=True)) / window
    aroon_down = 100 * (window - close.rolling(window).apply(lambda x: x.argmin(), raw=True)) / window
    return aroon_up.values, aroon_down.values

# 3. Commodity Channel Index (CCI)
def cci(data, window=20, constant=0.015):
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])
    close = pd.Series(data['Close'])
    
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=window).mean()
    mean_dev = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
    
    cci = (tp - sma_tp) / (constant * mean_dev)
    return cci.values

# 4. Detrended Price Oscillator (DPO)
def dpo(data, window=20):
    close = pd.Series(data['Close'])
    sma = close.rolling(window=window).mean()
    dpo = close.shift(int(window / 2 + 1)) - sma
    return dpo.values

# 5. Exponential Moving Average (EMA)
def ema(data, window=12):
    close = pd.Series(data['Close'])
    ema = close.ewm(span=window, adjust=False).mean()
    return ema.values

# 6. Simple Moving Average (SMA)
def sma(data, window=20):
    close = pd.Series(data['Close'])
    sma = close.rolling(window=window).mean()
    return sma.values

# 7. Parabolic SAR (PSAR)
def psar(data, step=0.02, max_step=0.2):
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])
    close = pd.Series(data['Close'])
    
    psar = close.copy()  # Placeholder
    af = step
    ep = high[0] if close[1] > close[0] else low[0]
    uptrend = True
    for i in range(1, len(close)):
        if uptrend:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)
            if low[i] < psar[i]:
                uptrend = False
                af = step
                ep = low[i]
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)
            if high[i] > psar[i]:
                uptrend = True
                af = step
                ep = high[i]
    return psar.values

# 8. Trix (TRIX)
def trix(data, window=15):
    close = pd.Series(data['Close'])
    ema1 = close.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    
    trix = 100 * ema3.pct_change()
    return trix.values

# 9. Ichimoku Kinkō Hyō
def ichimoku(data, window1=9, window2=26, window3=52):
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])
    
    tenkan_sen = (high.rolling(window=window1).max() + low.rolling(window=window1).min()) / 2
    kijun_sen = (high.rolling(window=window2).max() + low.rolling(window=window2).min()) / 2
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(window2)
    senkou_span_b = ((high.rolling(window=window3).max() + low.rolling(window=window3).min()) / 2).shift(window2)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values

# 10. Vortex Indicator (VI)
def vortex(data, window=14):
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])
    close = pd.Series(data['Close'])
    
    vm_plus = (high.diff(1).abs())
    vm_minus = (low.diff(1).abs())
    
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    tr14 = tr.rolling(window=window).sum()
    
    vi_plus = vm_plus.rolling(window=window).sum() / tr14
    vi_minus = vm_minus.rolling(window=window).sum() / tr14
    
    return vi_plus.values, vi_minus.values

# 11. KST (Know Sure Thing)
def kst(data, roc1=10, roc2=15, roc3=20, roc4=30, sma1=10, sma2=10, sma3=10, sma4=15):
    close = pd.Series(data['Close'])
    
    rcma1 = close.pct_change(roc1).rolling(sma1).mean()
    rcma2 = close.pct_change(roc2).rolling(sma2).mean()
    rcma3 = close.pct_change(roc3).rolling(sma3).mean()
    rcma4 = close.pct_change(roc4).rolling(sma4).mean()
    
    kst = (rcma1 * 1) + (rcma2 * 2) + (rcma3 * 3) + (rcma4 * 4)
    return kst.values

# 12. Mass Index
def mass_index(data, window_fast=9, window_slow=25):
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])
    
    hl_diff = high - low
    single_ema = hl_diff.ewm(span=window_fast, adjust=False).mean()
    double_ema = single_ema.ewm(span=window_fast, adjust=False).mean()
    
    ema_ratio = single_ema / double_ema
    mass_index = ema_ratio.rolling(window=window_slow).sum()
    
    return mass_index.values

# 13. Schaff Trend Cycle (STC)
def stc(data, window_fast=23, window_slow=50, cycle=10, smooth1=3, smooth2=3):
    close = pd.Series(data['Close'])
    
    ema_fast = close.ewm(span=window_fast, adjust=False).mean()
    ema_slow = close.ewm(span=window_slow, adjust=False).mean()
    
    macd = ema_fast - ema_slow
    k_macd = macd.rolling(window=cycle).apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    d_macd = k_macd.rolling(window=smooth1).mean()
    
    stc = d_macd.rolling(window=smooth2).mean()
    return stc.values

# 14. TRIX
def trix(data, window=15):
    close = pd.Series(data['Close'])
    ema1 = close.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    
    trix = 100 * ema3.pct_change()
    return trix.values

# 15. Weighted Moving Average (WMA)
def wma(data, window=9):
    close = pd.Series(data['Close'])
    
    weights = np.arange(1, window + 1)
    wma = close.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    
    return wma.values
