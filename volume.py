import pandas as pd
import numpy as np


#1. Accumulation/Distribution Index (ADI)
def acc_dist_index(data):
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])
    close = pd.Series(data['Close'])
    volume = pd.Series(data['Volume'])
    
    clv = ((close - low) - (high - close)) / (high - low)  # Close Location Value
    adi = (clv * volume).cumsum()  # ADI calculation
    return adi.values  # Convert to numpy array

# 2. Chaikin Money Flow (CMF)
def haikin_money_flow(data, window=20):
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])
    close = pd.Series(data['Close'])
    volume = pd.Series(data['Volume'])
    
    mfv = ((close - low) - (high - close)) / (high - low) * volume  # Money Flow Volume
    cmf = mfv.rolling(window=window).sum() / volume.rolling(window=window).sum()  # CMF calculation
    return cmf.values  # Convert to numpy array

# 3. Ease of Movement (EoM)
def ease_of_movement(data, window=14):
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])
    volume = pd.Series(data['Volume'])
    
    distance_moved = (high + low) / 2 - (high.shift(1) + low.shift(1)) / 2
    box_ratio = volume / (high - low)
    eom = distance_moved / box_ratio
    eom_sma = eom.rolling(window=window).mean()  # Smoothed EoM
    return eom_sma.values  # Convert to numpy array

# 4. Force Index (FI)
def force_index(data, window=13):
    close = pd.Series(data['Close'])
    volume = pd.Series(data['Volume'])
    
    fi = (close - close.shift(1)) * volume
    fi_sma = fi.rolling(window=window).mean()  # Smoothed Force Index
    return fi_sma.values  # Convert to numpy array

# 5. Money Flow Index (MFI)
def money_flow_index(data, window=14):
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])
    close = pd.Series(data['Close'])
    volume = pd.Series(data['Volume'])
    
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
    
    positive_flow_sum = pd.Series(positive_flow).rolling(window=window).sum()
    negative_flow_sum = pd.Series(negative_flow).rolling(window=window).sum()
    
    money_flow_ratio = positive_flow_sum / negative_flow_sum
    mfi = 100 - (100 / (1 + money_flow_ratio))  # MFI calculation
    return mfi.values  # Convert to numpy array

# 6. Negative Volume Index (NVI)
def negative_volume_index(data):
    close = pd.Series(data['Close'])
    volume = pd.Series(data['Volume'])
    
    nvi = np.ones(len(volume)) * 1000  # Initialize NVI with 1000
    for i in range(1, len(volume)):
        if volume[i] < volume[i - 1]:
            nvi[i] = nvi[i - 1] * (1 + (close[i] - close[i - 1]) / close[i - 1])
        else:
            nvi[i] = nvi[i - 1]
    return nvi  # Return numpy array

# 7. On-Balance Volume (OBV)
def on_balance_volume(data):
    close = pd.Series(data['Close'])
    volume = pd.Series(data['Volume'])
    
    obv = np.zeros(len(volume))
    for i in range(1, len(volume)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return obv  # Return numpy array

# 8. Volume-Price Trend (VPT)
def volume_price_trend(data):
    close = pd.Series(data['Close'])
    volume = pd.Series(data['Volume'])
    
    vpt = np.zeros(len(volume))
    for i in range(1, len(volume)):
        vpt[i] = vpt[i - 1] + volume[i] * ((close[i] - close[i - 1]) / close[i - 1])
    return vpt  # Return numpy array

# 9. Volume Weighted Average Price (VWAP)
def volume_weighted_average_price(data, window=14):
    high = pd.Series(data['High'])
    low = pd.Series(data['Low'])
    close = pd.Series(data['Close'])
    volume = pd.Series(data['Volume'])
    
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
    return vwap.values  # Convert to numpy array

