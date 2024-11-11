import numpy as np
import pandas as pd

def cumulative_return(data):
    close = pd.Series(data['Close'])
    cr = (close / close.iloc[0] - 1) * 100  # Tính CR theo phần trăm
    return cr.to_numpy()

def daily_log_return(data):
    close = pd.Series(data['Close'])
    dlr = np.log(close / close.shift(1))  # Tính log return
    return dlr.to_numpy()

def daily_return(data):
    close = pd.Series(data['Close'])
    dr = (close / close.shift(1) - 1) * 100  # Tính return theo phần trăm
    return dr.to_numpy()
