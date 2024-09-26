import pandas as pd
import numpy as np


def choose_position(roi, trade_threshold = 0.0005):
    pos =0
    # Predict position base on change in future
    if roi > trade_threshold:
        pos = 1
    elif roi < -trade_threshold:
        pos = -1
    else:
        pos = 0

    return pos

def backtest_position_ps(position, price, periods):
    # Shift positions to align with future price changes and handle NaN by filling with 0
    pos = pd.Series(position, index=pd.Series(price).index).shift(1).fillna(0)
    pos = pd.Series(pos).rolling(periods).sum() #pos for 10 hour predict

    price_array = pd.Series(price).shift(1).fillna(0)

    pos_diff = pos.diff()
    fee = pos_diff*price_array*0.05*0.01

    # Calculate price changes over the given periods
    ch = pd.Series(price) - price_array

    # Calculate total PnL
    total_pnl = pos*ch - fee
    return total_pnl

def returns(pnl):
    returns = pnl.pct_change().dropna()
    return returns

def sharpe_ratio(returns):
    std = np.std(returns) if np.std(returns) != 0 else 0.001
    sharpe_ratio = np.sqrt(252) * np.mean(returns) / std
    return sharpe_ratio

def sortino_ration(returns):
    downside_risk = np.std(returns[returns < 0])
    sortino_ratio = np.sqrt(252) * np.mean(returns) / downside_risk
    return sortino_ratio

def MDD(returns):
    cum_returns = (1 + returns).cumprod()
    mdd = (cum_returns.cummax() - cum_returns).max()
    return mdd
    
def calmar_ratio(returns, mdd):
    calmar_ratio = np.mean(returns) / mdd
    return calmar_ratio

def volatility(returns):
    volatility = np.std(returns) * np.sqrt(252)
    return volatility

def sharpe_for_vn30f(y_pred, y_price, trade_threshold, fee_perc, periods):

    # Predict position base on change in future
    pos = [choose_position(roi, trade_threshold) for roi in y_pred]
    pos = np.array(pos)

    # Calculate PNL
    pnl = backtest_position_ps(pos, y_price, fee_perc, periods)
    pnl = np.cumsum(pnl)

    # Standardalize PNL to date
    daily_pnl = [pnl.iloc[i] for i in range(0, len(pnl), 241)]
    daily_pnl = pd.Series(daily_pnl).fillna(0)

    # Calculate Sharpe
    sharpe = calculate_sharpe_ratio(daily_pnl)

    return pos, pnl, daily_pnl, sharpe