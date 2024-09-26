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
    trade_fee = 1 + 2.7 
    overnight_fee = 2.55
    overnight_manage_fee = 320 if 0.000024*price < 320 else 1600 if 0.000024*price > 1600 else 0.000024*price
    
    # Calculate price changes over the given periods
    ch = pd.Series(price) - price_array

    # Calculate total PnL
    total_pnl = pos*ch - trade_fee
    return total_pnl

def backtest_position_ps_fee(position, price, periods):
    # Shift positions to align with future price changes and handle NaN by filling with 0
    pos = pd.Series(position, index=pd.Series(price).index).shift(1).fillna(0)
    pos = pd.Series(pos).rolling(periods).sum()  # pos for 10-hour prediction

    price_array = pd.Series(price).shift(1).fillna(0)

    pos_diff = pos.diff()
    trade_fee = 1 + 2.7  # Example trading fee per trade
    overnight_fee = 2.55  # Example overnight fee
    
    # Calculate price changes over the given periods
    ch = pd.Series(price) - price_array

    # Initialize total PnL
    total_pnl = pos * ch - trade_fee

    # Mark overnight trades (from 2:45 PM until 9 AM the next day)
    time_index = pd.Series(price).index
    end_of_day = time_index.indexer_between_time('14:45', '23:59')  # After 2:45 PM
    overnight_positions = pos.iloc[end_of_day]  # Positions held overnight

    # Apply overnight manage fee for overnight positions
    manage_fee = pd.Series(320 if 0.000024*price < 320 else 1600 if 0.000024*price > 1600 else 0.000024*price, index=pd.Series(price).index)

    # Apply overnight fee and manage fee to trades held overnight
    for i in overnight_positions.index:
        if pos.loc[i] != 0:
            total_pnl.loc[i] -= overnight_fee + manage_fee.loc[i]

    return total_pnl

def cal_returns(pnl):
    returns = pnl.pct_change().dropna()
    return returns

def cal_sharpe_ratio(returns):
    std = np.std(returns) if np.std(returns) != 0 else 0.001
    sharpe_ratio = np.sqrt(252) * np.mean(returns) / std
    return sharpe_ratio

def cal_sortino_ratio(returns):
    downside_risk = np.std(returns[returns < 0])
    sortino_ratio = np.sqrt(252) * np.mean(returns) / downside_risk
    return sortino_ratio

def cal_MDD(returns):
    cum_returns = (1 + returns).cumprod()
    mdd = (cum_returns.cummax() - cum_returns).max()
    return mdd
    
def cal_calmar_ratio(returns, mdd):
    calmar_ratio = np.mean(returns) / mdd
    return calmar_ratio

def cal_volatility(returns):
    volatility = np.std(returns) * np.sqrt(252)
    return volatility

def metrics_for_vn30f(y_pred, y_price, trade_threshold, fee_perc, periods):

    # Predict position base on change in future
    pos = [choose_position(roi, trade_threshold) for roi in y_pred]
    pos = np.array(pos)

    # Calculate PNL
    pnl = backtest_position_ps(pos, y_price, fee_perc, periods)
    pnl = np.cumsum(pnl)

    # Standardalize PNL to date
    daily_pnl = [pnl.iloc[i] for i in range(0, len(pnl), 241)]
    daily_pnl = pd.Series(daily_pnl).fillna(0)

    returns = cal_returns(daily_pnl)
    sharpe_ratio = cal_sharpe_ratio(returns)
    sortino_ratio = cal_sortino_ratio(returns)
    mdd = cal_MDD(returns)
    calmar_ratio = cal_calmar_ratio(returns, mdd)
    volatility = cal_volatility(returns)


    return pos, pnl, daily_pnl, returns, sharpe_ratio, sortino_ratio, calmar_ratio, mdd, volatility
