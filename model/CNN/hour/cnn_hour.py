
from datetime import datetime, timedelta, timezone

import pandas as pd

import numpy as np

import requests

import os

import logging

import sys

import optuna

import joblib

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import LSTM, Dropout, Dense

from sklearn.preprocessing import StandardScaler

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import save_model, load_model

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout,Input

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from datetime import datetime, timedelta

from ta import add_all_ta_features

import pandas_ta as ta

import matplotlib.pyplot as plt

import matplotlib

"""## Formulas"""

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

def backtest_position_ps(position, price, percentage, periods):

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

def calculate_sharpe_ratio(pnl):

    pnl = np.diff(pnl)

    std = np.std(pnl) if np.std(pnl) != 0 else 0.001

    sharpe = np.mean(pnl)/std*np.sqrt(252)

    return sharpe

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

def calculate_hitrate(pos_predict, pos_true):

    if len(pos_predict) != len(pos_true):

        raise ValueError("Độ dài của hai mảng không khớp")



    # Tính số lượng dự đoán đúng (các phần tử tương ứng giống nhau)

    correct_predictions = np.sum(pos_predict == pos_true)



    # Tính tỷ lệ hit rate

    hit_rate_value = correct_predictions / len(pos_predict)



    return hit_rate_value

"""# Function for data"""

def scale_data(data,fit):

    index = data.index

    scaler = StandardScaler()

#     data = np.where(np.isinf(data), np.nan, data

    data.replace([np.inf, -np.inf], 0, inplace=True)

    data.fillna(0, inplace=True)

    fit.replace([np.inf, -np.inf], 0, inplace=True)

    fit.fillna(0, inplace=True)

    data = pd.DataFrame(data,index=index)

    data = data.fillna(0)

    scaler.fit(fit)

    data=pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)



    return data

def split_data(data):

    """

    Hàm này chia dữ liệu thành 2 phần: tập huấn luyện và tập hold out.



    Args:

    data (pandas.DataFrame): DataFrame chứa dữ liệu cần chia.



    Returns:

    pandas.DataFrame: DataFrame chứa dữ liệu tập huấn luyện.

    pandas.DataFrame: DataFrame chứa dữ liệu tập giữ lại.

    """

    # Chia dữ liệu thành 3 phần

    new_part = np.array_split(data, 3)



    # Access each part individually

    hold_out = new_part[2]

    train_data = pd.concat([new_part[0], new_part[1]], axis=0)



    return train_data, hold_out



def split_optuna_data(data):

    """

    Hàm này chia dữ liệu thành các tập train và test để sử dụng trong quá trình tối ưu hóa bằng Optuna.

​

    Args:

    data (pandas.DataFrame): DataFrame chứa dữ liệu cần chia.

​

    Returns:

    pandas.DataFrame: DataFrame chứa dữ liệu train (đã được chuẩn hóa).

    pandas.DataFrame: DataFrame chứa dữ liệu test (đã được chuẩn hóa).

    pandas.Series: Series chứa nhãn tương ứng với dữ liệu train.

    pandas.Series: Series chứa nhãn tương ứng với dữ liệu test.

    """



    # Chia dữ liệu thành tập train và tập hold out

    train_data, _ = split_data(data)



    # Loại bỏ các cột không cần thiết



    optuna_data = train_data.drop(['Open','High','Low','Close','Volume', 'Return','Unnamed: 0'], axis=1)

    rename_dict = {col: i for i, col in enumerate(optuna_data.columns)}

    optuna_data=optuna_data.rename(columns=rename_dict)

    # Chuẩn hóa dữ liệu





    # Chia dữ liệu thành tập train và tập test

    X_train, X_valid, y_train, y_valid = train_test_split(optuna_data, train_data['Return'], test_size=0.5, shuffle=False)





    # Lấy các dòng tương ứng từ train_data và train_data['Return']


    # Đặt cột 'datetime' làm chỉ số

    train_data.set_index('Unnamed: 0', inplace=True)

    train_data=train_data.rename(columns=rename_dict)

    train_data.index.name = 'datetime'

    train_data.index = pd.to_datetime(train_data.index)





    train_data_X_train = train_data.iloc[X_train.index]

    train_data_X_valid = train_data.iloc[X_valid.index]

    train_data_y_train = train_data.iloc[y_train.index, -1]

    train_data_y_valid = train_data.iloc[y_valid.index, -1]





    X_train=scale_data(X_train,X_train)

    X_valid=scale_data(X_valid,X_train)



    temp = train_data_X_train.drop(['Open','High','Low','Close','Volume', 'Return'], axis=1)

    temp=scale_data(temp,X_train)

    train_data_X_train= pd.concat([train_data_X_train[[ 'Open','High','Low','Close','Volume', 'Return']], temp], axis=1)



    temp = train_data_X_valid.drop(['Open','High','Low','Close','Volume', 'Return'], axis=1)

    temp=scale_data(temp,X_train)

    train_data_X_valid= pd.concat([train_data_X_valid[[ 'Open','High','Low','Close','Volume', 'Return']], temp], axis=1)

#     return X_train, X_valid, y_train, y_valid, train_data

    # Thay thế các giá trị infinity hoặc NaN thành 0





    return X_train, X_valid, y_train, y_valid, train_data,train_data_X_train, train_data_X_valid, train_data_y_train, train_data_y_valid

def backtest_data(data):

    train_data = data.copy()



    # Chuyển đổi cột 'Date' và 'time' thành một cột datetime

    train_data['datetime'] = pd.to_datetime(train_data['Date'] + ' ' + train_data['time'])



    # Đặt cột 'datetime' làm chỉ số

    train_data.set_index('datetime', inplace=True)



    # Xóa cột 'Date' và 'time' nếu không cần thiết nữa

    train_data.drop(columns=['Date', 'time'], inplace=True)



    return train_data

"""## Get data"""

data = pd.read_csv('final_hour.csv')

data =  data.fillna(0)

try:

    data['Unnamed: 0'] = pd.to_datetime(data['Date'] + ' ' + data['time'])

    data = data.drop(columns=['Date', 'time'])

except:

    pass

"""# Backtesting class

"""

from backtesting import Backtest, Strategy
import numpy as np
import pandas as pd

class ModelBasedStrategy(Strategy):
    trade_threshold = 0
    feat = []  # Khai báo biến lớp feat
    model = None  # Khai báo biến lớp model
    point_value = 1  # Giá trị mỗi điểm
    forecast_steps = 3  # Số bước dự báo
    take_profit = 0.02  # Ngưỡng chốt lời (2%)
    stop_loss = 0.02  # Ngưỡng cắt lỗ (2%)
    trailing_stop_pct = 0.02  # Trailing stop 2% (2% dưới mức giá cao nhất đạt được)

    def init(self):
        # Lưu trữ dự báo tín hiệu mua/bán từ mô hình
        self.pred = self.I(self.predict_model, self.data.Close)
        self.pos = self.calculate_positions(self.pred[0])
        self.i = 0
        self.entry_price = None  # Giá vào lệnh
        self.highest_price = None  # Lưu trữ giá cao nhất đạt được trong quá trình giữ vị thế

    def next(self):
        if self.i < len(self.pos):
            current_pos = self.pos[self.i]
            self.i += 1
            current_price = self.data.Close[-1] * self.point_value

            # Kiểm tra nếu không có vị thế mở
            if not self.position:
                if current_pos == 1:
                    self.buy(size=0.4)
                    self.entry_price = current_price
                    self.highest_price = current_price  # Thiết lập giá cao nhất ban đầu
                elif current_pos == -1:
                    self.sell(size=0.4)
                    self.entry_price = current_price
                    self.highest_price = current_price  # Thiết lập giá cao nhất ban đầu

            # Nếu đã có vị thế mở, kiểm tra chốt lời/cắt lỗ và trailing stop
            else:
                # Cập nhật giá cao nhất (hoặc thấp nhất nếu bán) kể từ khi mở vị thế
                if self.position.is_long:
                    self.highest_price = max(self.highest_price, current_price)
                else:
                    self.highest_price = min(self.highest_price, current_price)

                # Tính tỷ suất sinh lời hiện tại
                if self.position.is_long:
                    tssl = (current_price - self.entry_price) / self.entry_price
                else:
                    tssl = (self.entry_price - current_price) / self.entry_price

                # Đóng vị thế nếu đạt mức TP hoặc SL
                if tssl >= self.take_profit or tssl <= -self.stop_loss:
                    self.position.close()
                    self.entry_price = None
                    self.highest_price = None
                    return

                # Tính trailing stop dựa trên giá cao nhất đạt được
                trailing_stop = self.highest_price * (1 - self.trailing_stop_pct) if self.position.is_long else self.highest_price * (1 + self.trailing_stop_pct)
# Đóng vị thế nếu giá đi ngược lại mức trailing stop
                if (self.position.is_long and current_price <= trailing_stop) or (self.position.is_short and current_price >= trailing_stop):
                    self.position.close()
                    self.entry_price = None
                    self.highest_price = None
                    return

                # Lấy dự báo cho các bước tiếp theo
                future_positions = self.pos[self.i:self.i + self.forecast_steps]

                # Kiểm tra tín hiệu đảo chiều trong 2 bước tới (thay vì 1 bước)
                if self.check_reversal(future_positions, current_pos):
                    self.position.close()  # Đóng vị thế hiện tại
                    if current_pos == 1:
                        self.buy(size=0.4)
                    elif current_pos == -1:
                        self.sell(size=0.4)
                    self.entry_price = current_price
                    self.highest_price = current_price  # Thiết lập giá cao nhất mới cho vị thế tiếp theo

    def predict_model(self, price):
        # Dự báo từ mô hình đã cung cấp dựa trên các đặc trưng hiện tại
        test_data = self.data.df[self.feat]  # Sử dụng feat trong predict_model
        pred = self.model.predict(test_data)  # Sử dụng model để dự báo
        return pred

    def calculate_positions(self, predictions):
        # Xác định tín hiệu mua/bán dựa trên ngưỡng dự đoán
        return [choose_position(roi, self.trade_threshold) for roi in predictions]

    def check_reversal(self, future_positions, current_pos):
        """
        Kiểm tra tín hiệu đảo chiều:
        - Đảm bảo tín hiệu thay đổi trong 2 bước liên tiếp mới đảo chiều.
        """
        # Nếu hiện tại đang mua mà có tín hiệu bán trong 2 bước tiếp theo (hoặc ngược lại)
        if current_pos == 1 and -1 in future_positions[:2]:  # Kiểm tra 2 bước tiếp theo
            return True
        elif current_pos == -1 and 1 in future_positions[:2]:  # Kiểm tra 2 bước tiếp theo
            return True
        return False

def choose_position(roi, trade_threshold=0):
    # Hàm đơn giản để xác định tín hiệu mua/bán
    if roi > trade_threshold:
        return 1  # Tín hiệu mua
    elif roi < -trade_threshold:
        return -1  # Tín hiệu bán
    else:
        return 0  # Không mở vị thế
from scipy.stats import gmean
import numpy as np
import pandas as pd
from backtesting import Backtest

class CustomBacktest(Backtest):
    def __init__(self, *args, cash=1000, margin=0.13, **kwargs):
        super().__init__(*args, cash=cash, margin=margin, **kwargs)
        self._initial_cash = cash
        self.margin = margin  # Tỷ lệ margin

    def calculate_total_fees(self, entry_price, exit_price, size, entry_day, exit_day):
        platform_fee = 2000 * size
        exchange_fee = 2700 * size
        overnight_days = max(0, (exit_day - entry_day).days)
        overnight_fee = 2000 * overnight_days * size
        personal_income_tax = (exit_price * 100000 * self.margin) * 0.1 / 100 / 2 * size
        total_fee = (platform_fee + exchange_fee + overnight_fee + personal_income_tax) / 100000
        return total_fee

    def geometric_mean(self, returns: pd.Series) -> float:
        returns = returns.fillna(0) + 1
        if np.any(returns <= 0):
            return 0
        return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1

    def run(self, **strategy_args):
        stats = super().run(**strategy_args)
        fees_for_trades = []
        margin_required_list = []
        current_equity_list = []

        current_equity = self._initial_cash

        for idx, trade in stats._trades.iterrows():
            entry_price = trade['EntryPrice']
            exit_price = trade['ExitPrice']
            size = abs(trade['Size'])
            entry_day = trade['EntryTime'].date()
            exit_day = trade['ExitTime'].date()

            # Tính ký quỹ cần thiết cho giao dịch
            margin_required = entry_price * self.margin
            margin_required_list.append(margin_required)

            # Kiểm tra nếu equity >= margin_required
            if current_equity < margin_required:
                print(f"Không đủ ký quỹ cho giao dịch tại {idx}. Equity hiện tại: {current_equity}, Ký quỹ yêu cầu: {margin_required}")
                continue

            # Tính phí cho giao dịch này
            total_fee = self.calculate_total_fees(entry_price, exit_price, size, entry_day, exit_day)
            fees_for_trades.append(total_fee)

            # Cập nhật vốn hiện tại sau khi trừ phí và tính lãi/lỗ của giao dịch
            pnl_after_fees = trade['PnL'] - total_fee
            current_equity += pnl_after_fees
            current_equity_list.append(current_equity)

            # Cập nhật PnL sau khi trừ phí
            stats._trades.at[idx, 'PnL_after_fees'] = pnl_after_fees

        # Thêm cột 'Fees', 'Margin Required', và 'Current Equity' vào _trades và tổng phí đã áp dụng
        stats._trades['Fees'] = fees_for_trades
        stats._trades['Margin Required'] = margin_required_list
        stats._trades['Current Equity'] = current_equity_list
        stats['Custom Total Fees'] = sum(fees_for_trades)

        # Cập nhật vốn cuối cùng sau khi trừ phí
        final_cash = stats['Equity Final [$]'] - stats['Custom Total Fees']
        stats['Equity Final [$]'] = final_cash
        stats['Return [%]'] = ((final_cash - self._initial_cash) / self._initial_cash) * 100

        # Tính toán Return (Ann.) [%] và Volatility (Ann.) [%] sử dụng công thức chuẩn hóa
        equity_df = stats._equity_curve
        index = equity_df.index
        day_returns = equity_df['Equity'].resample('D').last().dropna().pct_change().dropna()

        if len(day_returns) > 0:
            gmean_day_return = self.geometric_mean(day_returns)
            annual_trading_days = 365 if index.dayofweek.to_series().between(5, 6).mean() > 2/7 * .6 else 252
            annualized_return = (1 + gmean_day_return) ** annual_trading_days - 1
            stats['Return (Ann.) [%]'] = annualized_return * 100

            # Áp dụng công thức chuẩn cho Volatility (Ann.) [%]
            daily_var = day_returns.var(ddof=1)
            stats['Volatility (Ann.) [%]'] = np.sqrt(daily_var * annual_trading_days) * 100
        else:
            stats['Return (Ann.) [%]'] = np.nan
            stats['Volatility (Ann.) [%]'] = np.nan

        self._update_metrics(stats)
        return stats

    def _update_metrics(self, stats):
        equity_df = stats._equity_curve
        index = equity_df.index
        day_returns = equity_df['Equity'].resample('D').last().dropna().pct_change().dropna()

        if len(day_returns) > 0:
            gmean_day_return = gmean(1 + day_returns) - 1
            annual_trading_days = 365 if index.dayofweek.to_series().between(5, 6).mean() > 2/7 * .6 else 252
            annualized_return = (1 + gmean_day_return) ** annual_trading_days - 1
            stats['Return (Ann.) [%]'] = annualized_return * 100

            # Volatility (Ann.) based on standard formula
            daily_var = day_returns.var(ddof=1)
            stats['Volatility (Ann.) [%]'] = np.sqrt(daily_var * annual_trading_days) * 100
        else:
            stats['Return (Ann.) [%]'] = np.nan
            stats['Volatility (Ann.) [%]'] = np.nan


        # Sharpe Ratio
        risk_free_rate = 0
        stats['Sharpe Ratio'] = (
            (stats['Return (Ann.) [%]'] - risk_free_rate) / stats['Volatility (Ann.) [%]']
            if stats['Volatility (Ann.) [%]'] > 0 else np.nan
        )

        # Sortino Ratio
        downside_risk = day_returns[day_returns < 0].std() * np.sqrt(annual_trading_days) * 100
        stats['Sortino Ratio'] = (stats['Return (Ann.) [%]'] - risk_free_rate) / downside_risk if downside_risk > 0 else np.nan

        # Calmar Ratio
        equity_curve = stats._equity_curve['Equity']
        max_drawdown = (equity_curve.cummax() - equity_curve).max()
        stats['Calmar Ratio'] = stats['Return (Ann.) [%]'] / max_drawdown if max_drawdown > 0 else np.nan

        # Win Rate
        stats['Win Rate [%]'] = (stats._trades['PnL_after_fees'] > 0).mean() * 100 if not stats._trades.empty else 0
# Các chỉ số giao dịch khác
        stats['Best Trade [%]'] = stats._trades['ReturnPct'].max() * 100 if not stats._trades.empty else 0
        stats['Worst Trade [%]'] = stats._trades['ReturnPct'].min() * 100 if not stats._trades.empty else 0
        stats['Avg. Trade [%]'] = stats._trades['ReturnPct'].mean() * 100 if not stats._trades.empty else 0

        try:
            stats['Profit Factor'] = stats._trades[stats._trades['PnL_after_fees'] > 0]['PnL_after_fees'].sum() / abs(stats._trades[stats._trades['PnL_after_fees'] < 0]['PnL_after_fees'].sum())
        except:
            stats['Profit Factor'] = 0

        stats['Expectancy [%]'] = stats._trades['ReturnPct'].mean() * 100 if not stats._trades.empty else 0

        try:
            stats['SQN'] = np.sqrt(len(stats._trades)) * stats._trades['PnL_after_fees'].mean() / stats._trades['PnL_after_fees'].std()
        except:
            stats['SQN'] = 0
# Chạy backtest với CustomBacktest
def run_model_backtest(df, selected_features, model):
    bt = CustomBacktest(df, ModelBasedStrategy, cash=1000, commission=0, margin=0.13, hedging=True, exclusive_orders=False)
    stats = bt.run(feat=selected_features, model=model)
    return stats

"""# Select features using Optuna"""

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout







def create_custom_cnn(input_shape):

    model = Sequential()



    # First convolutional block

    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))

    model.add(Dropout(0.5))



    # Second convolutional block

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))

    model.add(MaxPooling1D(pool_size=1))



    # Third convolutional block

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))



    model.add(Flatten())



    model.add(Dense(50, activation='relu'))



    # Output layer for regression (no activation function for linear output)

    model.add(Dense(1))



    # Compile the model

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])



    return model

def objective(trial, X_train, X_valid, y_train, y_valid, y_price, train_data_X_train, train_data_X_valid):



    # Select features based on Optuna's suggestions

    selected_features = []



    at_least_one_feature = False



    for col in X_train.columns:

        select_feature = trial.suggest_categorical(col, [0, 1])

        if select_feature:

#             if len(selected_features) ==10:

#                 break

            selected_features.append(col)

            at_least_one_feature = True



    # If no feature was selected, force selection of at least one feature

    if not at_least_one_feature:

        # Randomly select one feature to be included

        forced_feature = trial.suggest_categorical('forced_feature', X_train.columns.tolist())

        selected_features.append(forced_feature)



    for t in trial.study.trials:

        if t.state != optuna.trial.TrialState.COMPLETE:

            continue

        if t.params == trial.params:

            return np.nan # t.values  # Return the previous value without re-evaluating i







    # Use only the selected features in training



    X_train_selected = X_train[selected_features]

    X_valid_selected = X_valid[selected_features]



    # Train the model



    input_shape = (X_train_selected.shape[1], 1)



    model = create_custom_cnn(input_shape)

    model.fit( X_train_selected , y_train, epochs=5, batch_size=32, validation_data=(X_valid_selected, y_valid), verbose=0)



    stats= run_model_backtest( train_data_X_train,selected_features,model)

    stats1= run_model_backtest(  train_data_X_valid,selected_features,model)

    ret=stats['Return (Ann.) [%]']



    volatility=stats['Volatility (Ann.) [%]']

    ret1=stats1['Return (Ann.) [%]']



    volatility1=stats1['Volatility (Ann.) [%]']

    try: sharpe=ret/volatility

    except: sharpe=0

    try: sharpe1=ret1/volatility1

    except: sharpe1=0

    trade=stats1['# Trades']

    # Save trade value in the trial object for later access

    trial.set_user_attr('trade', trade)

    try:

        gs= (abs((abs(sharpe / sharpe1))-1))

    except:

        gs=0





    return ret,volatility,gs

X_train, X_valid, y_train, y_valid,train_data, train_data_X_train, train_data_X_valid, train_data_y_train, train_data_y_valid = split_optuna_data(data)

"""## Define number of trials (no 2)"""



# Create a study object and optimize the objective function



study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'])

unique_trials = 5

i=0

while unique_trials > len(set(str(t.params) for t in study.trials)):
    try:

        i+=1

        study.optimize(lambda trial: objective(trial, X_train, X_valid, y_train, y_valid, train_data['Close'], train_data_X_train, train_data_X_valid), n_trials=1)

    #     print('Trade times in sample:',i)



        study.trials_dataframe().fillna(0).sort_values('values_0').to_csv('cnn_feature_trials.csv')

        joblib.dump(study, 'cnnmodel.pkl')
    except:
        continue

# train_data_X_train

print(study.trials_dataframe().fillna(0).sort_values('values_0'))

# study = joblib.load(open("rabmodel.pkl", "rb"))

trials = study.trials

completed_trials = [t for t in study.trials if t.values is not None]



# Sort the completed trials based on their objective values

completed_trials.sort(key=lambda trial: trial.values, reverse=True)



# Define top pnl to take for clustering

top_trials = completed_trials



new_df_no_close_col = data.drop([ 'Open','High','Low','Close','Volume', 'Return','Unnamed: 0'], axis=1)



# Extract hyperparameters from top trials

top_features_list = []



for trial in top_trials:

  best_selected_features = [col for idx, col  in enumerate(new_df_no_close_col.columns) if trial.params[idx] == 1] # if bug try change from idx to col

  top_features_list.append(best_selected_features)

top_pnl = []





for best_selected_features in top_features_list:



    new_df_selected = data[[ 'Open','High','Low','Close','Volume', 'Return','Unnamed: 0']+best_selected_features]

    train_select_col_data, _ = split_data(new_df_selected)



    retrain_data = train_select_col_data.drop([ 'Open','High','Low','Close','Volume', 'Return','Unnamed: 0'], axis=1)



#     retrain_data = scale_data(retrain_data)



    X_train, X_valid, y_train, y_valid = train_test_split(retrain_data,

                                                      train_select_col_data['Return'],

                                                      test_size=0.5,shuffle=False)

    X_train=scale_data(X_train,X_train)

    X_valid=scale_data(X_valid,X_train)

    # Create and train model

    # model = RandomForestRegressor()

    # model.fit(X_train, y_train)

    model = create_custom_cnn((X_train.shape[1], 1))



    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_valid, y_valid), verbose=0)

    trade_threshold = 0

    # Make predictions

    y_pred_valid = model.predict(X_valid)

    _, pnl_valid, _, _ = sharpe_for_vn30f(y_pred_valid, y_valid, trade_threshold=trade_threshold, fee_perc=0.01, periods=10)

    pnl_valid_no_nan = np.nan_to_num(pnl_valid, nan=0)

    top_pnl.append(pnl_valid_no_nan)

"""Drop too high correlation PNL array"""

pnl = pd.DataFrame(top_pnl)

pnl = pnl.transpose()



# Calculate the correlation matrix

corr_matrix = pnl.corr().abs()



# Create a mask to only look at the upper triangle (to avoid duplicate checks)

upper_triangle_mask = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))



# Identify columns to drop

to_drop = [column for column in upper_triangle_mask.columns if any(upper_triangle_mask[column] > 0.95)]



# Drop the columns with high correlation

pnl_dropped = pnl.drop(columns=to_drop)



print("Columns to drop:", to_drop)

print("DataFrame after dropping columns:")

print(pnl_dropped)

pnl_array = np.array(pnl_dropped.transpose())

pnl_array = pnl_array[:100]

# Identify columns with all zero values

cols_to_keep = ~np.all(pnl_array == 0, axis=0)



# Drop columns with all zero values

pnl_array = pnl_array[:, cols_to_keep]

"""# ONC"""

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_samples, silhouette_score

def cov2corr(cov):

    # Derive the correlation matrix from a covariance matrix

    std = np.sqrt(np.diag(cov))

    corr = cov/np.outer(std,std)

    corr[corr<-1], corr[corr>1] = -1,1 #for numerical errors

    return corr



def clusterKMeansBase(corr0, maxNumClusters=10, n_init=10, debug=False):

    corr0[corr0 > 1] = 1

    dist_matrix = ((1-corr0)/2.)**.5

    silh_coef_optimal = pd.Series(dtype='float64') #observations matrixs

    kmeans, stat = None, None

    maxNumClusters = min(maxNumClusters, int(np.floor(dist_matrix.shape[0]/2)))

    print("maxNumClusters"+str(maxNumClusters))

    for init in range(0, n_init):

    #The [outer] loop repeats the first loop multiple times, thereby obtaining different initializations. Ref: de Prado and Lewis (2018)

    #DETECTION OF FALSE INVESTMENT STRATEGIES USING UNSUPERVISED LEARNING METHODS

        for num_clusters in range(2, maxNumClusters+1):

            #(maxNumClusters + 2 - num_clusters) # go in reverse order to view more sub-optimal solutions

            kmeans_ = KMeans(n_clusters=num_clusters, n_init=10) #, random_state=3425) #n_jobs=None #n_jobs=None - use all CPUs

            kmeans_ = kmeans_.fit(dist_matrix)

            silh_coef = silhouette_samples(dist_matrix, kmeans_.labels_)

            stat = (silh_coef.mean()/silh_coef.std(), silh_coef_optimal.mean()/silh_coef_optimal.std())



            # If this metric better than the previous set as the optimal number of clusters

            if np.isnan(stat[1]) or stat[0] > stat[1]:

                silh_coef_optimal = silh_coef

                kmeans = kmeans_

                if debug==True:

                    print(kmeans)

                    print(stat)

                    silhouette_avg = silhouette_score(dist_matrix, kmeans_.labels_)

                    print("For n_clusters ="+ str(num_clusters)+ "The average silhouette_score is :"+ str(silhouette_avg))

                    print("********")



    newIdx = np.argsort(kmeans.labels_)

    #print(newIdx)



    corr1 = corr0.iloc[newIdx] #reorder rows

    corr1 = corr1.iloc[:, newIdx] #reorder columns



    clstrs = {i:corr0.columns[np.where(kmeans.labels_==i)[0]].tolist() for i in np.unique(kmeans.labels_)} #cluster members

    silh_coef_optimal = pd.Series(silh_coef_optimal, index=dist_matrix.index)



    return corr1, clstrs, silh_coef_optimal



def makeNewOutputs(corr0, clstrs, clstrs2):

    clstrsNew, newIdx = {}, []

    for i in clstrs.keys():

        clstrsNew[len(clstrsNew.keys())] = list(clstrs[i])



    for i in clstrs2.keys():

        clstrsNew[len(clstrsNew.keys())] = list(clstrs2[i])



    newIdx = [j for i in clstrsNew for j in clstrsNew[i]]

    corrNew = corr0.loc[newIdx, newIdx]



    dist = ((1 - corr0) / 2.)**.5

    kmeans_labels = np.zeros(len(dist.columns))

    for i in clstrsNew.keys():

        idxs = [dist.index.get_loc(k) for k in clstrsNew[i]]

        kmeans_labels[idxs] = i



    silhNew = pd.Series(silhouette_samples(dist, kmeans_labels), index=dist.index)



    return corrNew, clstrsNew, silhNew



def clusterKMeansTop(corr0: pd.DataFrame, maxNumClusters=None, n_init=10):

    if maxNumClusters == None:

        maxNumClusters = corr0.shape[1]-1



    corr1, clstrs, silh = clusterKMeansBase(corr0, maxNumClusters=min(maxNumClusters, corr0.shape[1]-1), n_init=10)#n_init)

    print("clstrs length:"+str(len(clstrs.keys())))

    print("best clustr:"+str(len(clstrs.keys())))

    #for i in clstrs.keys():

    #    print("std:"+str(np.std(silh[clstrs[i]])))



    clusterTstats = {i:np.mean(silh[clstrs[i]])/np.std(silh[clstrs[i]]) for i in clstrs.keys()}

    tStatMean = np.sum(list(clusterTstats.values()))/len(clusterTstats)

    redoClusters = [i for i in clusterTstats.keys() if clusterTstats[i] < tStatMean]

    #print("redo cluster:"+str(redoClusters))

    if len(redoClusters) <= 2:

        print("If 2 or less clusters have a quality rating less than the average then stop.")

        print("redoCluster <=1:"+str(redoClusters)+" clstrs len:"+str(len(clstrs.keys())))

        return corr1, clstrs, silh

    else:

        keysRedo = [j for i in redoClusters for j in clstrs[i]]

        corrTmp = corr0.loc[keysRedo, keysRedo]

        _, clstrs2, _ = clusterKMeansTop(corrTmp, maxNumClusters=min(maxNumClusters, corrTmp.shape[1]-1), n_init=n_init)

        print("clstrs2.len, stat:"+str(len(clstrs2.keys())))

        #Make new outputs, if necessary

        dict_redo_clstrs = {i:clstrs[i] for i in clstrs.keys() if i not in redoClusters}

        corrNew, clstrsNew, silhNew = makeNewOutputs(corr0, dict_redo_clstrs, clstrs2)

        newTstatMean = np.mean([np.mean(silhNew[clstrsNew[i]])/np.std(silhNew[clstrsNew[i]]) for i in clstrsNew.keys()])

        if newTstatMean <= tStatMean:

            print("newTstatMean <= tStatMean"+str(newTstatMean)+ " (len:newClst)"+str(len(clstrsNew.keys()))+" <= "+str(tStatMean)+ " (len:Clst)"+str(len(clstrs.keys())))

            return corr1, clstrs, silh

        else:

            print("newTstatMean > tStatMean"+str(newTstatMean)+ " (len:newClst)"+str(len(clstrsNew.keys()))

                  +" > "+str(tStatMean)+ " (len:Clst)"+str(len(clstrs.keys())))

            return corrNew, clstrsNew, silhNew

            #return corr1, clstrs, silh, stat

# FREQUENCY FEATURE TABLE

correlation_matrix = np.corrcoef(top_pnl)

corr = pd.DataFrame(correlation_matrix)

corr=corr.fillna(0)

#Draw ground truth

matplotlib.pyplot.matshow(corr) #invert y-axis to get origo at lower left corner

matplotlib.pyplot.gca().xaxis.tick_bottom()

matplotlib.pyplot.gca().invert_yaxis()

matplotlib.pyplot.colorbar()

matplotlib.pyplot.show()



#draw prediction based on ONC

corrNew, clstrsNew, silhNew = clusterKMeansTop(corr)

matplotlib.pyplot.matshow(corrNew)

matplotlib.pyplot.gca().xaxis.tick_bottom()

matplotlib.pyplot.gca().invert_yaxis()

matplotlib.pyplot.colorbar()

matplotlib.pyplot.show()

cluster_lists = []



# Iterate through each cluster and its members

for cluster_number, cluster_indices in clstrsNew.items():

    cluster_list = []



    # Iterate through each index in the cluster

    for idx in cluster_indices:

        trial_number = top_trials[idx].number

        cluster_list.append(trial_number)



    cluster_lists.append(cluster_list)



# Print the lists for each cluster

for i, cluster_list in enumerate(cluster_lists):

    print(f"Cluster {i}: {cluster_list}")

top_10_features_per_cluster = []



for cluster_number, cluster_indices in clstrsNew.items():

    cluster_frequency = {}



    for idx in cluster_indices:

        trial_params = top_trials[idx].params

        for key, value in trial_params.items():

            if value == 1:

                cluster_frequency[key] = cluster_frequency.get(key, 0) + 1



    sorted_cluster_frequency = sorted(cluster_frequency.items(), key=lambda x: x[1], reverse=True)

    top_10_features_cluster = [feature for feature, _ in sorted_cluster_frequency[:10]]



    top_10_features_per_cluster.append(top_10_features_cluster)

    print(f"Top 10 features for Cluster {cluster_number}: {top_10_features_cluster}")

top10_feat = pd.DataFrame(top_10_features_per_cluster)

selected_columns_cluster = []

selected_columns_cluster_with_info = []

for item in top_10_features_per_cluster:

  selected_columns = new_df_no_close_col.iloc[:, item]

  selected_columns_cluster.append(selected_columns)

  # Add the required columns to the existing selected columns for each cluster

  selected_columns_with_info = pd.concat([data[[ 'Open','High','Low','Close','Volume', 'Return','Unnamed: 0']], selected_columns], axis=1)

  selected_columns_cluster_with_info.append(selected_columns_with_info)

import pickle

# Save to disk

with open('top_10_features_per_cluster.pkl', 'wb') as f:

    pickle.dump(top_10_features_per_cluster, f)



# Save to disk

with open('top_10_list.pkl', 'wb') as f:

    pickle.dump(selected_columns_cluster, f)

# Save the new files with additional columns

# with open('top_10_features_per_cluster_with_info.pkl', 'wb') as f:

#     pickle.dump(top_10_features_per_cluster, f)



with open('top_10_list_with_info.pkl', 'wb') as f:

    pickle.dump(selected_columns_cluster_with_info, f)

"""# Model selection

## Hyperparameter Tuning
"""

import joblib

# Các cột không phải indicators

non_indicator_columns = [ 'Open', 'High', 'Low', 'Close', 'Volume', 'Return','Unnamed: 0']



# Hàm để lấy các cột indicators từ DataFrame (bỏ qua các cột không phải indicators)

def get_indicator_columns(df, non_indicators):

    return [col for col in df.columns if col not in non_indicators]



# Tạo list mới cho các DataFrame đã sắp xếp

sorted_selected_columns_cluster_with_info = []



# Duyệt qua list_df2 để sắp xếp lại list_df1

for df2 in selected_columns_cluster:

    # Lấy danh sách các cột indicators từ DataFrame trong list_df2

    indicator_columns_df2 = df2.columns.tolist()



    # Tìm DataFrame trong list_df1 có danh sách indicators khớp với df2

    for df1 in selected_columns_cluster_with_info:

        indicator_columns_df1 = get_indicator_columns(df1, non_indicator_columns)



        # Nếu danh sách cột indicators của df1 khớp với df2, thêm df1 vào list đã sắp xếp

        if indicator_columns_df1 == indicator_columns_df2:

            sorted_selected_columns_cluster_with_info.append(df1)

            break

def objective_params(trial, X_train, X_valid, y_train, y_valid, y_close, train_data_X_train, train_data_X_valid):

    # Define the hyperparameter search space

    filters = trial.suggest_categorical('filters', [32,64,128, 256])

    dropout = trial.suggest_float('dropout', 0.1, 0.5)

    batch_size = trial.suggest_categorical('batch_size', [16,32, 64])

    epochs = trial.suggest_int('epochs', 10, 50)





    # Check duplication and skip if it's detected.

    for t in trial.study.trials:

        if t.state != optuna.trial.TrialState.COMPLETE:

            continue

        if t.params == trial.params:

            return np.nan #t.values  # Return the previous value without re-evaluating i



    # custom_early_stopping_instance = CustomEarlyStopping(min_delta=min_delta, patience=patience, verbose=True)

    selected_features = X_train.columns






    model = Sequential()



    # First convolutional block

    model.add(Conv1D(filters=filters, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))

    model.add(Dropout(dropout))



    # Second convolutional block

    model.add(Conv1D(filters=filters, kernel_size=3, activation='relu'))

    model.add(MaxPooling1D(pool_size=1))



    # Third convolutional block

    model.add(Conv1D(filters=filters, kernel_size=3, activation='relu'))



    model.add(Flatten())



    model.add(Dense(50, activation='relu'))



    # Output layer for regression (no activation function for linear output)

    model.add(Dense(1))



    # Compile the model

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(

        X_train, y_train,

        validation_data=(X_valid, y_valid),

        epochs=epochs,

        batch_size=batch_size,



        verbose=0

    )

    # train_data_X_valid=backtest_data(train_data_X_valid)

    stats= run_model_backtest( train_data_X_train,selected_features,model)

    stats1= run_model_backtest(  train_data_X_valid,selected_features,model)

    ret=stats['Return (Ann.) [%]']



    volatility=stats['Volatility (Ann.) [%]']

    ret1=stats1['Return (Ann.) [%]']



    volatility1=stats1['Volatility (Ann.) [%]']

    try: sharpe=ret/volatility

    except: sharpe=0

    try: sharpe1=ret1/volatility1

    except: sharpe1=0

    trade=stats1['# Trades']

    # Save trade value in the trial object for later access

    trial.set_user_attr('trade', trade)

    try:

        gs= (abs((abs(sharpe / sharpe1))-1))

    except:

        gs=0

    return ret,volatility,gs

best_params_list = []

for idx, data_item in enumerate(selected_columns_cluster):





    info_train_cols, _ = split_data(selected_columns_cluster_with_info[idx])

    train_cols, _ = split_data(data_item)

#     optuna_data = scale_data(train_cols)



#     temp= info_train_cols.drop(['Close', 'Open','High','Low','Volume', 'Return','Unnamed: 0'], axis=1)



#     info_optuna_data = scale_data(temp)

#     temp= pd.concat([info_train_cols[[ 'Open','High','Low','Close','Volume', 'Return','Unnamed: 0']], info_optuna_data], axis=1)

    info_train_cols.set_index('Unnamed: 0', inplace=True)

    info_train_cols.index.name = 'datetime'

    info_train_cols.index = pd.to_datetime(info_train_cols.index)

    X_train, X_valid, y_train, y_valid = train_test_split(train_cols,

                                                            train_data['Return'],

                                                            test_size=0.5,

                                                            shuffle=False)

    X_train=scale_data(X_train,X_train)

    X_valid=scale_data(X_valid,X_train)

    train_data_X_train, train_data_X_valid, train_data_y_train, train_data_y_valid = train_test_split(info_train_cols,

                                                            train_data['Return'],

                                                            test_size=0.5,

                                                            shuffle=False)

    temp = train_data_X_train.drop(['Open','High','Low','Close','Volume', 'Return'], axis=1)

    temp=scale_data(temp,X_train)

    train_data_X_train= pd.concat([train_data_X_train[[ 'Open','High','Low','Close','Volume', 'Return']], temp], axis=1)



    temp = train_data_X_valid.drop(['Open','High','Low','Close','Volume', 'Return'], axis=1)

    temp=scale_data(temp,X_train)

    train_data_X_valid= pd.concat([train_data_X_valid[[ 'Open','High','Low','Close','Volume', 'Return']], temp], axis=1)



    study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'])

    unique_trials = 1

    while unique_trials > len(set(str(t.params) for t in study.trials)):

        try:

          study.optimize(lambda trial: objective_params(trial, X_train, X_valid, y_train, y_valid, train_data['Close'], train_data_X_train, train_data_X_valid), n_trials=1)

          study.trials_dataframe().fillna(0).sort_values('values_0').to_csv(f'hypertuning{idx}.csv')

          joblib.dump(study, f'{unique_trials}hypertuningcluster{idx}.pkl')

        except:

          continue



    # Retrieve all trials

    trials = study.trials



    completed_trials = [t for t in study.trials if t.values is not None]



    # Sort trials based on objective values

    completed_trials.sort(key=lambda trial: trial.values, reverse=True)

    params = completed_trials[0].params

    best_params_list.append(params)

    early_stopping =EarlyStopping (monitor='val_loss', patience=5, restore_best_weights=True)

    filters1= params['filters']

    dropout1= params['dropout']

    batch_size1 = params['batch_size']

    epochs1= params['epochs']

    # Select top 1 trials





    model = Sequential()



    # First convolutional block

    model.add(Conv1D(filters=filters1, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))

    model.add(Dropout(dropout1))



    # Second convolutional block

    model.add(Conv1D(filters=filters1, kernel_size=3, activation='relu'))

    model.add(MaxPooling1D(pool_size=1))



    # Third convolutional block

    model.add(Conv1D(filters=filters1, kernel_size=3, activation='relu'))



    model.add(Flatten())



    model.add(Dense(50, activation='relu'))



    # Output layer for regression (no activation function for linear output)

    model.add(Dense(1))



    # Compile the model

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    model.fit(

        X_train, y_train,

        validation_data=(X_valid, y_valid),

        epochs=epochs1,

        batch_size=batch_size1,



        callbacks=[early_stopping],

        verbose=0

    )

    save_model(model,f'best_in_cluster_{idx}.keras')

with open('best_params_list.pkl', 'wb') as f:

  pickle.dump(best_params_list, f)

"""# Test and save result"""

return_data = []

sharpe_list = []

volatility=[]



result = None



train_data, hold_out = split_data(data)





for idx, data_item in enumerate(selected_columns_cluster):

    train_cols, hold_out_cols = split_data(data_item)

    # _, info_hold_out_cols= split_data(sorted_selected_columns_cluster_with_info[idx])

#     _, test_cols = split_data(data_item)

    # optuna_data = scale_data(test_cols)



    temp= hold_out.drop(['Close', 'Open','High','Low','Volume', 'Return','Unnamed: 0'], axis=1)

    optuna_data = train_data.drop(['Open','High','Low','Close','Volume', 'Return','Unnamed: 0'], axis=1)

    X_train, X_valid, y_train, y_valid = train_test_split(optuna_data,

                                                            train_data['Return'],

                                                            test_size=0.5,

                                                            shuffle=False)

#     info_optuna_data = scale_data(temp)

    temp=scale_data(temp,X_train)

    temp= pd.concat([hold_out[[ 'Open','High','Low','Close','Volume', 'Return','Unnamed: 0']], temp], axis=1)



    temp.set_index('Unnamed: 0', inplace=True)

    temp.index.name = 'datetime'

    temp.index = pd.to_datetime(temp.index)

    selected_features = hold_out_cols.columns

    test_data=temp



    # Create and train model

    # model.load_model(f"best_in_cluster_{idx}.json")

    # loaded_model = joblib.load(f'best_in_cluster_{idx}.pkl')

    # model = joblib.load(f'best_in_cluster_{idx}.pkl')

    model = load_model(f"best_in_cluster_{idx}.keras", compile=False)

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Make predictions





    stats1= run_model_backtest( test_data,selected_features,model)

    print(stats1)



    return_data.append(stats1['Return (Ann.) [%]'])

    sharpe_list.append(stats1['Sharpe Ratio'])

    volatility.append(stats1['Volatility (Ann.) [%]'])

#Top 10 feature into list

feature=[]

for i in top_10_features_per_cluster:

    listToStr = ' '.join([str(elem) for elem in i])

    feature.append(listToStr)

print(feature)

feature=[]

for i in top_10_features_per_cluster:

    listToStr = ' '.join([str(elem) for elem in i])

    feature.append(listToStr)



name=[]

for i in range(len(selected_columns_cluster)):

  name.append( 'Cluster '+ str(i))



dict = {'Top 10 Feature' : feature, 'Best params': best_params_list, 'Best sharpe':sharpe_list,"Return (Ann.) [%]": return_data,'Volatility':volatility}

df_result = pd.DataFrame(dict)

df_result.to_csv('result.csv')

print(df_result)



