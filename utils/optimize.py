import os
import joblib
import pickle
import pandas as pd
import numpy as np
import xgboost  as xgb
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
import optuna
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout,Input
from sklearn import preprocessing
from keras.layers import Conv1D,Flatten,MaxPooling1D,Bidirectional,LSTM,Dropout,TimeDistributed,MaxPool2D
from keras.layers import Dense,GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import save_model, load_model
sys.path.append(os.path.abspath('../')) # Thêm đường dẫn của thư mục Capstone vào sys.path
from xgboost import callback
from Capstone.utils.backtest import run_model_backtest
from Capstone.utils.backtest_DL import run_model_backtest_dl
from Capstone.data.data_utils import split_optuna_data, scale_data, split_data, train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

def choose_position(roi, trade_threshold=0):
    # Hàm đơn giản để xác định tín hiệu mua/bán
    if roi > trade_threshold:
        return 1  # Tín hiệu mua
    elif roi < -trade_threshold:
        return -1  # Tín hiệu bán
    else:
        return 0  # Không mở vị thế

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
def create_custom_cnn(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
#----------------------------------------------- Feature select----------------------------------------------------

def objective_xgb(trial, X_train, X_valid, y_train, y_valid, y_price, train_data_X_train, train_data_X_valid):

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
    model = xgb.XGBRegressor()
    model.fit(X_train_selected, y_train)
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

def feature_select_xgb(data, cwd, new_df_no_close_col, feat_trials):
    X_train, X_valid, y_train, y_valid, train_data, train_data_X_train, train_data_X_valid, train_data_y_train, train_data_y_valid = split_optuna_data(data)
    # Define number of trials 

    study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'])
    i=0
    while feat_trials > len(set(str(t.params) for t in study.trials)):
        try:
            i+=1
            study.optimize(lambda trial: objective_xgb(trial, X_train, X_valid, y_train, y_valid, train_data['Close'], train_data_X_train, train_data_X_valid), n_trials=1)
            study.trials_dataframe().fillna(0).sort_values('values_0').to_csv(cwd + 'xgb_feature_trials.csv')
            joblib.dump(study,cwd + 'xgbmodel.pkl')
        except:
            continue

    sort_df = study.trials_dataframe().fillna(0).sort_values('values_0')

    completed_trials = [t for t in study.trials if t.values is not None]
    completed_trials.sort(key=lambda trial: trial.values, reverse=True) # Sort the completed trials based on their objective values

    # Define top pnl to take for clustering
    top_trials = completed_trials

    # Extract hyperparameters from top trials
    top_features_list = []

    for trial in top_trials:
        best_selected_features = [col for idx, col  in enumerate(new_df_no_close_col.columns) if trial.params[idx] == 1] # if bug try change from idx to col
        top_features_list.append(best_selected_features)

    return train_data, sort_df, top_trials, top_features_list




def objective_lgbm(trial, X_train, X_valid, y_train, y_valid, y_price, train_data_X_train, train_data_X_valid):
    # Select features based on Optuna's suggestions
    selected_features = []
    at_least_one_feature = False
    for col in X_train.columns:
        select_feature = trial.suggest_categorical(col, [0, 1])
        if select_feature:
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
    model = LGBMRegressor()
    model.fit(X_train_selected, y_train)
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

def feature_select_lgbm(data, cwd, new_df_no_close_col, feat_trials):
    X_train, X_valid, y_train, y_valid, train_data, train_data_X_train, train_data_X_valid, train_data_y_train, train_data_y_valid = split_optuna_data(data)
    # Define number of trials 

    study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'])
    i=0
    while feat_trials > len(set(str(t.params) for t in study.trials)):
        try:
            i+=1
            study.optimize(lambda trial: objective_lgbm(trial, X_train, X_valid, y_train, y_valid, train_data['Close'], train_data_X_train, train_data_X_valid), n_trials=1)
            study.trials_dataframe().fillna(0).sort_values('values_0').to_csv(cwd + 'lgbm_feature_trials.csv')
            joblib.dump(study,cwd + 'lgbmmodel.pkl')
        except:
            continue

    sort_df = study.trials_dataframe().fillna(0).sort_values('values_0')

    completed_trials = [t for t in study.trials if t.values is not None]
    completed_trials.sort(key=lambda trial: trial.values, reverse=True) # Sort the completed trials based on their objective values

    # Define top pnl to take for clustering
    top_trials = completed_trials

    # Extract hyperparameters from top trials
    top_features_list = []

    for trial in top_trials:
        best_selected_features = [col for idx, col  in enumerate(new_df_no_close_col.columns) if trial.params[idx] == 1] # if bug try change from idx to col
        top_features_list.append(best_selected_features)

    return train_data, sort_df, top_trials, top_features_list


def objective_rf(trial, X_train, X_valid, y_train, y_valid, y_price, train_data_X_train, train_data_X_valid):

    # Select features based on Optuna's suggestions
    selected_features = []

    at_least_one_feature = False

    for col in X_train.columns:
        select_feature = trial.suggest_categorical(col, [0, 1])
        if select_feature:

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
    model = RandomForestRegressor()
    model.fit(X_train_selected, y_train)
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

    trial.set_user_attr('trade', trade)
    try:
        gs= (abs((abs(sharpe / sharpe1))-1))
    except:
        gs=0
    return ret,volatility,gs

def feature_select_rf(data, cwd, new_df_no_close_col, feat_trials):
    X_train, X_valid, y_train, y_valid, train_data, train_data_X_train, train_data_X_valid, train_data_y_train, train_data_y_valid = split_optuna_data(data)
    # Define number of trials 

    study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'])
    i=0
    while feat_trials > len(set(str(t.params) for t in study.trials)):
        try:
            i+=1
            study.optimize(lambda trial: objective_rf(trial, X_train, X_valid, y_train, y_valid, train_data['Close'], train_data_X_train, train_data_X_valid), n_trials=1)
            study.trials_dataframe().fillna(0).sort_values('values_0').to_csv(cwd + 'rf_feature_trials.csv')
            joblib.dump(study,cwd + 'rfmodel.pkl')
        except:
            continue

    sort_df = study.trials_dataframe().fillna(0).sort_values('values_0')

    completed_trials = [t for t in study.trials if t.values is not None]
    completed_trials.sort(key=lambda trial: trial.values, reverse=True) # Sort the completed trials based on their objective values

    # Define top pnl to take for clustering
    top_trials = completed_trials

    # Extract hyperparameters from top trials
    top_features_list = []

    for trial in top_trials:
        best_selected_features = [col for idx, col  in enumerate(new_df_no_close_col.columns) if trial.params[idx] == 1] # if bug try change from idx to col
        top_features_list.append(best_selected_features)

    return train_data, sort_df, top_trials, top_features_list


def objective_lstm(trial, X_train, X_valid, y_train, y_valid, y_price, train_data_X_train, train_data_X_valid):
    selected_features = []
    at_least_one_feature = False
    for col in X_train.columns:
        select_feature = trial.suggest_categorical(col, [0, 1])
        if select_feature:
            selected_features.append(col)
            at_least_one_feature = True
    if not at_least_one_feature:
        # Randomly select one feature to be included
        forced_feature = trial.suggest_categorical('forced_feature', X_train.columns.tolist())
        selected_features.append(forced_feature)
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.params == trial.params:
            return np.nan 
    X_train_selected = X_train[selected_features]
    X_valid_selected = X_valid[selected_features]
    input_shape = (X_train.shape[1:])
    model = Sequential()
    model.add(LSTM(units = 50, activation = 'relu', return_sequences=True
                  ,input_shape = (X_train_selected.shape[1], 1)))
    model.add(Dropout(0.1))
    model.add(LSTM(units=50))
    model.add(Dense(1))  # Assuming this output is suitable for the prediction targets
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    model.fit(X_train_selected, y_train, epochs=5, batch_size=32, validation_data=(X_valid, y_valid), verbose=0)
    stats= run_model_backtest_dl( train_data_X_train,selected_features,model)
    stats1= run_model_backtest_dl(  train_data_X_valid,selected_features,model)
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

def feature_select_lstm(data, cwd, new_df_no_close_col, feat_trials):
    X_train, X_valid, y_train, y_valid, train_data, train_data_X_train, train_data_X_valid, train_data_y_train, train_data_y_valid = split_optuna_data(data)
    # Define number of trials 

    study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'])
    i=0
    while feat_trials > len(set(str(t.params) for t in study.trials)):
        try:
            i+=1
            study.optimize(lambda trial: objective_lstm(trial, X_train, X_valid, y_train, y_valid, train_data['Close'], train_data_X_train, train_data_X_valid), n_trials=1)
            study.trials_dataframe().fillna(0).sort_values('values_0').to_csv(cwd + 'lstm_feature_trials.csv')
            joblib.dump(study,cwd + 'lstmmodel.pkl')
        except:
            continue

    sort_df = study.trials_dataframe().fillna(0).sort_values('values_0')

    completed_trials = [t for t in study.trials if t.values is not None]
    completed_trials.sort(key=lambda trial: trial.values, reverse=True) # Sort the completed trials based on their objective values

    # Define top pnl to take for clustering
    top_trials = completed_trials

    # Extract hyperparameters from top trials
    top_features_list = []

    for trial in top_trials:
        best_selected_features = [col for idx, col  in enumerate(new_df_no_close_col.columns) if trial.params[idx] == 1] # if bug try change from idx to col
        top_features_list.append(best_selected_features)

    return train_data, sort_df, top_trials, top_features_list

def objective_cnn(trial, X_train, X_valid, y_train, y_valid, y_price, train_data_X_train, train_data_X_valid):
    # Select features based on Optuna's suggestions
    selected_features = []
    at_least_one_feature = False
    for col in X_train.columns:
        select_feature = trial.suggest_categorical(col, [0, 1])
        if select_feature:
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
    stats= run_model_backtest_dl( train_data_X_train,selected_features,model)
    stats1= run_model_backtest_dl(  train_data_X_valid,selected_features,model)
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

def feature_select_cnn(data, cwd, new_df_no_close_col, feat_trials):
    X_train, X_valid, y_train, y_valid, train_data, train_data_X_train, train_data_X_valid, train_data_y_train, train_data_y_valid = split_optuna_data(data)
    # Define number of trials 

    study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'])
    i=0
    while feat_trials > len(set(str(t.params) for t in study.trials)):
        try:
            i+=1
            study.optimize(lambda trial: objective_cnn(trial, X_train, X_valid, y_train, y_valid, train_data['Close'], train_data_X_train, train_data_X_valid), n_trials=1)
            study.trials_dataframe().fillna(0).sort_values('values_0').to_csv(cwd + 'cnn_feature_trials.csv')
            joblib.dump(study,cwd + 'cnnmodel.pkl')
        except:
            continue

    sort_df = study.trials_dataframe().fillna(0).sort_values('values_0')

    completed_trials = [t for t in study.trials if t.values is not None]
    completed_trials.sort(key=lambda trial: trial.values, reverse=True) # Sort the completed trials based on their objective values

    # Define top pnl to take for clustering
    top_trials = completed_trials

    # Extract hyperparameters from top trials
    top_features_list = []

    for trial in top_trials:
        best_selected_features = [col for idx, col  in enumerate(new_df_no_close_col.columns) if trial.params[idx] == 1] # if bug try change from idx to col
        top_features_list.append(best_selected_features)

    return train_data, sort_df, top_trials, top_features_list

def objective_cnn_lstm(trial, X_train, X_valid, y_train, y_valid, y_price, train_data_X_train, train_data_X_valid):
    # Select features based on Optuna's suggestions
    selected_features = []
    at_least_one_feature = False
    for col in X_train.columns:
        select_feature = trial.suggest_categorical(col, [0, 1])
        if select_feature:

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
    # Create a Sequential model
    model = Sequential()
    #add model layers
    model.add(Conv1D(64, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(1))
    model.add(Conv1D(128, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    # model.add(Flatten())
    model.add(LSTM(50,return_sequences=True))
    model.add(Dropout(0.1))
    model.add(LSTM(50,return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    model.compile(optimizer='RMSprop', loss='mse')
    model.fit( X_train_selected , y_train, epochs=10, batch_size=32, validation_data=(X_valid_selected, y_valid), verbose=0)
    stats= run_model_backtest_dl( train_data_X_train,selected_features,model)
    stats1= run_model_backtest_dl(  train_data_X_valid,selected_features,model)
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

def feature_select_cnn_lstm(data, cwd, new_df_no_close_col, feat_trials):
    X_train, X_valid, y_train, y_valid, train_data, train_data_X_train, train_data_X_valid, train_data_y_train, train_data_y_valid = split_optuna_data(data)
    # Define number of trials 

    study = optuna.create_study(directions=['maximize', 'minimize', 'minimize'])
    i=0
    while feat_trials > len(set(str(t.params) for t in study.trials)):
        try:
            i+=1
            study.optimize(lambda trial: objective_cnn_lstm(trial, X_train, X_valid, y_train, y_valid, train_data['Close'], train_data_X_train, train_data_X_valid), n_trials=1)
            study.trials_dataframe().fillna(0).sort_values('values_0').to_csv(cwd + 'cnn_lstm_feature_trials.csv')
            joblib.dump(study,cwd + 'cnn_lstmmodel.pkl')
        except:
            continue

    sort_df = study.trials_dataframe().fillna(0).sort_values('values_0')

    completed_trials = [t for t in study.trials if t.values is not None]
    completed_trials.sort(key=lambda trial: trial.values, reverse=True) # Sort the completed trials based on their objective values

    # Define top pnl to take for clustering
    top_trials = completed_trials

    # Extract hyperparameters from top trials
    top_features_list = []

    for trial in top_trials:
        best_selected_features = [col for idx, col  in enumerate(new_df_no_close_col.columns) if trial.params[idx] == 1] # if bug try change from idx to col
        top_features_list.append(best_selected_features)

    return train_data, sort_df, top_trials, top_features_list
#----------------------------------------------- ONC----------------------------------------------------
def retrieve_top_pnl_xgb(data, top_features_list, drop_list):
    top_pnl = []
    for best_selected_features in top_features_list:

        new_df_selected = data[drop_list+best_selected_features]
        train_select_col_data, _ = split_data(new_df_selected)

        retrain_data = train_select_col_data.drop(drop_list, axis=1)

        X_train, X_valid, y_train, y_valid = train_test_split(retrain_data,
                                                        train_select_col_data['Return'],
                                                        test_size=0.5,shuffle=False)
        X_train=scale_data(X_train,X_train)
        X_valid=scale_data(X_valid,X_train)
        # Create and train model
        model = xgb.XGBRegressor()
        model.fit(X_train, y_train)
        trade_threshold = 0
        # Make predictions
        y_pred_valid = model.predict(X_valid)
        _, pnl_valid, _, _ = sharpe_for_vn30f(y_pred_valid, y_valid, trade_threshold=trade_threshold, fee_perc=0.01, periods=10)
        pnl_valid_no_nan = np.nan_to_num(pnl_valid, nan=0)
        top_pnl.append(pnl_valid_no_nan)
        # pnl = pd.DataFrame(top_pnl)
        # pnl = pnl.transpose()
    return top_pnl

def retrieve_top_pnl_lgbm(data, top_features_list, drop_list):
    top_pnl = []
    for best_selected_features in top_features_list:

        new_df_selected = data[drop_list+best_selected_features]
        train_select_col_data, _ = split_data(new_df_selected)

        retrain_data = train_select_col_data.drop(drop_list, axis=1)

        X_train, X_valid, y_train, y_valid = train_test_split(retrain_data,
                                                        train_select_col_data['Return'],
                                                        test_size=0.5,shuffle=False)
        X_train=scale_data(X_train,X_train)
        X_valid=scale_data(X_valid,X_train)
        # Create and train model
        model = LGBMRegressor()
        model.fit(X_train, y_train)
        trade_threshold = 0
        # Make predictions
        y_pred_valid = model.predict(X_valid)
        _, pnl_valid, _, _ = sharpe_for_vn30f(y_pred_valid, y_valid, trade_threshold=trade_threshold, fee_perc=0.01, periods=10)
        pnl_valid_no_nan = np.nan_to_num(pnl_valid, nan=0)
        top_pnl.append(pnl_valid_no_nan)
        # pnl = pd.DataFrame(top_pnl)
        # pnl = pnl.transpose()
    return top_pnl

def retrieve_top_pnl_rf(data, top_features_list, drop_list):
    top_pnl = []
    for best_selected_features in top_features_list:

        new_df_selected = data[drop_list+best_selected_features]
        train_select_col_data, _ = split_data(new_df_selected)

        retrain_data = train_select_col_data.drop(drop_list, axis=1)

        X_train, X_valid, y_train, y_valid = train_test_split(retrain_data,
                                                        train_select_col_data['Return'],
                                                        test_size=0.5,shuffle=False)
        X_train=scale_data(X_train,X_train)
        X_valid=scale_data(X_valid,X_train)
        # Create and train model
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        trade_threshold = 0
        # Make predictions
        y_pred_valid = model.predict(X_valid)
        _, pnl_valid, _, _ = sharpe_for_vn30f(y_pred_valid, y_valid, trade_threshold=trade_threshold, fee_perc=0.01, periods=10)
        pnl_valid_no_nan = np.nan_to_num(pnl_valid, nan=0)
        top_pnl.append(pnl_valid_no_nan)
        # pnl = pd.DataFrame(top_pnl)
        # pnl = pnl.transpose()
    return top_pnl

def retrieve_top_pnl_cnn(data, top_features_list, drop_list):
    top_pnl = []
    for best_selected_features in top_features_list:

        new_df_selected = data[drop_list+best_selected_features]
        train_select_col_data, _ = split_data(new_df_selected)

        retrain_data = train_select_col_data.drop(drop_list, axis=1)

        X_train, X_valid, y_train, y_valid = train_test_split(retrain_data,
                                                        train_select_col_data['Return'],
                                                        test_size=0.5,shuffle=False)
        X_train=scale_data(X_train,X_train)
        X_valid=scale_data(X_valid,X_train)
        # Create and train model
        model = create_custom_cnn((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_valid, y_valid), verbose=0)
        trade_threshold = 0
        # Make predictions
        y_pred_valid = model.predict(X_valid)
        _, pnl_valid, _, _ = sharpe_for_vn30f(y_pred_valid, y_valid, trade_threshold=trade_threshold, fee_perc=0.01, periods=10)
        pnl_valid_no_nan = np.nan_to_num(pnl_valid, nan=0)
        top_pnl.append(pnl_valid_no_nan)
        # pnl = pd.DataFrame(top_pnl)
        # pnl = pnl.transpose()
    return top_pnl

def retrieve_top_pnl_lstm(data, top_features_list, drop_list):
    top_pnl = []
    for best_selected_features in top_features_list:

        new_df_selected = data[drop_list+best_selected_features]
        train_select_col_data, _ = split_data(new_df_selected)

        retrain_data = train_select_col_data.drop(drop_list, axis=1)

        X_train, X_valid, y_train, y_valid = train_test_split(retrain_data,
                                                        train_select_col_data['Return'],
                                                        test_size=0.5,shuffle=False)
        X_train=scale_data(X_train,X_train)
        X_valid=scale_data(X_valid,X_train)
        # Create and train model
        model = Sequential()
        model.add(LSTM(units = 50, activation = 'relu', return_sequences=True
                    ,input_shape = (X_train.shape[1], 1)))
        model.add(Dropout(0.1))
        model.add(LSTM(units=50))
        model.add(Dense(1))  # Assuming this output is suitable for the prediction targets
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
        model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_valid, y_valid), verbose=0)
        trade_threshold = 0
        # Make predictions
        y_pred_valid = model.predict(X_valid)
        _, pnl_valid, _, _ = sharpe_for_vn30f(y_pred_valid, y_valid, trade_threshold=trade_threshold, fee_perc=0.01, periods=10)
        pnl_valid_no_nan = np.nan_to_num(pnl_valid, nan=0)
        top_pnl.append(pnl_valid_no_nan)
        # pnl = pd.DataFrame(top_pnl)
        # pnl = pnl.transpose()
    return top_pnl

def retrieve_top_pnl_cnn_lstm(data, top_features_list, drop_list):
    top_pnl = []
    for best_selected_features in top_features_list:

        new_df_selected = data[drop_list+best_selected_features]
        train_select_col_data, _ = split_data(new_df_selected)

        retrain_data = train_select_col_data.drop(drop_list, axis=1)

        X_train, X_valid, y_train, y_valid = train_test_split(retrain_data,
                                                        train_select_col_data['Return'],
                                                        test_size=0.5,shuffle=False)
        X_train=scale_data(X_train,X_train)
        X_valid=scale_data(X_valid,X_train)
        input_shape = (X_train.shape[1], 1)
        # Create and train model
        model = Sequential()
        #add model layers
        model.add(Conv1D(64, kernel_size=1, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(1))
        model.add(Conv1D(128, kernel_size=1, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
        # model.add(Flatten())
        model.add(LSTM(50,return_sequences=True))
        model.add(Dropout(0.1))
        model.add(LSTM(50,return_sequences=False))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(optimizer='RMSprop', loss='mse')
        model.fit( X_train , y_train, epochs=10, batch_size=32, validation_data=(X_valid, y_valid), verbose=0)
        trade_threshold = 0
        # Make predictions
        y_pred_valid = model.predict(X_valid)
        _, pnl_valid, _, _ = sharpe_for_vn30f(y_pred_valid, y_valid, trade_threshold=trade_threshold, fee_perc=0.01, periods=10)
        pnl_valid_no_nan = np.nan_to_num(pnl_valid, nan=0)
        top_pnl.append(pnl_valid_no_nan)
        # pnl = pd.DataFrame(top_pnl)
        # pnl = pnl.transpose()
    return top_pnl

def get_indicator_columns(df, drop_list):
    # Hàm để lấy các cột indicators từ DataFrame (bỏ qua các cột không phải indicators)
    return [col for col in df.columns if col not in drop_list]

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

def process_clusters_and_save(clstrsNew, top_trials, new_df_no_close_col, data, cwd, drop_list, output_folder="output_clusters", saving=True):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cluster_lists = []
    top_10_features_per_cluster = []
    selected_columns_cluster = []
    selected_columns_cluster_with_info = []

    # Iterate through each cluster and its members
    for cluster_number, cluster_indices in clstrsNew.items():
        cluster_list = []

        # Collect trial numbers for each cluster
        for idx in cluster_indices:
            trial_number = top_trials[idx].number
            cluster_list.append(trial_number)

        cluster_lists.append(cluster_list)

        # Calculate feature frequencies for top 10 features in each cluster
        cluster_frequency = {}
        for idx in cluster_indices:
            trial_params = top_trials[idx].params
            for key, value in trial_params.items():
                if value == 1:
                    cluster_frequency[key] = cluster_frequency.get(key, 0) + 1

        # Sort features by frequency and get top 10
        sorted_cluster_frequency = sorted(cluster_frequency.items(), key=lambda x: x[1], reverse=True)
        top_10_features_cluster = [feature for feature, _ in sorted_cluster_frequency[:10]]
        top_10_features_per_cluster.append(top_10_features_cluster)

    # Select columns for each cluster based on top features
    for item in top_10_features_per_cluster:
        selected_columns = new_df_no_close_col.iloc[:, item]
        selected_columns_cluster.append(selected_columns)
        # Add the required columns to the existing selected columns for each cluster
        selected_columns_with_info = pd.concat([data[[ 'Open','High','Low','Close','Volume', 'Return','Unnamed: 0']], selected_columns], axis=1)
        selected_columns_cluster_with_info.append(selected_columns_with_info)
        
    if saving == True:
        # Save top 10 features, selected columns, and selected columns with additional info to pickle files
        with open(cwd + 'top_10_features_per_cluster.pkl', 'wb') as f:
            pickle.dump(top_10_features_per_cluster, f)
        with open(cwd + 'top_10_list.pkl', 'wb') as f:
            pickle.dump(selected_columns_cluster, f)
        with open(cwd + 'top_10_list_with_info.pkl', 'wb') as f:
            pickle.dump(selected_columns_cluster_with_info, f)

        print("Pickle files saved to disk.")

    # Return DataFrames if further processing is needed
    return top_10_features_per_cluster, selected_columns_cluster, selected_columns_cluster_with_info

#----------------------------------------------- Hyperparameter Tunning----------------------------------------------------

class CustomEarlyStopping(callback.TrainingCallback):
    def __init__(self, min_delta, patience, verbose=False):
        super().__init__()
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.best_score = np.inf
        self.wait = 0
        self.stopped_epoch = 0

    def after_iteration(self, model, epoch, evals_log):
        if not evals_log:
            return False
        metric_name = next(iter(evals_log['validation_0']))
        score = evals_log['validation_0'][metric_name][-1]
        if score < (self.best_score - self.min_delta):
            self.best_score = score
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.verbose:
                    print(f"\nStopping. Best score: {self.best_score}")
                self.stopped_epoch = epoch
                return True
        return False

    def get_best_score(self):
        return self.best_score
    
def objective_params_xgb(trial, X_train, X_valid, y_train, y_valid, y_close, train_data_X_train, train_data_X_valid):
    # Define the hyperparameter search space
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': 8000,  # does not matter, think of it as max epochs, and we stop the model based on early stopping, so any extremely high number works
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # can't comment, never played with that
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),  # you dont want to sample less than 50% of your data
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),  # you dont want to sample less than 30% of your features pr boosting round
        }

    # Check duplication and skip if it's detected.
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.params == trial.params:
            return np.nan #t.values  # Return the previous value without re-evaluating i
    min_delta = 0.0001
    patience = 30
    custom_early_stopping_instance = CustomEarlyStopping(min_delta=min_delta, patience=patience, verbose=True)
    selected_features = X_train.columns

    # Train the model
    model = xgb.XGBRegressor(**params, callbacks=[custom_early_stopping_instance])
    model.fit(X_train, y_train)
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

def objective_params_lgbm(trial, X_train, X_valid, y_train, y_valid, y_close, train_data_X_train, train_data_X_valid):
    # Define the hyperparameter search space
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'max_depth': trial.suggest_int('max_depth', -1, 50),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }

    # Check duplication and skip if it's detected.
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.params == trial.params:
            return np.nan #t.values  # Return the previous value without re-evaluating i
    min_delta = 0.0001
    patience = 30
    custom_early_stopping_instance = CustomEarlyStopping(min_delta=min_delta, patience=patience, verbose=True)
    selected_features = X_train.columns

    # Train the model
    model = LGBMRegressor(**params, callbacks=[custom_early_stopping_instance])
    model.fit(X_train, y_train)
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


def objective_params_rf(trial, X_train, X_valid, y_train, y_valid, y_close, train_data_X_train, train_data_X_valid):
    # Define the hyperparameter search space
    params = {
        'n_estimators' : trial.suggest_int("n_estimators", 50, 200),
        'max_depth' : trial.suggest_int("max_depth", 2, 20),
        'min_samples_split' : trial.suggest_int("min_samples_split", 2, 10),
        'min_samples_leaf' : trial.suggest_int("min_samples_leaf", 1, 4),
        }

    # Check duplication and skip if it's detected.
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.params == trial.params:
            return np.nan #t.values  # Return the previous value without re-evaluating i
   
    selected_features = X_train.columns

    # Train the model
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
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


def objective_params_lstm(trial, X_train, X_valid, y_train, y_valid, y_close, train_data_X_train, train_data_X_valid):
    # Define the hyperparameter search space
    units = trial.suggest_int('units', 50, 200)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16,32, 64])
    epochs = trial.suggest_int('epochs', 10, 50)

    # Check duplication and skip if it's detected.
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.params == trial.params:
            return np.nan #t.values  # Return the previous value without re-evaluating i
   
    selected_features = X_train.columns

    # Train the model
    model = Sequential()

    model.add(LSTM(units = units, activation = 'relu', return_sequences=True
                  ,input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dense(1))  # Assuming this output is suitable for the prediction targets
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    # model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_valid, y_valid), verbose=0)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    # train_data_X_valid=backtest_data(train_data_X_valid)
    stats= run_model_backtest_dl( train_data_X_train,selected_features,model)
    stats1= run_model_backtest_dl(  train_data_X_valid,selected_features,model)
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

def objective_params_cnn(trial, X_train, X_valid, y_train, y_valid, y_close, train_data_X_train, train_data_X_valid):
    # Define the hyperparameter search space
    filters = trial.suggest_categorical('filters', [32,64,128, 256])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16,32,64])
    epochs = trial.suggest_int('epochs', 10, 50)

    # Check duplication and skip if it's detected.
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.params == trial.params:
            return np.nan #t.values  # Return the previous value without re-evaluating i
   
    selected_features = X_train.columns

    # Train the model
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
    stats= run_model_backtest_dl( train_data_X_train,selected_features,model)
    stats1= run_model_backtest_dl(  train_data_X_valid,selected_features,model)
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

def objective_params_cnn_lstm(trial, X_train, X_valid, y_train, y_valid, y_close, train_data_X_train, train_data_X_valid):
    # Define the hyperparameter search space
    filters1 = trial.suggest_categorical('filters1', [64,128, 256])
    filters2 = trial.suggest_categorical('filters2', [64,128, 256])
    filters3 = trial.suggest_categorical('filters3', [64,128, 256])
    units= trial.suggest_int('units', 50, 200)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [16,32, 64])
    epochs = trial.suggest_int('epochs', 10, 50)

    # Check duplication and skip if it's detected.
    for t in trial.study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.params == trial.params:
            return np.nan #t.values  # Return the previous value without re-evaluating i
   
    selected_features = X_train.columns
    input_shape = (X_train.shape[1], 1)
    # Train the model
    model = Sequential()
    #add model layers
    model.add(Conv1D(filters1, kernel_size=1, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(1))
    model.add(Conv1D(filters2, kernel_size=1, activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=filters3, kernel_size=3, activation='relu'))
    # model.add(Flatten())
    model.add(LSTM(units,return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units,return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='RMSprop', loss='mse')
    # model.fit( X_train , y_train, epochs=10, batch_size=32, validation_data=(X_valid, y_valid), verbose=0)
    model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    # train_data_X_valid=backtest_data(train_data_X_valid)
    stats= run_model_backtest_dl( train_data_X_train,selected_features,model)
    stats1= run_model_backtest_dl(  train_data_X_valid,selected_features,model)
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


def sort_select_cluster(selected_columns_cluster, selected_columns_cluster_with_info, drop_list):
    # Tạo list mới cho các DataFrame đã sắp xếp
    sorted_selected_columns_cluster_with_info = []

    # Duyệt qua list_df2 để sắp xếp lại list_df1
    for df2 in selected_columns_cluster:
        # Lấy danh sách các cột indicators từ DataFrame trong list_df2
        indicator_columns_df2 = df2.columns.tolist()

        # Tìm DataFrame trong list_df1 có danh sách indicators khớp với df2
        for df1 in selected_columns_cluster_with_info:
            indicator_columns_df1 = get_indicator_columns(df1, drop_list)

            # Nếu danh sách cột indicators của df1 khớp với df2, thêm df1 vào list đã sắp xếp
            if indicator_columns_df1 == indicator_columns_df2:
                sorted_selected_columns_cluster_with_info.append(df1)
                break

    return sorted_selected_columns_cluster_with_info  

def hyper_tuning_xgb(train_data, cwd, selected_columns_cluster, selected_columns_cluster_with_info, tuning_trials, saving=True):
    best_params_list = []
    for idx, data_item in enumerate(selected_columns_cluster):


        info_train_cols, _ = split_data(selected_columns_cluster_with_info[idx])
        train_cols, _ = split_data(data_item)

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
        while tuning_trials > len(set(str(t.params) for t in study.trials)):
            try:
                study.optimize(lambda trial: objective_params_xgb(trial, X_train, X_valid, y_train, y_valid, train_data['Close'], train_data_X_train, train_data_X_valid), n_trials=1)
                study.trials_dataframe().fillna(0).sort_values('values_0').to_csv(cwd + f'hypertuning{idx}.csv')
                joblib.dump(study,cwd + f'hypertuningcluster{idx}.pkl')
            except:
                continue

        completed_trials = [t for t in study.trials if t.values is not None]
        completed_trials.sort(key=lambda trial: trial.values, reverse=True) # Sort trials based on objective values

        # Select top 1 trials
        params = completed_trials[0].params
        best_params_list.append(params)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        model.save_model(cwd + f'best_in_cluster_{idx}.json')

    if saving==True:
        with open(cwd + 'best_params_list.pkl', 'wb') as f:
            pickle.dump(best_params_list, f)

    return best_params_list

def hyper_tuning_lgbm(train_data, cwd, selected_columns_cluster, selected_columns_cluster_with_info, tuning_trials, saving=True):
    best_params_list = []
    for idx, data_item in enumerate(selected_columns_cluster):


        info_train_cols, _ = split_data(selected_columns_cluster_with_info[idx])
        train_cols, _ = split_data(data_item)

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
        while tuning_trials > len(set(str(t.params) for t in study.trials)):
            try:
                study.optimize(lambda trial: objective_params_lgbm(trial, X_train, X_valid, y_train, y_valid, train_data['Close'], train_data_X_train, train_data_X_valid), n_trials=1)
                study.trials_dataframe().fillna(0).sort_values('values_0').to_csv(cwd + f'hypertuning{idx}.csv')
                joblib.dump(study,cwd + f'hypertuningcluster{idx}.pkl')
            except:
                continue

        completed_trials = [t for t in study.trials if t.values is not None]
        completed_trials.sort(key=lambda trial: trial.values, reverse=True) # Sort trials based on objective values

        # Select top 1 trials
        params = completed_trials[0].params
        best_params_list.append(params)

        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)    
        joblib.dump(model,cwd + f'best_in_cluster_{idx}.pkl')
    if saving==True:
        with open(cwd + 'best_params_list.pkl', 'wb') as f:
            pickle.dump(best_params_list, f)
    return best_params_list

def hyper_tuning_rf(train_data, cwd, selected_columns_cluster, selected_columns_cluster_with_info, tuning_trials, saving=True):
    best_params_list = []
    for idx, data_item in enumerate(selected_columns_cluster):


        info_train_cols, _ = split_data(selected_columns_cluster_with_info[idx])
        train_cols, _ = split_data(data_item)

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
        while tuning_trials > len(set(str(t.params) for t in study.trials)):
            try:
                study.optimize(lambda trial: objective_params_rf(trial, X_train, X_valid, y_train, y_valid, train_data['Close'], train_data_X_train, train_data_X_valid), n_trials=1)
                study.trials_dataframe().fillna(0).sort_values('values_0').to_csv(cwd + f'hypertuning{idx}.csv')
                joblib.dump(study,cwd + f'hypertuningcluster{idx}.pkl')
            except:
                continue

        completed_trials = [t for t in study.trials if t.values is not None]
        completed_trials.sort(key=lambda trial: trial.values, reverse=True) # Sort trials based on objective values

        # Select top 1 trials
        params = completed_trials[0].params
        best_params_list.append(params)

        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)    
        joblib.dump(model,cwd + f'best_in_cluster_{idx}.pkl')
    if saving==True:
        with open(cwd + 'best_params_list.pkl', 'wb') as f:
            pickle.dump(best_params_list, f)
    return best_params_list

def hyper_tuning_lstm(train_data, cwd, selected_columns_cluster, selected_columns_cluster_with_info, tuning_trials, saving=True):
    best_params_list = []
    for idx, data_item in enumerate(selected_columns_cluster):


        info_train_cols, _ = split_data(selected_columns_cluster_with_info[idx])
        train_cols, _ = split_data(data_item)

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
        while tuning_trials > len(set(str(t.params) for t in study.trials)):
            try:
                study.optimize(lambda trial: objective_params_lstm(trial, X_train, X_valid, y_train, y_valid, train_data['Close'], train_data_X_train, train_data_X_valid), n_trials=1)
                study.trials_dataframe().fillna(0).sort_values('values_0').to_csv(cwd + f'hypertuning{idx}.csv')
                joblib.dump(study,cwd + f'hypertuningcluster{idx}.pkl')
            except:
                continue

        completed_trials = [t for t in study.trials if t.values is not None]
        completed_trials.sort(key=lambda trial: trial.values, reverse=True) # Sort trials based on objective values

        # Select top 1 trials
        params = completed_trials[0].params
        best_params_list.append(params)
        units1= params['units']
        dropout1= params['dropout']
        batch_size1 = params['batch_size']
        epochs1= params['epochs']
        early_stopping =EarlyStopping (monitor='val_loss', patience=5, restore_best_weights=True)
        model = Sequential()

        model.add(LSTM(units = units1, activation = 'relu', return_sequences=True

                    ,input_shape = (X_train.shape[1], 1)))
        model.add(Dropout(dropout1))
        model.add(LSTM(units=units1))
        model.add(Dense(1))
        model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
        model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=epochs1,
            batch_size=batch_size1,
            callbacks=[early_stopping],
            verbose=0
        )   
        save_model(model,cwd +f'best_in_cluster_{idx}.keras')
    if saving==True:
        with open(cwd + 'best_params_list.pkl', 'wb') as f:
            pickle.dump(best_params_list, f)
    return best_params_list

def hyper_tuning_cnn(train_data, cwd, selected_columns_cluster, selected_columns_cluster_with_info, tuning_trials, saving=True):
    best_params_list = []
    for idx, data_item in enumerate(selected_columns_cluster):


        info_train_cols, _ = split_data(selected_columns_cluster_with_info[idx])
        train_cols, _ = split_data(data_item)

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
        while tuning_trials > len(set(str(t.params) for t in study.trials)):
            try:
                study.optimize(lambda trial: objective_params_cnn(trial, X_train, X_valid, y_train, y_valid, train_data['Close'], train_data_X_train, train_data_X_valid), n_trials=1)
                study.trials_dataframe().fillna(0).sort_values('values_0').to_csv(cwd + f'hypertuning{idx}.csv')
                joblib.dump(study,cwd + f'hypertuningcluster{idx}.pkl')
            except:
                continue

        completed_trials = [t for t in study.trials if t.values is not None]
        completed_trials.sort(key=lambda trial: trial.values, reverse=True) # Sort trials based on objective values

        # Select top 1 trials
        params = completed_trials[0].params
        best_params_list.append(params)
        filters1= params['filters']

        dropout1= params['dropout']

        batch_size1 = params['batch_size']

        epochs1= params['epochs']
        early_stopping =EarlyStopping (monitor='val_loss', patience=5, restore_best_weights=True)
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
        save_model(model,cwd +f'best_in_cluster_{idx}.keras')
    if saving==True:
        with open(cwd + 'best_params_list.pkl', 'wb') as f:
            pickle.dump(best_params_list, f)
    return best_params_list

def hyper_tuning_cnn_lstm(train_data, cwd, selected_columns_cluster, selected_columns_cluster_with_info, tuning_trials, saving=True):
    best_params_list = []
    for idx, data_item in enumerate(selected_columns_cluster):


        info_train_cols, _ = split_data(selected_columns_cluster_with_info[idx])
        train_cols, _ = split_data(data_item)

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
        while tuning_trials > len(set(str(t.params) for t in study.trials)):
            try:
                study.optimize(lambda trial: objective_params_cnn_lstm(trial, X_train, X_valid, y_train, y_valid, train_data['Close'], train_data_X_train, train_data_X_valid), n_trials=1)
                study.trials_dataframe().fillna(0).sort_values('values_0').to_csv(cwd + f'hypertuning{idx}.csv')
                joblib.dump(study,cwd + f'hypertuningcluster{idx}.pkl')
            except:
                continue

        completed_trials = [t for t in study.trials if t.values is not None]
        completed_trials.sort(key=lambda trial: trial.values, reverse=True) # Sort trials based on objective values

        # Select top 1 trials
        params = completed_trials[0].params
        best_params_list.append(params)
        early_stopping =EarlyStopping (monitor='val_loss', patience=5, restore_best_weights=True)
        filters1= params['filters1']
        filters2= params['filters2']
        filters3= params['filters3']
        units= params['units']
        dropout= params['dropout']
        batch_size1 = params['batch_size']
        epochs1= params['epochs']
        input_shape = (X_train.shape[1], 1)
        model = Sequential()
        #add model layers
        model.add(Conv1D(filters1, kernel_size=1, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(1))
        model.add(Conv1D(filters2, kernel_size=1, activation='relu'))
        model.add(MaxPooling1D(2))
        model.add(Conv1D(filters=filters3, kernel_size=3, activation='relu'))
        # model.add(Flatten())
        model.add(LSTM(units,return_sequences=True))
        model.add(Dropout(dropout))
        model.add(LSTM(units,return_sequences=False))
        model.add(Dropout(dropout))
        model.add(Dense(1))
        model.compile(optimizer='RMSprop', loss='mse')
        model.fit(
            X_train, y_train,
            validation_data=(X_valid, y_valid),
            epochs=epochs1,
            batch_size=batch_size1,
            callbacks=[early_stopping],
            verbose=0
        )
        save_model(model,cwd +f'best_in_cluster_{idx}.keras')
    if saving==True:
        with open(cwd + 'best_params_list.pkl', 'wb') as f:
            pickle.dump(best_params_list, f)
    return best_params_list


def test_and_save_xgb(data, cwd, top_10_features_per_cluster, selected_columns_cluster, best_params_list, drop_list):
    return_data = []
    sharpe_list = []
    volatility=[]
    train_data, hold_out = split_data(data)

    for idx, data_item in enumerate(selected_columns_cluster):
        train_cols, hold_out_cols = split_data(data_item)
        # _, info_hold_out_cols= split_data(sorted_selected_columns_cluster_with_info[idx])
    #     _, test_cols = split_data(data_item)
        # optuna_data = scale_data(test_cols)

        temp= hold_out.drop(drop_list, axis=1)
        optuna_data = train_data.drop(drop_list, axis=1)
        X_train, X_valid, y_train, y_valid = train_test_split(optuna_data,
                                                                train_data['Return'],
                                                                test_size=0.5,
                                                                shuffle=False)
    #     info_optuna_data = scale_data(temp)
        temp=scale_data(temp,X_train)
        temp= pd.concat([hold_out[drop_list], temp], axis=1)

        temp.set_index('Unnamed: 0', inplace=True)
        temp.index.name = 'datetime'
        temp.index = pd.to_datetime(temp.index)
        selected_features = hold_out_cols.columns
        test_data=temp
        model = xgb.XGBRegressor()
        # Create and train model
        model.load_model(cwd + f"best_in_cluster_{idx}.json")

        # Make predictions
        # hold_out_cols.columns = optuna_data.columns

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
    df_result.to_csv(cwd + 'xgb_result.csv')
    return (df_result)


def test_and_save_lgbm(data, cwd, top_10_features_per_cluster, selected_columns_cluster, best_params_list, drop_list):
    return_data = []
    sharpe_list = []
    volatility=[]
    train_data, hold_out = split_data(data)

    for idx, data_item in enumerate(selected_columns_cluster):
        train_cols, hold_out_cols = split_data(data_item)
       
        temp= hold_out.drop(drop_list, axis=1)
        optuna_data = train_data.drop(drop_list, axis=1)
        X_train, X_valid, y_train, y_valid = train_test_split(optuna_data,
                                                                train_data['Return'],
                                                                test_size=0.5,
                                                                shuffle=False)
    
        temp=scale_data(temp,X_train)
        temp= pd.concat([hold_out[drop_list], temp], axis=1)

        temp.set_index('Unnamed: 0', inplace=True)
        temp.index.name = 'datetime'
        temp.index = pd.to_datetime(temp.index)
        selected_features = hold_out_cols.columns
        test_data=temp
        model = LGBMRegressor()
        # Create and train model
        
        model= joblib.load(cwd +f'best_in_cluster_{idx}.pkl')
        # Make predictions
        # hold_out_cols.columns = optuna_data.columns

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
    df_result.to_csv(cwd + 'lgbm_result.csv')
    return (df_result)

def test_and_save_rf(data, cwd, top_10_features_per_cluster, selected_columns_cluster, best_params_list, drop_list):
    return_data = []
    sharpe_list = []
    volatility=[]
    train_data, hold_out = split_data(data)

    for idx, data_item in enumerate(selected_columns_cluster):
        train_cols, hold_out_cols = split_data(data_item)
        # _, info_hold_out_cols= split_data(sorted_selected_columns_cluster_with_info[idx])
    #     _, test_cols = split_data(data_item)
        # optuna_data = scale_data(test_cols)

        temp= hold_out.drop(drop_list, axis=1)
        optuna_data = train_data.drop(drop_list, axis=1)
        X_train, X_valid, y_train, y_valid = train_test_split(optuna_data,
                                                                train_data['Return'],
                                                                test_size=0.5,
                                                                shuffle=False)
    #     info_optuna_data = scale_data(temp)
        temp=scale_data(temp,X_train)
        temp= pd.concat([hold_out[drop_list], temp], axis=1)

        temp.set_index('Unnamed: 0', inplace=True)
        temp.index.name = 'datetime'
        temp.index = pd.to_datetime(temp.index)
        selected_features = hold_out_cols.columns
        test_data=temp
        
        # Create and train model
        
        model= joblib.load(cwd +f'best_in_cluster_{idx}.pkl')
        # Make predictions
        # hold_out_cols.columns = optuna_data.columns

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
    df_result.to_csv(cwd + 'rf_result.csv')
    return (df_result)

def test_and_save_lstm(data, cwd, top_10_features_per_cluster, selected_columns_cluster, best_params_list, drop_list):
    return_data = []
    sharpe_list = []
    volatility=[]
    train_data, hold_out = split_data(data)

    for idx, data_item in enumerate(selected_columns_cluster):
        train_cols, hold_out_cols = split_data(data_item)
        # _, info_hold_out_cols= split_data(sorted_selected_columns_cluster_with_info[idx])
    #     _, test_cols = split_data(data_item)
        # optuna_data = scale_data(test_cols)

        temp= hold_out.drop(drop_list, axis=1)
        optuna_data = train_data.drop(drop_list, axis=1)
        X_train, X_valid, y_train, y_valid = train_test_split(optuna_data,
                                                                train_data['Return'],
                                                                test_size=0.5,
                                                                shuffle=False)
    #     info_optuna_data = scale_data(temp)
        temp=scale_data(temp,X_train)
        temp= pd.concat([hold_out[drop_list], temp], axis=1)

        temp.set_index('Unnamed: 0', inplace=True)
        temp.index.name = 'datetime'
        temp.index = pd.to_datetime(temp.index)
        selected_features = hold_out_cols.columns
        test_data=temp
        
        # Create and train model
        
        
        model = load_model(cwd +f"best_in_cluster_{idx}.keras", compile=False)
        model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
        # Make predictions
        # hold_out_cols.columns = optuna_data.columns

        stats1= run_model_backtest_dl( test_data,selected_features,model)
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
    df_result.to_csv(cwd + 'lstm_result.csv')
    return (df_result)

def test_and_save_cnn(data, cwd, top_10_features_per_cluster, selected_columns_cluster, best_params_list, drop_list):
    return_data = []
    sharpe_list = []
    volatility=[]
    train_data, hold_out = split_data(data)

    for idx, data_item in enumerate(selected_columns_cluster):
        train_cols, hold_out_cols = split_data(data_item)
        # _, info_hold_out_cols= split_data(sorted_selected_columns_cluster_with_info[idx])
    #     _, test_cols = split_data(data_item)
        # optuna_data = scale_data(test_cols)

        temp= hold_out.drop(drop_list, axis=1)
        optuna_data = train_data.drop(drop_list, axis=1)
        X_train, X_valid, y_train, y_valid = train_test_split(optuna_data,
                                                                train_data['Return'],
                                                                test_size=0.5,
                                                                shuffle=False)
    #     info_optuna_data = scale_data(temp)
        temp=scale_data(temp,X_train)
        temp= pd.concat([hold_out[drop_list], temp], axis=1)

        temp.set_index('Unnamed: 0', inplace=True)
        temp.index.name = 'datetime'
        temp.index = pd.to_datetime(temp.index)
        selected_features = hold_out_cols.columns
        test_data=temp
        
        # Create and train model
        
        
        model = load_model(cwd +f"best_in_cluster_{idx}.keras", compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        # Make predictions
        # hold_out_cols.columns = optuna_data.columns

        stats1= run_model_backtest_dl( test_data,selected_features,model)
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
    df_result.to_csv(cwd + 'cnn_result.csv')
    return (df_result)

def test_and_save_cnn_lstm(data, cwd, top_10_features_per_cluster, selected_columns_cluster, best_params_list, drop_list):
    return_data = []
    sharpe_list = []
    volatility=[]
    train_data, hold_out = split_data(data)

    for idx, data_item in enumerate(selected_columns_cluster):
        train_cols, hold_out_cols = split_data(data_item)
        # _, info_hold_out_cols= split_data(sorted_selected_columns_cluster_with_info[idx])
    #     _, test_cols = split_data(data_item)
        # optuna_data = scale_data(test_cols)

        temp= hold_out.drop(drop_list, axis=1)
        optuna_data = train_data.drop(drop_list, axis=1)
        X_train, X_valid, y_train, y_valid = train_test_split(optuna_data,
                                                                train_data['Return'],
                                                                test_size=0.5,
                                                                shuffle=False)
    #     info_optuna_data = scale_data(temp)
        temp=scale_data(temp,X_train)
        temp= pd.concat([hold_out[drop_list], temp], axis=1)

        temp.set_index('Unnamed: 0', inplace=True)
        temp.index.name = 'datetime'
        temp.index = pd.to_datetime(temp.index)
        selected_features = hold_out_cols.columns
        test_data=temp
        
        # Create and train model
        
        
        model = load_model(cwd +f"best_in_cluster_{idx}.keras", compile=False)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        # Make predictions
        # hold_out_cols.columns = optuna_data.columns

        stats1= run_model_backtest_dl( test_data,selected_features,model)
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
    df_result.to_csv(cwd + 'cnn_lstm_result.csv')
    return (df_result)