import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, inspect, text
from utils.backtest_DL import  *
from utils.backtest import  *
from utils.optimize import *
from data.data_utils import *
from datetime import datetime, timedelta, timezone
from utils.dbtool import read_from_postgresql_limit, read_feat_from_postgresql_limit, save_to_postgresql, append_to_postgresql, check_database_exists, get_row_difference,read_feat_from_postgresql_date
from data.data_utils import get_vn30f, add_features, add_finance_features
import subprocess
from collect_data import main as collect_main

exec(open('collect_data.py').read())
# Base directory containing models and data
os_dir="Capstone"
BASE_DIR = "used_model"
host = 'localhost'
port = '5432' 
dbname = 'postgres'
user = 'postgres'
password = 'postgres'
schema = 'public'
db_data = ['vn30f1m_min', 'vn30f1m_hour', 'vn30f1m_day']
db_feat = ['vn30f1m_min_feat', 'vn30f1m_hour_feat', 'vn30f1m_day_feat']
data_file = ['vn30f1m_3min.csv', 'vn30f1m_1hour.csv', 'vn30f1m_1day.csv']
feat_file = ['final_min.csv', 'final_hour.csv', 'final_day.csv']
resolution= ['3','1H','1D']
type = ["min", "hour", "day"]
symbol = 'VN30F1M'
# Step 1: Model, Timeframe, and Cluster Selection
st.title("Model Prediction Interface")

# Select model
models = [folder for folder in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, folder))]
selected_model = st.selectbox("Select Model", models)
    
model_path={
    'Random Forest':"Random Forest",
    'XGB':'xgb',
    'LGBM':'lgbm',
    'LSTM':'lstm',
    'CNN':'cnn',
    'CNN_LSTM':'cnn_lstm'
}
# Select timeframe
timeframes = ["1 day", "1 hour", "3 minute"]
selected_timeframe = st.selectbox("Select Timeframe", timeframes)
timepath={
    "1 day":'day',
    "1 hour":'hour',
    "3 minute":'min'
}
# Select cluster
if selected_model =='XGB':
    if selected_timeframe =='1 hour':
        clusters = [f"cluster_3"]  # Adjust the number of clusters as necessary
    else:
        clusters = [f"cluster_1"] 
elif selected_model =='LGBM':
    if selected_timeframe =='1 day':
        clusters = [f"cluster_1"] 
    if selected_timeframe =='1 hour':
        clusters = [f"cluster_{i}" for i in [0, 1]] 
    if selected_timeframe =='3 minute':
        clusters = [f"cluster_{i}" for i in [0, 2]] 
elif selected_model =='Random Forest':
    if selected_timeframe =='3 minute':
        clusters = [f"cluster_2"] 
    else: 
        clusters = [f"cluster_0"]
if selected_model and selected_timeframe:
    selected_cluster = st.selectbox("Select Cluster", clusters)

# Step 2: Display Training Plot
model_dir = os.path.join(BASE_DIR, model_path[selected_model], timepath[selected_timeframe])
if os.path.exists(model_dir):
    # Display single training plot (assuming one image file for training)
    training_plot_files = [f for f in os.listdir(model_dir) if f.startswith(f'training_sharpe_mean_plot_cluster_{selected_cluster[-1:]}') and f.endswith(('.png', '.jpg'))]
    if training_plot_files:
        st.subheader(f"Training Plot for {selected_model} - {selected_timeframe}")
        st.image(os.path.join(model_dir, training_plot_files[0]), caption="Training Plot")
    else:
        st.warning("No training plot found.")
st.subheader(f"Prediction Plots for {selected_model} - {selected_timeframe} - {selected_cluster}")
prediction_plot_files = [f for f in os.listdir(model_dir) if f.startswith(f'{selected_cluster.lower()}_prediction') and f.endswith(('.png', '.jpg'))]
if prediction_plot_files:
    for plot_file in prediction_plot_files:
        st.image(os.path.join(model_dir, plot_file), caption=plot_file)
else:
    st.warning(f"No prediction plots found for {selected_cluster}")
metrics_file = os.path.join(model_dir, f"result.csv")
if os.path.exists(metrics_file):

    result_df = pd.read_csv(metrics_file)
    selected_metrics=['Best sharpe','Return (Ann.) [%]','Volatility','Sortino Ratio','Calmar Ratio','Win Rate [%]','Avg. Trade [%]','Max. Drawdown [%]','Generalization Score (GS)',]
    metrics_df= result_df.loc[int(selected_cluster[-1:]),selected_metrics]
    st.subheader(f"Metrics for {selected_model} - {selected_timeframe} - {selected_cluster}")
    metrics_df = metrics_df.reset_index().transpose()
    metrics_df.columns = metrics_df.iloc[0]
    metrics_df = metrics_df.drop(metrics_df.index[0]).reset_index(drop=True)
    metrics_df = metrics_df.reset_index(drop=True)
    metrics_df = metrics_df.rename(columns={'index': 'Cluster No'})
    st.write(metrics_df)
else:
    st.warning(f"No metrics file found for {selected_cluster}")


# Step 3: Display Predicted and Actual Returns
## Prepare data
# test_file = f"test_{timepath[selected_timeframe]}.csv" # Assuming a common 'test.csv' file for test data
# test_data = pd.read_csv(test_file)
train_file = f'final_{timepath[selected_timeframe]}.csv'
data_train = pd.read_csv(train_file)
data_train['Unnamed: 0'] = pd.to_datetime(data_train['Date'] + ' ' + data_train['time'])
data_train = data_train.drop(columns=['Date', 'time'])
data_train =  data_train.fillna(0)

train_data, _ = split_data(data_train)

test_data= read_feat_from_postgresql_limit(f'vn30f1m_{timepath[selected_timeframe]}_feat', user, password, host, port, dbname, schema, 10)
test_data['Unnamed: 0'] = pd.to_datetime(test_data['Date'] + ' ' + test_data['time'])
hold_out  = test_data.copy()

## Predict data
cwd = f'{os.getcwd()}\\' + f'model\\{selected_model}\\{timepath[selected_timeframe]}\\' # Save path
with open(cwd + 'top_10_list.pkl', 'rb') as f:
    selected_columns_cluster = pickle.load(f)
train_cols, hold_out_cols = split_data(selected_columns_cluster[int(selected_cluster[-1:])])
temp= hold_out.drop(['Close', 'Open','High','Low','Volume', 'Return','Unnamed: 0'], axis=1)
optuna_data = train_data.drop(['Open','High','Low','Close','Volume', 'Return','Unnamed: 0'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(optuna_data,
                                                        train_data['Return'],
                                                        test_size=0.5,
                                                        shuffle=False)
# st.write(temp)
temp= temp[X_train.columns]

temp=scale_data(temp,X_train)
temp= pd.concat([hold_out[[ 'Open','High','Low','Close','Volume', 'Return','Unnamed: 0']], temp], axis=1)
temp.set_index('Unnamed: 0', inplace=True)
temp.index.name = 'datetime'
temp.index = pd.to_datetime(temp.index)
selected_features = hold_out_cols.columns

if selected_model == 'LSTM':
    model = load_model(cwd +f"best_in_cluster_{selected_cluster[-1:]}.keras", compile=False)
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    # stats1= run_model_backtest_dl( temp,selected_features,model)
elif selected_model == 'CNN_LSTM' or selected_model=='CNN':
    model = load_model(cwd +f"best_in_cluster_{selected_cluster[-1:]}.keras", compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    # stats1= run_model_backtest_dl( temp,selected_features,model)

elif selected_model == 'XGB':
    model = xgb.XGBRegressor()
    model.load_model(cwd + f"best_in_cluster_{selected_cluster[-1:]}.json")
    # stats1= run_model_backtest( temp,selected_features,model)
    
elif selected_model == 'LGBM' or selected_model=='Random Forest':
    model= joblib.load(cwd +f'best_in_cluster_{selected_cluster[-1:]}.pkl')
    # stats1= run_model_backtest( temp,selected_features,model)


pred = model.predict(temp[selected_features])
last_10_predictions = pd.DataFrame({
    'Time': test_data['Unnamed: 0'][-10:],
    'Actual Return': test_data['Return'][-10:],
    "Model Prediction's Return": pred[-10:],
    # "Recommended Action":list_result[-10:]
})   
st.title("Actual vs Predicted Return")
fig, ax = plt.subplots(figsize=(12, 6))
# Vẽ Return thực tế
ax.plot(hold_out['Unnamed: 0'], test_data['Return'][-10:], label='Actual Return', color='gray', linewidth=2)

# Vẽ Return dự đoán của model
ax.plot(hold_out['Unnamed: 0'], pred, label='Model Prediction', linewidth=2)

# Thêm tiêu đề và nhãn
ax.set_title(f'Actual vs Predicted Return For the last 10 times')
ax.set_xlabel('Date')
ax.set_ylabel('Return')
ax.legend()
st.pyplot(fig)

## Print the result
# st.subheader("Test Data Predicted Returns and Actual Timestamps")
st.write(last_10_predictions)

