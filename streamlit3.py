import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, inspect, text
from utils.backtest_DL import *
from utils.backtest import *
from utils.optimize import *
from data.data_utils import *
from datetime import datetime, timedelta, timezone
from utils.dbtool import read_from_postgresql_limit, read_feat_from_postgresql_limit, save_to_postgresql, append_to_postgresql, check_database_exists, get_row_difference, read_feat_from_postgresql_date_time
from data.data_utils import get_vn30f, add_features, add_finance_features
import subprocess
from collect_data import main as collect_main
from utils.dbtool import read_from_postgresql_limit, read_feat_from_postgresql_limit, save_to_postgresql, append_to_postgresql, check_database_exists, get_row_difference

exec(open('collect_data_2.py').read())
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
exec(open('collect_data.py').read())

# Base directory containing models and data
os_dir = "Capstone"
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

model_path = {
    'Random Forest': "Random Forest",
    'XGB': 'xgb',
    'LGBM': 'lgbm',
    'LSTM': 'lstm',
    'CNN': 'cnn',
    'CNN_LSTM': 'cnn_lstm'
}

# Select timeframe
timeframes = ["1 day", "1 hour", "3 minute"]
selected_timeframe = st.selectbox("Select Timeframe", timeframes)

timepath = {
    "1 day": 'day',
    "1 hour": 'hour',
    "3 minute": 'min'
}

# Select cluster
if selected_model == 'XGB':
    if selected_timeframe == '1 hour':
        clusters = [f"cluster_3"]
    else:
        clusters = [f"cluster_1"]
elif selected_model == 'LGBM':
    if selected_timeframe == '1 day':
        clusters = [f"cluster_1"]
    elif selected_timeframe == '1 hour':
        clusters = [f"cluster_{i}" for i in [0, 1]]
    elif selected_timeframe == '3 minute':
        clusters = [f"cluster_{i}" for i in [0, 1]]
elif selected_model == 'Random Forest':
    if selected_timeframe == '3 minute':
        clusters = [f"cluster_2"]
    else:
        clusters = [f"cluster_0"]

if selected_model and selected_timeframe:
    selected_cluster = st.selectbox("Select Cluster", clusters)

# Step 2: Display Training Plot
model_dir = os.path.join(BASE_DIR, model_path[selected_model], timepath[selected_timeframe])

if os.path.exists(model_dir):
    training_plot_files = [f for f in os.listdir(model_dir) if f.startswith(f'training_sharpe_mean_plot_cluster_{selected_cluster[-1:]}') and f.endswith(('.png', '.jpg'))]
    if training_plot_files:
        st.subheader(f"Training Plot for {selected_model} - {selected_timeframe}")
        st.image(os.path.join(model_dir, training_plot_files[0]), caption="Training Plot")
    else:
        st.warning("No training plot found.")

# Step 3: Select current position
position_options = ["None", "Buy", "Sell", "Hold"]
current_position = st.selectbox("Select your current position:", position_options)

# Step 4: Display Prediction Plot
prediction_plot_files = [f for f in os.listdir(model_dir) if f.startswith(f'{selected_cluster.lower()}_prediction') and f.endswith(('.png', '.jpg'))]
if prediction_plot_files:
    for plot_file in prediction_plot_files:
        st.image(os.path.join(model_dir, plot_file), caption=plot_file)
else:
    st.warning(f"No prediction plots found for {selected_cluster}")

# Load metrics data
metrics_file = os.path.join(model_dir, f"result.csv")
if os.path.exists(metrics_file):
    result_df = pd.read_csv(metrics_file)
    selected_metrics = ['Best sharpe','Return (Ann.) [%]','Volatility','Sortino Ratio','Calmar Ratio','Win Rate [%]','Avg. Trade [%]','Max. Drawdown [%]','Generalization Score (GS)',]
    metrics_df = result_df.loc[int(selected_cluster[-1:]), selected_metrics]
    st.subheader(f"Testing metrics for {selected_model} - {selected_timeframe} - {selected_cluster}")
    metrics_df = metrics_df.reset_index().transpose()
    metrics_df.columns = metrics_df.iloc[0]
    metrics_df = metrics_df.drop(metrics_df.index[0]).reset_index(drop=True)
    metrics_df = metrics_df.reset_index(drop=True)
    metrics_df = metrics_df.rename(columns={'index': 'Cluster No'})
    st.write(metrics_df)
else:
    st.warning(f"No metrics file found for {selected_cluster}")

# Step 5: Load data for prediction
train_file = f'final_{timepath[selected_timeframe]}.csv'
data_train = pd.read_csv(train_file)
data_train['Unnamed: 0'] = pd.to_datetime(data_train['Date'] + ' ' + data_train['time'])
data_train = data_train.drop(columns=['Date', 'time'])
data_train = data_train.fillna(0)

train_data, _ = split_data(data_train)
last_date_train =train_data['Unnamed: 0'].iloc[-1]
last_date_train =train_data['Unnamed: 0'].iloc[-1]
last_date_train =train_data['Unnamed: 0'].iloc[-1]
date, time = str(last_date_train).split(" ")


backtesting_data=read_feat_from_postgresql_date_time(f'vn30f1m_{timepath[selected_timeframe]}_feat', user, password, host, port, dbname, schema, date,time)
backtesting_data['Unnamed: 0'] = pd.to_datetime(backtesting_data['Date'] + ' ' + backtesting_data['time'])
backtesting_data = backtesting_data[backtesting_data['Unnamed: 0'] > last_date_train]

# Fetch test data from PostgreSQL
test_data = read_feat_from_postgresql_limit(f'vn30f1m_{timepath[selected_timeframe]}_feat', user, password, host, port, dbname, schema, 11)
test_data['Unnamed: 0'] = pd.to_datetime(test_data['Date'] + ' ' + test_data['time'])
hold_out = test_data.copy()

# Predict data
cwd = f'{os.getcwd()}\\' + f'used_model\\{selected_model}\\{timepath[selected_timeframe]}\\'  # Save path
with open(cwd + 'top_10_list.pkl', 'rb') as f:
    selected_columns_cluster = pickle.load(f)

train_cols, hold_out_cols = split_data(selected_columns_cluster[int(selected_cluster[-1:])])
temp = hold_out.drop(['Close', 'Open', 'High', 'Low', 'Volume', 'Return', 'Unnamed: 0'], axis=1)
optuna_data = train_data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Unnamed: 0'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(optuna_data, train_data['Return'], test_size=0.5, shuffle=False)
temp = temp[X_train.columns]
temp = scale_data(temp, X_train)
temp = pd.concat([hold_out[['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Unnamed: 0']], temp], axis=1)
temp.set_index('Unnamed: 0', inplace=True)
temp.index.name = 'datetime'
temp.index = pd.to_datetime(temp.index)
selected_features = hold_out_cols.columns

temp2 = backtesting_data.drop(['Close', 'Open', 'High', 'Low', 'Volume', 'Return', 'Unnamed: 0'], axis=1)
optuna_data = train_data.drop(['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Unnamed: 0'], axis=1)

temp2 = temp2[X_train.columns]
temp2= scale_data(temp2, X_train)
temp2 = pd.concat([backtesting_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Unnamed: 0']], temp2], axis=1)
temp2.set_index('Unnamed: 0', inplace=True)
temp2.index.name = 'datetime'
temp2.index = pd.to_datetime(temp2.index)

if selected_model == 'LSTM':
    model = load_model(cwd + f"best_in_cluster_{selected_cluster[-1:]}.keras", compile=False)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError()])
    stats1= run_model_backtest_dl( temp2,selected_features,model)
elif selected_model == 'CNN_LSTM' or selected_model == 'CNN':
    model = load_model(cwd + f"best_in_cluster_{selected_cluster[-1:]}.keras", compile=False)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    stats1= run_model_backtest_dl( temp2,selected_features,model)
elif selected_model == 'XGB':
    model = xgb.XGBRegressor()
    model.load_model(cwd + f"best_in_cluster_{selected_cluster[-1:]}.json")
    stats1= run_model_backtest( temp2,selected_features,model)
elif selected_model == 'LGBM' or selected_model == 'Random Forest':
    model = joblib.load(cwd + f'best_in_cluster_{selected_cluster[-1:]}.pkl')
    stats1= run_model_backtest( temp2,selected_features,model)
Return=[]
trades=[]
return_data = []
sharpe_list = []
volatility=[]
Sortino=[]
Calmar=[]
Win_Rate=[]
AvgTrade=[]
MDD=[]
GS=[]
feature=[]
trades.append(stats1)
return_data.append(stats1['Return (Ann.) [%]'])
sharpe_test=stats1['Return (Ann.) [%]']/stats1['Volatility (Ann.) [%]']
sharpe_list.append(float(sharpe_test))
volatility.append(float(stats1['Volatility (Ann.) [%]']))
Sortino.append(float(stats1['Sortino Ratio']))
Calmar.append(float(stats1['Calmar Ratio']))
Win_Rate.append(float(stats1['Win Rate [%]']))
AvgTrade.append(float(stats1['Avg. Trade [%]']))
MDD.append(float(stats1['Max. Drawdown [%]']))
tunning = joblib.load(open(cwd + f"hypertuningcluster{int(selected_cluster[-1:])}.pkl", "rb"))
tunning=tunning.trials_dataframe()
best_tunning_trial=tunning[-1:]
sharpe_train=best_tunning_trial['values_0'].values/best_tunning_trial['values_1'].values
sharpe_train=sharpe_train[0]
sharpe_train

GS.append(float(abs((sharpe_test / sharpe_train)-1)))
pred = model.predict(temp[selected_features])

last_10_predictions = pd.DataFrame({
    'Time': test_data['Unnamed: 0'][-10:].reset_index(drop=True),
    'Close Price':test_data['Close'][-10:].reset_index(drop=True),
    "Model Prediction's Close Price": (((pred*test_data['Close'])+test_data['Close'])[:10]),
    'Actual Return': test_data['Return'].shift(1)[-10:].reset_index(drop=True),
    "Model Prediction's Return": pred[:10],
})

new_row =  pd.DataFrame({
    'Time': "Next Trading Section",
    'Close Price': "Unknown",  
    "Model Prediction's Close Price": (((pred*test_data['Close'])+test_data['Close'])[-1:]),
    'Actual Return':"Unknown",
    "Model Prediction's Return": pred[-1:],
})
last_10_predictions=pd.concat([last_10_predictions, new_row], ignore_index=True)
current_metrics=pd.DataFrame({
    'Sharpe':sharpe_list,
"Return (Ann.) [%]": return_data,
'Volatility':volatility,
'Sortino Ratio':Sortino,
'Calmar Ratio':Calmar,
'Win Rate [%]':Win_Rate,
'Avg. Trade [%]':AvgTrade,
'Max. Drawdown [%]':MDD,
'Generalization Score (GS)':GS,
})
# Add Recommended Next Step
def recommend_next_step(current_position, prediction):
    if current_position == "Buy":
        return "Hold" if prediction >= 0 else "Sell"
    elif current_position == "Sell":
        return "Hold" if prediction <= 0 else "Buy"
    elif current_position == "None":
        return "Buy" if prediction > 0 else "Sell"
    return "Hold"

last_10_predictions["Recommended Next Step"] = last_10_predictions.apply(
    lambda row: recommend_next_step(current_position, row["Model Prediction's Return"]), axis=1
)

st.title("Actual vs Predicted Return")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(hold_out['Unnamed: 0'][-10:], test_data['Return'][-10:], label='Actual Return', color='gray', linewidth=2)
ax.plot(hold_out['Unnamed: 0'][-10:], pred[-10:], label='Model Prediction', linewidth=2)
ax.set_title(f'Actual vs Predicted Return For the last 10 times')
ax.set_xlabel('Date')
ax.set_ylabel('Return')
ax.legend()
st.pyplot(fig)

# Display predictions and recommended next step
st.write(last_10_predictions)
st.write("Lstest metrics of the model since test data to present")
st.write(current_metrics)



row = current_metrics.iloc[0].to_dict()

# Chuyển từ điển thành chuỗi theo định dạng mong muốn
row_str = ', '.join([f"{key}: {value}" for key, value in row.items()])

question = st.text_input("Ask your question to the chatbot:")
if st.button("Send Question"):
    if question.strip():
        try:
            # Run the chatbot script with the question as input
            result = subprocess.run(
                ["python", "llm.py", question, row_str],
                text=True,
                capture_output=True,
                check=True,
                encoding="utf-8",  # Đảm bảo sử dụng UTF-8
                errors="replace"  # Thay thế ký tự không hợp lệ
            )
            # Capture the response from the script's output
            response = result.stdout
            response_start = response.find("Response") + len("Response")
            true_response = response[response_start:].strip()
            cleaned_response = true_response.split("<|eot_id|>")[0].strip()

            st.success(f"Trả lời {cleaned_response}")
        except subprocess.CalledProcessError as e:
            st.error("Error occurred while running the chatbot script.")
            st.error(e)
    else:
        st.warning("Please enter a question before sending.")