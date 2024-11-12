import sys
import os

# Thêm đường dẫn của thư mục Capstone vào sys.path
sys.path.append(os.path.abspath('../'))

import pandas as pd
import requests
from datetime import datetime, timedelta, timezone

from Capstone.indicators.volatility import *  
from Capstone.indicators.volume import *      
from Capstone.indicators.other import *      
from Capstone.indicators.trend import *      

def get_vn30f(symbol, resolution, start_time=None, now_time=None):
    if start_time:
      start_time = datetime.strptime(start_time, '%Y-%m-%d')
      start_time = int((start_time - timedelta(hours=7)).timestamp())
    else:
        start_time = 0

    if now_time:
      now_time = datetime.strptime(now_time, '%Y-%m-%d')
      now_time = int((now_time - timedelta(hours=7)).timestamp())
    else:
      now_time = 9999999999

    def vn30f():
            return requests.get(f"https://services.entrade.com.vn/chart-api/chart?from={start_time}&resolution={resolution}&symbol={symbol}&to={now_time}").json()
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    ohlc_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',}
    vn30fm = pd.DataFrame(vn30f()).iloc[:,:6]
    vn30fm['t'] = vn30fm['t'].astype(int).apply(lambda x: datetime.utcfromtimestamp(x) + timedelta(hours = 7))
    vn30fm.columns = ['Date','Open','High','Low','Close','Volume']
    dt_object = datetime.utcfromtimestamp(start_time) + timedelta(hours = 7)
    now_object = datetime.utcfromtimestamp(now_time) + timedelta(hours = 7)

    print(f'===> Data {symbol} from {dt_object} to {now_object} has been appended ')

    return vn30fm

def get_data(start_time=None, now_time='2024-08-31', symbol='VN30F1M'):
  ''' Lấy dữ liệu và lưu thành file '''
  resolution= ['3','1H','1D']
  df_min = get_vn30f(symbol='VN30F1M', resolution=resolution[0], start_time=start_time, now_time=now_time)
  df_hour = get_vn30f(symbol='VN30F1M', resolution=resolution[1], start_time=start_time, now_time=now_time)
  df_day = get_vn30f(symbol='VN30F1M', resolution=resolution[2], start_time=start_time, now_time=now_time)

  df_min.to_csv('vn30f1m_3min.csv', index=False)
  df_hour.to_csv('vn30f1m_1hour.csv', index=False)
  df_day.to_csv('vn30f1m_1day.csv', index=False)

def process_data(data):
    ''' Xử lý dữ liệu bị trùng và thiếu '''
    data.set_index('Date', inplace =True)
    data.columns = ['Open','High','Low','Close','Volume']

    data['Date'] = [str(i)[:10] for i in data.index]
    data['time'] = [str(i)[11:] for i in data.index]

    data = data[~data.index.duplicated(keep='first')] # Handling duplicate
    data_model = data.pivot(index = 'Date', columns = 'time', values = ['Open','High','Low','Close','Volume']).ffill(axis = 1).stack().reset_index() # Handling missing values

    # Convert 'time' to datetime to filter rows before 9 AM
    data_model['time'] = pd.to_datetime(data_model['time'], format='%H:%M:%S').dt.time
    data_model = data_model[data_model['time'] >= pd.to_datetime('09:00:00').time()]

    return data_model


def drop_high_corr_columns(df, threshold=0.9, rolling_window=1):
  ''' Xóa các cột có correlation cao với nhau, giữ cột đầu tiên'''
  # ohlcv_columns = {'Open','High','Low','Close','Volume','Unnamed: 0'}
  ohlcv_columns = {'Date','time', 'Open','High','Low','Close','Volume'}
    # Identify non-OHLCV and non-date/time columns
  non_corr_columns = [col for col in df.columns if col not in ohlcv_columns]

  # Compute the correlation matrix only for non-OHLCV and non-date/time columns
  corr_matrix = df[non_corr_columns].corr().abs()

  # Create a set to keep track of columns to drop
  to_drop = set()

  # Iterate over the upper triangle of the correlation matrix
  for i in range(len(corr_matrix.columns)):
      for j in range(i + 1, len(corr_matrix.columns)):
          if corr_matrix.iloc[i, j] > threshold:
              # If the correlation is higher than the threshold, mark the column with the higher index to drop
              col_to_drop = corr_matrix.columns[j]
              if col_to_drop not in ohlcv_columns:
                  to_drop.add(col_to_drop)

  # Drop the columns from the DataFrame               
  df_dropped = df.drop(columns=to_drop)
  df_dropped['Return'] = ( df_dropped['Close'].shift(-rolling_window) -  df_dropped['Close'])/df_dropped['Close']
  df_dropped =  df_dropped.fillna(0)

  return df_dropped

def generate_features(data, params_dict, dataset_name):
    ''' Tạo feature cho dữ liệu'''
    features = pd.DataFrame(index=data.index)  # Placeholder for features

    for indicator_name, param_sets in params_dict.items():
        print(f"Applying {indicator_name} to {dataset_name} dataset...")
        for param_name, param_values in param_sets[dataset_name].items():
            for param_value in param_values:
                # Dynamically call the corresponding function and apply to the dataset
                indicator_func = globals().get(indicator_name.lower().replace(" ", "_"))  # Get the function by name

                if indicator_func:
                    try:
                        # If the function has multiple return values, unpack them
                        result = indicator_func(data, **{param_name: param_value})
                        if isinstance(result, tuple):
                            for idx, res in enumerate(result):
                                col_name = f"{indicator_name}_{param_name}_{param_value}_{idx}"
                                features[col_name] = res
                        else:
                            col_name = f"{indicator_name}_{param_name}_{param_value}"
                            features[col_name] = result
                    except Exception as e:
                        print(f"Error applying {indicator_name} with {param_name}={param_value}: {e}")

    return features

def add_features(data, params, type, rolling_window=1):
    ''' Thêm feature vào dữ liệu và xử lý'''
    data = process_data(data)
    for param in params:
        feature = generate_features(data, param, type)
        train_features = pd.concat([data,feature], axis=1)
    train_features['Return'] = (train_features['Close'].shift(-rolling_window) -  train_features['Close'])/train_features['Close']
    train_features =  train_features.fillna(0)

    train_features=drop_high_corr_columns(train_features)

    return train_features

def data_backtest(data):
    '''Đổi data thành dạng backtest'''
    new_data = data
    new_data.index = pd.to_datetime(data['Date'])
    new_data.index.name = None
    return new_data

def label_quarter(df):

    # Get the datetime index column
    df = df.copy()
    df['date'] = df.index


    # Calculate the 21st day of the first month in each quarter
    def get_quarter_label(date):
        if date.month in [1, 2, 3]:
            start_of_quarter = pd.Timestamp(year=date.year, month=1, day=21)
            if date >= start_of_quarter:
                return f"{date.year}Q1"
            else:
                return f"{date.year - 1}Q4"

        elif date.month in [4, 5, 6]:
            start_of_quarter = pd.Timestamp(year=date.year, month=4, day=21)
            if date >= start_of_quarter:
                return f"{date.year}Q2"
            else:
                return f"{date.year}Q1"

        elif date.month in [7, 8, 9]:
            start_of_quarter = pd.Timestamp(year=date.year, month=7, day=21)
            if date >= start_of_quarter:
                return f"{date.year}Q3"
            else:
                return f"{date.year}Q2"

        elif date.month in [10, 11, 12]:
            start_of_quarter = pd.Timestamp(year=date.year, month=10, day=21)
            if date >= start_of_quarter:
                return f"{date.year}Q4"
            else:
                return f"{date.year}Q3"


    df['quarter_label'] = df['date'].apply(get_quarter_label)

    return df['quarter_label']

# Additional function to apply exponential decay
def apply_exponential_decay(df, lambda_value):
    # Ensure 'date' is in datetime format
    df = df.copy()
    df['date'] = pd.to_datetime(df.index)

    # Determine start of quarter for each row
    df['quarter_start'] = df['date'].apply(lambda d: pd.Timestamp(year=d.year, month=(d.month - 1) // 3 * 3 + 1, day=1))

    # Calculate the time passed within each quarter (in days for daily data)
    df['days_within_quarter'] = (df['date'] - df['quarter_start']).dt.days

    # Apply exponential decay to each financial feature
    financial_features = [col for col in df.columns if col not in ['date', 'quarter_label', 'quarter_start', 'days_within_quarter']]
    for feature in financial_features:
        df[feature] = df[feature] * np.exp(-lambda_value * df['days_within_quarter'])

    # Drop helper columns to return the final DataFrame
    df = df.drop(columns=['quarter_start', 'days_within_quarter', 'date'])
    return df

def add_finance_features(data, financial_statements):
    data_bt = data_backtest(data)
    data_bt = data_bt[['Close']].copy()
    data_bt['quarter_label'] = label_quarter(data_bt)

    # Step 1: Convert index to 'date' column in DataFrames and keep only necessary columns
    df_bt = data_bt.reset_index()[['index', 'Close', 'quarter_label']].copy()
    df_bt = df_bt.rename(columns={'index': 'date'})
    df_bt['quarter_label'] = df_bt['quarter_label'].astype(str)

    # Step 2: Merge DataFrames based on 'quarter_label'
    merged_df = pd.merge(df_bt, financial_statements, on='quarter_label', how='left')

    # Step 3: After merging, convert column 'date' to index again
    merged_df = merged_df.set_index('date')

    # Step 4: Fill in NaN values ​​equal to the previous quarter's value
    merged_df = merged_df.fillna(method='ffill')

    # Step 5: Apple exponential decay
    merged_df = apply_exponential_decay(merged_df, lambda_value=0.01)  # Daily decay rate
    
    columns_to_drop = ['Close', 'quarter_label']
    data_clean = merged_df.drop(columns=columns_to_drop, errors='ignore')

    data_combined = pd.concat([data_backtest(data), data_clean], axis=1)

    return data_combined
