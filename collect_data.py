import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from utils.dbtool import read_from_postgresql_limit, read_feat_from_postgresql_limit, save_to_postgresql, append_to_postgresql, check_database_exists, get_row_difference
from data.data_utils import get_vn30f, add_features, add_finance_features


def check_and_save_data(user, password, host, port, dbname, db_data, data_file, db_feat, feat_file):
    for i in range(len(db_data)):
        if check_database_exists(db_data[i], user, password, host, port, dbname) == False:
            print(f'Saving {db_data[i]}')
            df = pd.read_csv(data_file[i])
            save_to_postgresql(df, db_data[i], user, password, host, port, dbname)

    for i in range(len(db_feat)):
        if check_database_exists(db_feat[i], user, password, host, port, dbname) == False:
            print(f'Saving {db_feat[i]}')
            df = pd.read_csv(feat_file[i])
            save_to_postgresql(df, db_feat[i], user, password, host, port, dbname)

def get_latest_time(table_name, user, password, host, port, dbname, schema):
    last_time_str = ""
    df = read_from_postgresql_limit(table_name, user, password, host, port, dbname, schema, limit=5)  
    for i in range(len(df) - 1, -1, -1):      
        if pd.notna(df.iloc[i]['Date']):
            last_time_str = df.iloc[i]['Date']
            break

    last_time = datetime.strptime(last_time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc) + timedelta(hours=7)
    return last_time, now

def get_latest_time_feat(table_name, user, password, host, port, dbname, schema):
    last_time_str = ""
    df = read_feat_from_postgresql_limit(table_name, user, password, host, port, dbname, schema, limit=5)
    for i in range(len(df) - 1, -1, -1):
        if pd.notna(df.iloc[i]['Date']) and pd.notna(df.iloc[i]['time']):
            last_time_str = df.iloc[i]['Date']+' ' +df.iloc[i]['time']
            break

    last_time = datetime.strptime(last_time_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc) + timedelta(hours=7)
    return last_time, now

def add_feature(df, params, financial_statements, type):
    ti_features = add_features(df, params, type, drop=False)

    data_combined = add_finance_features(ti_features, financial_statements)
    data_combined =  data_combined.fillna(0)
    data_combined.index.name = 'Unnamed: 0'
    return data_combined

def main():
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

    # Load the JSON parameter files
    with open('data/momentum_params.json', 'r') as f:
        momentum_params = json.load(f)
    with open('data/volume_params.json', 'r') as f:
        volume_params = json.load(f)
    with open('data/volatility_params.json', 'r') as f:
        volatility_params = json.load(f)
    with open('data/trend_params.json', 'r') as f:
        trend_params = json.load(f)

    params = [momentum_params, volume_params, volatility_params, trend_params]

    financial_statements = pd.read_csv('financial_indicators.csv')
    financial_statements = financial_statements.rename(columns={'period': 'quarter_label'})
    financial_statements['quarter_label'] = financial_statements['quarter_label'].astype(str)

    check_and_save_data(user, password, host, port, dbname, db_data, data_file, db_feat, feat_file)

    # Append new OHCLV data
    for i in range(len(db_data)):
        last_time, now = get_latest_time(db_data[i], user, password, host, port, dbname, schema)
        if now > last_time:
            new_data = get_vn30f(symbol, resolution[i] ,str(last_time)[:19], str(now)[:19])
            append_to_postgresql(new_data[1:], db_data[i], user, password, host, port, dbname)
    
    # Append new data with features
    for i in range(len(db_feat)):
        last_time_feat, _ = get_latest_time_feat(db_feat[i], user, password, host, port, dbname, schema)

        limit = get_row_difference(last_time_feat, user, password, host, port, dbname, schema, db_data[i])
        old_df = read_from_postgresql_limit(db_feat[i], user, password, host, port, dbname, schema, 1)
        ohclv_data = read_from_postgresql_limit(db_data[i], user, password, host, port, dbname, schema, limit + 50)
        print(limit)
        old_columns = [col for col in old_df.columns if col != 'Unnamed: 0']
        new_df = add_feature(ohclv_data, params, financial_statements, type[i])
        new_df=new_df[old_columns][-limit:]
        new_df.to_csv(f'newdf_{type[i]}.csv')
        try:
            append_to_postgresql(new_data[-limit:], db_feat[i], user, password, host, port, dbname)
        except:
            pass

if __name__ == '__main__':
    main()




    