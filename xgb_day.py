import os 
import pickle
import pandas as pd
import numpy as np
from data.data_utils import draw_corr
from utils.optimize import feature_select_xgb, clusterKMeansTop, retrieve_top_pnl, process_clusters_and_save, hyper_tuning_xgb, test_and_save_xgb

def train(data, cwd, feat_trials, tuning_trials):
    # Columns to drop in some functions
    drop_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Unnamed: 0']
    drop_list_tuning = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return']
    new_df_no_close_col = data.drop(drop_list, axis=1)

    ### Feature selection stage
    train_data, sort_df, top_trials, top_features_list = feature_select_xgb(data, cwd, new_df_no_close_col, feat_trials)

    ### ONC
    top_pnl = retrieve_top_pnl(data, top_features_list, drop_list)
    # FREQUENCY FEATURE TABLE
    # Before onc
    correlation_matrix = np.corrcoef(top_pnl)
    corr = pd.DataFrame(correlation_matrix)
    corr = corr.fillna(0)
    draw_corr(corr)
    # After onc
    corrNew, clstrsNew, _ = clusterKMeansTop(corr)
    draw_corr(corrNew)
    # Saving clusters
    top_10_features_per_cluster, selected_columns_cluster, selected_columns_cluster_with_info = process_clusters_and_save(clstrsNew, top_trials, new_df_no_close_col, data, cwd, drop_list)
    
    ### Hyperparameters tuning
    best_params_list = hyper_tuning_xgb(train_data, cwd, selected_columns_cluster, selected_columns_cluster_with_info, tuning_trials, drop_list_tuning)
    
    return top_10_features_per_cluster, selected_columns_cluster, best_params_list
def test(data, cwd, top_10_features_per_cluster, selected_columns_cluster, best_params_list):
    ### Testing and saving
    if 'Unnamed: 0' in data.columns:
        drop_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Unnamed: 0']
    else:
        drop_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return']
    df = test_and_save_xgb(data, cwd, top_10_features_per_cluster, selected_columns_cluster, best_params_list, drop_list)
    return df

def main(feat_trials, tuning_trials, training=True):
    ### Prepare variables
    cwd = f'{os.getcwd()}\\' + 'model\XGB\day\\' # Save path
    # Prepare data
    data = pd.read_csv('final_day.csv')
    data =  data.fillna(0)
    try:
        data['Unnamed: 0'] = pd.to_datetime(data['Date'] + ' ' + data['time'])
        data = data.drop(columns=['Date', 'time'])
    except:
        pass

    if training==True:
        top_10_features_per_cluster, selected_columns_cluster, best_params_list = train(data, cwd, feat_trials, tuning_trials)
    else:
        with open(cwd + 'top_10_list.pkl', 'rb') as f:
            selected_columns_cluster = pickle.load(f)
        with open(cwd + 'top_10_features_per_cluster.pkl', 'rb') as f:
            top_10_features_per_cluster = pickle.load(f)
        with open(cwd + 'best_params_list.pkl', 'rb') as f:
            best_params_list = pickle.load(f)

    test(data, cwd, top_10_features_per_cluster, selected_columns_cluster, best_params_list)

if __name__ == "__main__":
    feat_trials = 5
    tuning_trials = 3
    training = True
    main(feat_trials, tuning_trials, training)
