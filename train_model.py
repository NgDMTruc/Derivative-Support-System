
import os 
import pandas as pd
import numpy as np

from data.data_utils import draw_corr
from utils.optimize import *


def main():
    ### Prepare variables
    # cwd = f'{os.getcwd()}\\'
                    #### Change model and timeframe here ####
    cwd = f'model\\cnn_lstm\\day\\'
    feat_trials = 8
    tuning_trials = 10

    # Prepare data
    data = pd.read_csv('final_day.csv')
    data =  data.fillna(0)
    try:
        data['Unnamed: 0'] = pd.to_datetime(data['Date'] + ' ' + data['time'])
        data = data.drop(columns=['Date', 'time'])
    except:
        pass

    # Columns to drop in some functions
    drop_list = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Unnamed: 0']
    new_df_no_close_col = data.drop(drop_list, axis=1)

    ### Feature selection stage
                            #### Change model here ####
    train_data, sort_df, top_trials, top_features_list = feature_select_cnn_lstm(data, cwd, new_df_no_close_col, feat_trials)

    ### ONC
                            #### Change model here ####
    top_pnl = retrieve_top_pnl_cnn_lstm(data, top_features_list, drop_list)
    # FREQUENCY FEATURE TABLE
    # Before onc
    correlation_matrix = np.corrcoef(top_pnl)
    corr = pd.DataFrame(correlation_matrix)
    corr = corr.fillna(0)
    draw_corr(corr)
    # After onc
    corrNew, clstrsNew, silhNew = clusterKMeansTop(corr)
    draw_corr(corrNew)
    # Saving clusters
    top_10_features_per_cluster, selected_columns_cluster, selected_columns_cluster_with_info = process_clusters_and_save(clstrsNew, top_trials, new_df_no_close_col, data, cwd, drop_list, output_folder="xgb_day_cluster")
    
    ### Hyperparameters tuning
                        #### Change model here ####
    best_params_list = hyper_tuning_cnn_lstm(train_data, cwd, selected_columns_cluster, selected_columns_cluster_with_info, tuning_trials)
    
    ### Testing and saving
                        #### Change model here ####
    test_and_save_cnn_lstm(data, cwd, top_10_features_per_cluster, selected_columns_cluster, best_params_list, drop_list)

if __name__ == "__main__":
    main()
