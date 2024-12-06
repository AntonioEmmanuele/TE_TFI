import argparse
import pandas as pd
import numpy as np
from src.te_tfi_fns import list_partitioning, generate_hyperparameter_grid
import os
from tslearn.clustering import TimeSeriesKMeans
from src.te_tfi_model import TE_TFI
from lib.ts_manip import sliding_win_target
#sliding_win_cluster_aware, sliding_win_cluster_aware_multivariate
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
#from lib.ucr_parser import get_series_name
import joblib
import json5
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from multiprocessing import Pool
from tqdm import tqdm

def validate_series_general(configurations, model, x_labels, y_labels, cv_order, starting_percentage):
    starting_rolling = int(starting_percentage * len(x_labels))
    rolling_offset   = int((len(x_labels) - starting_rolling) / cv_order)
    mse = []
    mape = []
    mae = []
    # For each configuration
    for conf in tqdm(configurations, desc = "CPU evaluating configurations"):
        mse_per_conf = []     # Vector containing the metrics for each cv.
        mape_per_conf= []
        mae_per_conf = []
        #print(f"Training for {conf}")
        for i in range(cv_order):
            # Split samples based on the rolling window approach, remember that these samples
            # are the win_trees which are the outputs of the cluster.
            train_x = x_labels[ : starting_rolling + i*rolling_offset]
            train_y = y_labels[ : starting_rolling + i*rolling_offset]
            val_x   = x_labels[starting_rolling + i*rolling_offset : starting_rolling + (i + 1)*rolling_offset]
            val_y   = y_labels[starting_rolling + i*rolling_offset : starting_rolling + (i + 1)*rolling_offset]
            #model = DecisionTreeRegressor(**conf, random_state = 42)
            model.set_params(**conf)
            model.fit(train_x, train_y)
            outcomes = model.predict(val_x)
            mse_per_conf.append(mean_squared_error(y_true = val_y, y_pred = outcomes))
            mae_per_conf.append(mean_absolute_error(y_true = val_y, y_pred = outcomes))
            mape_per_conf.append(mean_absolute_percentage_error(y_true = val_y, y_pred = outcomes))
        #print(f"Params {mse_per_conf} {mae_per_conf} {mape_per_conf}" )
        mse.append(np.mean(mse_per_conf))
        mape.append(np.mean(mape_per_conf))
        mae.append(np.mean(mae_per_conf))
    return np.array(mse), np.array(mape), np.array(mae)

# def hyp_model(  model,
#                 param_grig,
#                 t_X,
#                 t_y,
#                 cv_order = 5,           
#                 tree_cv_perc_start = 0.5,        
#                 n_jobs = os.cpu_count(),    
#                 disable_tqdm = False,       
#                 is_multivariate = False,
#                 target_column = None
#             ):

#     list_params = generate_hyperparameter_grid(param_grid)
#     params_per_cpu = list_partitioning(list_params, n_jobs)
#     pool = Pool(n_jobs)
#     id_results = 0 # MSE MAPE or MAE for CV
#     results = [[], [], []]
#     # Compute execute the cross-validation with pool and select the best series.            
#     args = [[cpu_param, model, t_X, t_y, cv_order, tree_cv_perc_start] for cpu_param in params_per_cpu]
#     #mse, mape, mae = pool.starmap(validate_series, args)
#     results = pool.starmap(validate_series_general, args)
#     # print(len(results[0][0]))
#     # exit(1)
#     results[0] = np.concatenate(results[0]) # MSE
#     results[1] = np.concatenate(results[1]) # MAPE
#     results[2] = np.concatenate(results[2]) # MAE
#     best_result_idx = np.argmin(results[id_results])
#     pool.close()
#     pool.join()
#     return list_params[best_result_idx], results[0][best_result_idx], results[1][best_result_idx], results[2][best_result_idx]


def hyp_model(  model,
                param_grig,
                t_X,
                t_y,
                cv_order = 5,           
                tree_cv_perc_start = 0.5,        
                n_jobs = os.cpu_count(),    
                disable_tqdm = False,       
                is_multivariate = False,
                target_column = None
            ):

    list_params = generate_hyperparameter_grid(param_grid)
    print(len(list_params))
    params_per_cpu = list_partitioning(list_params, n_jobs)
    pool = Pool(n_jobs)
    id_results = 0 # MSE MAPE or MAE for CV
    results = [[], [], []]
    # Compute execute the cross-validation with pool and select the best series.            
    args = [[cpu_param, model, t_X, t_y, cv_order, tree_cv_perc_start] for cpu_param in params_per_cpu]
    #mse, mape, mae = pool.starmap(validate_series, args)
    results = pool.starmap(validate_series_general, args)
    mse_results = []
    mape_results = []
    mae_results = []
    for res in results:
        mse_results.extend(res[0])
        mape_results.extend(res[1])
        mae_results.extend(res[2])

    br = [mse_results, mape_results, mae_results]
    best_result_idx = np.argmin(br[id_results])
    pool.close()
    pool.join()
    return list_params[best_result_idx], br[0][best_result_idx], br[1][best_result_idx], br[2][best_result_idx]



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Script di esempio per accettare argomenti.")
    
    parser.add_argument('--model', type=str, required=True )
    parser.add_argument('--path_stagionality', type=str, required=False, default = None)
    parser.add_argument('--series_path', type=str, required=True)
    parser.add_argument('--lag_percentage', type = float, required = False, default=1.0)
    parser.add_argument('--cv_order', type = int, required = False, default= 5)
    parser.add_argument('--win_size', type=int, required=False, default=50)
    parser.add_argument('--win_tree_perc', type=float, required=False, default=1.0)
    parser.add_argument('--n_jobs', type=int, required=False, default=os.cpu_count())
    parser.add_argument('--out_path', type=str, required=False, default="./")
    parser.add_argument('--is_multivariate', type=int, required=False, default=0)
    parser.add_argument('--target_column', type=str, required=False, default=None)

    args = parser.parse_args()
    
    if args.model == "RT":
        model = DecisionTreeRegressor()
        param_grid = {
            'max_depth': [5, 7, 10, 15, 20],      
            'min_samples_split': [2, 5, 10, 20, 50],  # Min samples to split a node
            'min_samples_leaf': [1, 5, 10, 20, 50],   # Min samples at a leaf node
            'max_features': [ 1.0, 'sqrt', 'log2']    # Max features considered for splitting
        }
    else:
        print("Model not supported !")
        exit(1)
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    
    # Load the series.
    series = np.loadtxt(args.series_path, dtype=float)
    file = args.series_path.split("/")[-1]
    series_name = file[:3]
    len_series = len(series)
    
    # Manage seasonality.
    if args.path_stagionality is not None:
        stag_csv = pd.read_csv(args.path_stagionality)
        has_seasonality = stag_csv.loc[stag_csv['file'] == file, 'has_seasonality'].values[0]
        has_seasonality = stag_csv.loc[stag_csv['file'] == file, 'has_seasonality'].values[0]
        if has_seasonality:
            biggest_lag = stag_csv.loc[stag_csv['file'] == file, 'biggest_lag'].values[0]
            win_size = int(biggest_lag * args.lag_percentage)
        else:
            win_size = args.win_size
    else:
        win_size = args.win_size
    # Params
    train_size = int(0.7 * len(series))
    train_series = series[0 : train_size]
    t_X, t_y = sliding_win_target(series, win_size, 1)
    print("Launching Hyp")
    best_cfg, best_mse, best_mape, best_mae =  hyp_model(model, param_grid, t_X, t_y, 5, 0.5, args.n_jobs, False, False, None)
    print(f"CFG : {best_cfg} MSE : {best_mse} MAPE: {best_mape} MAE: {best_mae}")
    print(best_cfg)
    print(best_mse)
    print(f"Validating the final model.")
    # Validate the final accuracy
    t_X, t_y = sliding_win_target(series, win_size, 1)
    # Obtain results
    definitive_mse, definitive_mape, definitive_mae =  validate_series_general([best_cfg], model, t_X, t_y, cv_order = 5, starting_percentage=0.5)
    print(f"Final Model: Best-CFG MSE : {definitive_mse} MAPE: {definitive_mape} MAE: {definitive_mae}")
    out_dir_model = os.path.join(args.out_path, f"models")
    if not os.path.exists(out_dir_model):
        os.makedirs(out_dir_model) 
    out_models = os.path.join(out_dir_model, f"{series_name}.joblib")
    joblib.dump(model, out_models) 
    # Save the JSON5 file.
    dict_out = {
        "Series": series_name,
        "Hyp MSE" : best_mse,
        "Hyp MAPE": best_mape,
        "Hyp MAE": best_mae,
        "Final MSE": definitive_mse,
        "Final MAPE": definitive_mape,
        "Final MAE": definitive_mae,
        "Seasonal" : has_seasonality,
        "Win" : win_size,
    }
    csv_df = pd.DataFrame(dict_out, index = [0])
    csv_out_stats = os.path.join(args.out_path, "stats_hyp.csv")
    add_header = not os.path.exists(csv_out_stats)
    csv_df.to_csv(csv_out_stats, sep = ",", header=add_header, index=False, mode = "a")
    out_hyp_dir =  os.path.join(args.out_path, f"hy_models")    
    if not os.path.exists(out_hyp_dir):
        os.makedirs(out_hyp_dir)
    out_hyp_model = os.path.join(out_hyp_dir, f"{series_name}.json5")
    with open(out_hyp_model, "w") as f:
        json5.dump(best_cfg, f, indent = 2)