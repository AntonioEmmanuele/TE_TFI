import argparse
import pandas as pd
import numpy as np
from src.te_tfi_fns import list_partitioning, generate_hyperparameter_grid
import os
from tslearn.clustering import TimeSeriesKMeans
from src.te_tfi_model import TE_TFI
from lib.ts_manip import sliding_win_target, custom_mape
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json5
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from multiprocessing import Pool
from tqdm import tqdm
import time 
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import random

def get_series(folder, series_name):
    list_series = os.listdir(folder)
    for series_file in list_series:
        if series_file[:3] == series_name:
            return series_file
    assert 1 == 0, "Malformatted series"
    return None

def validate_timings(model, t_X):
    # # Time validation on a rolling window of 10%
    # results = []
    # average_single = []
    # # Perform predictions in 10% increments
    # for percentage in range(10, 101, 10):  # 10%, 20%, ..., 100%
    #     # Determine the number of test samples to include
    #     num_samples = int(len(t_X) * (percentage / 100))
    #     #X_subset = t_X[:num_samples]
    #     X_subset = t_X[:num_samples]
    #     #y_subset = y_test[:num_samples]

    #     # Measure inference time
    #     start_time = time.perf_counter()
    #     predictions = model.predict(X_subset)
    #     end_time = time.perf_counter()
    #     inference_time_microseconds = (end_time - start_time) * 1e6  # Convert to microseconds
    #     results.append(inference_time_microseconds)
    #     average_single.append(inference_time_microseconds/len(X_subset))
    # return results, average_single
    num_samples = int(len(t_X) * (15.0 / 100))
    X_subset = t_X[:num_samples]
    st = 0
    for x in X_subset:
        start_time = time.perf_counter()
        predictions = model.predict([x]) # WORKS ONLY ON THE SAME LEN WIN_CLUST WIN_TREE !!!
        end_time = time.perf_counter()
        inference_time_microseconds = (end_time - start_time) * 1e6  # Convert to microseconds
        st += inference_time_microseconds
    return st / len(X_subset)
 

def validate_timings_te_tfi(model, t_X):
    
    num_samples = int(len(t_X) * (15.0 / 100))
    X_subset = t_X[:num_samples]
    st = 0
    for x in X_subset:
        start_time = time.perf_counter()
        #predictions, _ = model.predict_clust_ts(X_subset, X_subset ) # WORKS ONLY ON THE SAME LEN WIN_CLUST WIN_TREE !!!
        idx = model.cluster.predict([x])
        model.trees[idx[0]].predict([x])
        end_time = time.perf_counter()
        inference_time_microseconds = (end_time - start_time) * 1e6  # Convert to microseconds
        st += inference_time_microseconds
    return st / len(X_subset)
    #    # Time validation on a rolling window of 10%
    # results = []
    # average_single = []
 
    #        
    # for percentage in range(10, 101, 10):  # 10%, 20%, ..., 100%
    
    # # Perform predictions in 10% increments
    # for percentage in range(10, 101, 10):  # 10%, 20%, ..., 100%
    #     # Determine the number of test samples to include
    #     num_samples = int(len(t_X) * (percentage / 100))
    #     #X_subset = t_X[:num_samples]
    #     X_subset = t_X[:num_samples]
        
    #     # Measure inference time
    #     start_time = time.perf_counter()
    #     predictions, _ = model.predict_clust_ts(X_subset, X_subset ) # WORKS ONLY ON THE SAME LEN WIN_CLUST WIN_TREE !!!
    #     end_time = time.perf_counter()
    #     inference_time_microseconds = (end_time - start_time) * 1e6  # Convert to microseconds
    #     results.append(inference_time_microseconds)
    #     average_single.append(inference_time_microseconds/len(X_subset))
#    return results, average_single

series_path = "./datasets/processed"
in_dir = "./experiments/ucr_no_preprocess_timings"
names_pth =  os.path.join(in_dir, "series_names.json5")
files_pth =  os.path.join(in_dir, "series_paths.json5")
out_dir = "./experiments/ucr_no_preprocess_timings/results"
hy_models = "hy_models"
stats_hyp = "stats_hyp.csv"
models = ["RT", "RF", "XGB", "Dual-Stage"]

with open(files_pth, 'r') as f:
    series_paths = json5.load(f)

with open(names_pth, 'r') as f:
    series_names = json5.load(f)

parameter_paths = ["./experiments/ucr_no_preprocess/hyp_rt_ucr/", 
                   "./experiments/ucr_no_preprocess/hyp_randomforest_ucr/", 
                   "./experiments/ucr_no_preprocess/hyp_xgboost_ucr/",
                   "./experiments/ucr_no_preprocess/hyp_test_ucr_new/",
                   ]
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
    
models_stats = []
for i in range(0, len(parameter_paths)):
    models_stats.append(pd.read_csv(os.path.join(parameter_paths[i],"stats_hyp.csv"), dtype={'Series': str}))
models_stats[-1] = models_stats[-1].loc[models_stats[-1].groupby('Series')['Final MSE'].idxmin()]
models_stats[-1] = models_stats[-1].reset_index(drop=True)

# Create the series for the window
for series_name, series_file_path  in tqdm(zip(series_names, series_paths), desc = "Testing models"):    
    series = np.loadtxt(series_file_path)
    train_size = int(0.7 * len(series))
    test_series = series[train_size : ]
    print(f"\n Series Name {series_name} \n")
    for model_idx in range(0, len(parameter_paths)):
    
        df = models_stats[model_idx]
        
        if models[model_idx] != "Dual-Stage":
            window = df.loc[df['Series'] == series_name, 'Win'].iloc[0]
            t_X, t_y = sliding_win_target(test_series, window, 1)
        else:
            num_clusters = df.loc[df['Series'] == series_name, 'NumClusters'].iloc[0]
            lag_percentage = df.loc[df['Series'] == series_name, 'Perc'].iloc[0]
            window = df.loc[df['Series'] == series_name, 'Win_Tree'].iloc[0]
            t_X, t_y = sliding_win_target(test_series, window, 1)

        if models[model_idx] == "XGB":
            #print("Entering XGB")
            singlecore_csv_out = os.path.join(out_dir, f"xgb_singlecore.csv")
            multicore_csv_out = os.path.join(out_dir, f"xgb_multicore.csv")
                                    
            sp_singlecore_csv_out = os.path.join(out_dir, f"sp_xgb_singlecore.csv")
            sp_multicore_csv_out = os.path.join(out_dir, f"sp_xgb_multicore.csv")

            general_path = os.path.join(in_dir, "xgb_models")
            if not os.path.exists(general_path):
                assert 1 == 0, "Invalid folder"
            model_out = os.path.join(general_path, f"{series_name}.joblib")
            model = joblib.load(model_out)            
            time_res = validate_timings(model, t_X)
            
            # # Test for single core
            # model.set_params(n_jobs = 1)
            # results, single_preds = validate_timings(model, t_X)
            # # Append to CSV Single core.
            # percentages = [f"{p}%" for p in range(10, 101, 10)]
            # # Append the total
            # total_singlecore_df = pd.DataFrame.from_dict({series_name:results}, orient="index", columns=percentages)
            # total_singlecore_df.to_csv(singlecore_csv_out, index=False, mode="a", header=not os.path.exists(singlecore_csv_out))
            # # Append single preds
            # sp_singlecore_df = pd.DataFrame.from_dict({series_name:single_preds}, orient="index", columns=percentages)
            # sp_singlecore_df.to_csv(sp_singlecore_csv_out, index=False, mode="a", header=not os.path.exists(sp_singlecore_csv_out))
            # # Redo for multicore
            # model.set_params(n_jobs = os.cpu_count())
            # results, single_preds = validate_timings(model, t_X)            
            # # Append the total
            # total_multicore_df = pd.DataFrame.from_dict({series_name:results}, orient="index", columns=percentages)
            # total_multicore_df.to_csv(multicore_csv_out, index=False, mode="a", header=not os.path.exists(multicore_csv_out))
            # # Append single preds
            # sp_multicore_df = pd.DataFrame.from_dict({series_name:single_preds}, orient="index", columns=percentages)
            # sp_multicore_df.to_csv(sp_multicore_csv_out, index=False, mode="a", header=not os.path.exists(sp_multicore_csv_out))
            

        elif models[model_idx] == "RF":
    
            singlecore_csv_out = os.path.join(out_dir, f"rf_singlecore.csv")
            multicore_csv_out = os.path.join(out_dir, f"rf_multicore.csv")                   
            sp_singlecore_csv_out = os.path.join(out_dir, f"sp_rf_singlecore.csv")
            sp_multicore_csv_out = os.path.join(out_dir, f"sp_rf_multicore.csv")

            general_path = os.path.join(in_dir, "rf_models")
            if not os.path.exists(general_path):
                assert 1 == 0, "Invalid folder"
            model_out = os.path.join(general_path, f"{series_name}.joblib")
            model = joblib.load(model_out)
            time_res = validate_timings(model, t_X)
            print(f" RF {time_res}")
            # # Test for single core
            # model.set_params(n_jobs = 1)
            # results, single_preds = validate_timings(model, t_X)
            # # Append to CSV Single core.
            # percentages = [f"{p}%" for p in range(10, 101, 10)]
            # # Append the total
            # total_singlecore_df = pd.DataFrame.from_dict({series_name:results}, orient="index", columns=percentages)
            # total_singlecore_df.to_csv(singlecore_csv_out, index=False, mode="a", header=not os.path.exists(singlecore_csv_out))
            # # Append single preds
            # sp_singlecore_df = pd.DataFrame.from_dict({series_name:single_preds}, orient="index", columns=percentages)
            # sp_singlecore_df.to_csv(sp_singlecore_csv_out, index=False, mode="a", header=not os.path.exists(sp_singlecore_csv_out))
            # # Redo for multicore
            # model.set_params(n_jobs = os.cpu_count())
            # results, single_preds = validate_timings(model, t_X)            
            # # Append the total
            # total_multicore_df = pd.DataFrame.from_dict({series_name:results}, orient="index", columns=percentages)
            # total_multicore_df.to_csv(multicore_csv_out, index=False, mode="a", header=not os.path.exists(multicore_csv_out))
            # # Append single preds
            # sp_multicore_df = pd.DataFrame.from_dict({series_name:single_preds}, orient="index", columns=percentages)
            # sp_multicore_df.to_csv(sp_multicore_csv_out, index=False, mode="a", header=not os.path.exists(sp_multicore_csv_out))
            
        
        elif models[model_idx] == "RT":
            singlecore_csv_out = os.path.join(out_dir, f"rt_singlecore.csv")
            sp_singlecore_csv_out = os.path.join(out_dir, f"sp_rt_singlecore.csv")
            general_path = os.path.join(in_dir, "rt_models")
            if not os.path.exists(general_path):
                assert 1 == 0, "Invalid folder"
            model_out = os.path.join(general_path, f"{series_name}.joblib")
            model = joblib.load(model_out)
            time_res = validate_timings(model, t_X)
            print(f" RT {time_res}")
            # results, single_preds = validate_timings(model, t_X)
            # # Append to CSV Single core.
            # percentages = [f"{p}%" for p in range(10, 101, 10)]
            # # Append the total
            # total_singlecore_df = pd.DataFrame.from_dict({series_name:results}, orient="index", columns=percentages)
            # total_singlecore_df.to_csv(singlecore_csv_out, index=False, mode="a", header=not os.path.exists(singlecore_csv_out))
            # # Append single preds
            # sp_singlecore_df = pd.DataFrame.from_dict({series_name:single_preds}, orient="index", columns=percentages)
            # sp_singlecore_df.to_csv(sp_singlecore_csv_out, index=False, mode="a", header=not os.path.exists(sp_singlecore_csv_out))
            

        elif models[model_idx] == "Dual-Stage":
            # singlecore_csv_out = os.path.join(out_dir, f"dual_stage_singlecore.csv")
            # multicore_csv_out = os.path.join(out_dir, f"dual_stage_multicore.csv")                   
            # sp_singlecore_csv_out = os.path.join(out_dir, f"sp_dual_stage_singlecore.csv")
            # sp_multicore_csv_out = os.path.join(out_dir, f"sp_dual_stage_multicore.csv")

            general_path = os.path.join(in_dir, "dual_stage")
            if not os.path.exists(general_path):
                assert 1 == 0, "Invalid folder"
            model_out = os.path.join(general_path, f"{series_name}.joblib")
            model = joblib.load(model_out)
            time_res = validate_timings_te_tfi(model, t_X)
            print(f" DS {time_res}")
            
            # # Re-enable pool
            # model.pool = Pool(processes= os.cpu_count())
            # # Test for single core
            # model.n_jobs = 1
            # results, single_preds = validate_timings_te_tfi(model, t_X)
            # # Append to CSV Single core.
            # percentages = [f"{p}%" for p in range(10, 101, 10)]
            # # Append the total
            # total_singlecore_df = pd.DataFrame.from_dict({series_name:results}, orient="index", columns=percentages)
            # total_singlecore_df.to_csv(singlecore_csv_out, index=False, mode="a", header=not os.path.exists(singlecore_csv_out))
            # # Append single preds
            # sp_singlecore_df = pd.DataFrame.from_dict({series_name:single_preds}, orient="index", columns=percentages)
            # sp_singlecore_df.to_csv(sp_singlecore_csv_out, index=False, mode="a", header=not os.path.exists(sp_singlecore_csv_out))
            # # Redo for multicore
            # model.n_jobs = os.cpu_count()
            # results, single_preds = validate_timings_te_tfi(model, t_X)            
            # # Append the total
            # total_multicore_df = pd.DataFrame.from_dict({series_name:results}, orient="index", columns=percentages)
            # total_multicore_df.to_csv(multicore_csv_out, index=False, mode="a", header=not os.path.exists(multicore_csv_out))
            # # Append single preds
            # sp_multicore_df = pd.DataFrame.from_dict({series_name:single_preds}, orient="index", columns=percentages)
            # sp_multicore_df.to_csv(sp_multicore_csv_out, index=False, mode="a", header=not os.path.exists(sp_multicore_csv_out))
        dict_results = {"Series" : series_name, "Model": models[model_idx], "AvgTime": time_res}
        df = pd.DataFrame(dict_results)
        timing_path = os.join(out_dir, "timings.csv")
        df.to_csv(timing_path, mode = "a", index= False, header = not os.path.exists(timing_path))
names_out = os.path.join(out_dir, "series_names.json5")
files_out = os.path.join(out_dir, "series_paths.json5")

