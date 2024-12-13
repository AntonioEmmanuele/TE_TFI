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

models_to_train = 250
# Experiments
series_path = "./datasets/processed"
out_dir = "./experiments/ucr_no_preprocess_timings_complete"
hy_models = "hy_models"
stats_hyp = "stats_hyp.csv"
models = ["RT", "RF", "XGB", "Dual-Stage"]
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

# # Reset index if needed
# min_mse_df = min_mse_df.reset_index(drop=True)

series_list = set(models_stats[-1]["Series"].astype("str").to_list())
#samples_series = random.sample(list(series_list), models_to_train)
samples_series = list(series_list)
test_samples_sizes = []
files_to_save = []
# Create the series for the window
for series_name in tqdm(samples_series, desc = "Training models"):
    series_file = get_series(series_path, series_name)
    series_file_path = os.path.join(series_path, series_file)
    files_to_save.append(series_file_path)
    series = np.loadtxt(series_file_path)
    train_size = int(0.7 * len(series))
    train_series = series[0 : train_size]
    #test_series = series[train_size : ]

    # Train all the other models
    for model_idx in range(0, len(parameter_paths)):
    
        df = models_stats[model_idx]
        #test_samples_sizes.append(test_X)
        if models[model_idx] != "Dual-Stage":
            window = df.loc[df['Series'] == series_name, 'Win'].iloc[0]
            t_X, t_y = sliding_win_target(train_series, window, 1)
            #test_X, test_y = sliding_win_target(test_series, window, 1)
            path = os.path.join(os.path.join(parameter_paths[model_idx], hy_models), f"{series_name}.json5")
            with open(path, 'r') as f:
                parameters = json5.load(f)
        else:
            num_clusters = df.loc[df['Series'] == series_name, 'NumClusters'].iloc[0]
            lag_percentage = df.loc[df['Series'] == series_name, 'Perc'].iloc[0]
            window = df.loc[df['Series'] == series_name, 'Win_Tree'].iloc[0]
            t_X, t_y = sliding_win_target(train_series, window, 1)
            #test_X, test_y = sliding_win_target(test_series, window, 1)
            path = os.path.join(os.path.join(parameter_paths[model_idx], "hyp_trees"), f"{series_name}_{num_clusters}_{lag_percentage}.json5")
            with open(path, 'r') as f:
                parameters = json5.load(f)
        if models[model_idx] == "XGB":
            general_path = os.path.join(out_dir, "xgb_models")
            if not os.path.exists(general_path):
                os.makedirs(general_path)
            model = XGBRegressor(n_jobs = 1, n_estimators = 100, objective = 'reg:squarederror', random_state = 42, **parameters)
            model.fit(t_X, t_y)
            model_out = os.path.join(general_path, f"{series_name}.joblib")
            joblib.dump(model, model_out)
        elif models[model_idx] == "RF":
            general_path = os.path.join(out_dir, "rf_models")
            if not os.path.exists(general_path):
                os.makedirs(general_path)
            model = RandomForestRegressor(n_jobs = 1, n_estimators = 100, random_state = 42, **parameters)
            model.fit(t_X, t_y.ravel())
            model_out = os.path.join(general_path, f"{series_name}.joblib")
            joblib.dump(model, model_out)
        elif models[model_idx] == "RT":
            general_path = os.path.join(out_dir, "rt_models")
            if not os.path.exists(general_path):
                os.makedirs(general_path)
            model = DecisionTreeRegressor(random_state=42, ** parameters)
            model.fit(t_X, t_y)
            model_out = os.path.join(general_path, f"{series_name}.joblib")
            joblib.dump(model, model_out)
        elif models[model_idx] == "Dual-Stage":
            general_path = os.path.join(out_dir, "dual_stage")
            if not os.path.exists(general_path):
                os.makedirs(general_path)
            te_tfi = TE_TFI(cluster_type="KMeans", n_clusters = num_clusters, cluster_cfg = { "max_iter" : 500, "verbose": False}, tree_confs = parameters, n_jobs= 1)
            te_tfi.fit_clust_ts(t_X, t_X, t_y, False)
            te_tfi.pool = None
            model_out = os.path.join(general_path, f"{series_name}.joblib")
            joblib.dump(te_tfi, model_out)

names_out = os.path.join(out_dir, "series_names.json5")
files_out = os.path.join(out_dir, "series_paths.json5")

with open(names_out, 'w') as f:
   json5.dump(samples_series, f, indent=2)

with open(files_out, 'w') as f:
   json5.dump(files_to_save, f, indent=2)