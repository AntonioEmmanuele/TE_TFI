import argparse
import pandas as pd
import numpy as np
from src.te_tfi_fns import hyp_trees
import os
from tslearn.clustering import TimeSeriesKMeans
from src.te_tfi_model import TE_TFI
from lib.ts_manip import sliding_win_cluster_aware, sliding_win_cluster_aware_multivariate
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
#from lib.ucr_parser import get_series_name
import joblib
import json5
from sklearn.preprocessing import MinMaxScaler
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Script di esempio per accettare argomenti.")
    
    parser.add_argument('--num_cluster', type=int, required=True )
    parser.add_argument('--path_stagionality', type=str, required=False)
    parser.add_argument('--series_path', type=str, required=True)
    parser.add_argument('--lag_percentage', type = float, required = False, default=1.0)
    parser.add_argument('--cv_order', type = int, required = False, default= 5)
    parser.add_argument('--win_clust', type=int, required=False, default=50)
    parser.add_argument('--win_tree_perc', type=float, required=False, default=1.0)
    parser.add_argument('--n_jobs', type=int, required=False, default=os.cpu_count())
    parser.add_argument('--out_path', type=str, required=False, default="./")
    parser.add_argument('--is_multivariate', type=int, required=False, default=0)
    parser.add_argument('--target_column', type=str, required=False, default=None)

    
    args = parser.parse_args()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    #assert not (args.is_multivariate != None and args.target_column == None)
    if not args.is_multivariate:
        series = np.loadtxt(args.series_path, dtype=float)
        file = args.series_path.split("/")[-1]
        series_name = file[:3]
    else:
        series_df = pd.read_csv(args.series_path)
        # Remove useless series
        if "date" in series_df.columns:
            series_df.drop(columns = ["date"], inplace = True)
        if "DateTime" in series_df.columns:
            series_df.drop(columns = ["DateTime"], inplace = True)
        scaler = MinMaxScaler()
        # Convert the series. 
        series = pd.DataFrame(scaler.fit_transform(series_df), columns=series_df.columns)
        file = args.series_path.split("/")[-1]
        series_name = file[:-4]
    stag_csv = pd.read_csv(args.path_stagionality)

    has_seasonality = stag_csv.loc[stag_csv['file'] == file, 'has_seasonality'].values[0]
    if has_seasonality:
        biggest_lag = stag_csv.loc[stag_csv['file'] == file, 'biggest_lag'].values[0]
        win_clust = int(biggest_lag * args.lag_percentage)
    else:
        win_clust = args.win_clust

    if args.is_multivariate:
        if args.target_column is None:
            target_column = stag_csv.loc[stag_csv['file'] == file, 'feature'].values[0]
        else:
            target_column = args.target_column

    win_tree = int(win_clust*args.win_tree_perc)
    
    param_grid = {
        'max_depth': [5, 7, 10, 15, 20],      
        'min_samples_split': [2, 5, 10, 20, 50],  # Min samples to split a node
        'min_samples_leaf': [1, 5, 10, 20, 50],   # Min samples at a leaf node
        'max_features': [ 1.0, 'sqrt', 'log2']    # Max features considered for splitting
    }


    trees_cfg, trees_best_mse, trees_best_mape, trees_best_mae, sil_score = hyp_trees(
                                cluster_type = "KMeans",
                                cluster_cfg = { "max_iter" : 500, "verbose": True},
                                num_clusters = args.num_cluster,
                                tree_params = param_grid,
                                time_series = series,
                                cv_order = 5,
                                tree_cv_perc_start = 0.5,
                                clust_perc = -1,
                                n_jobs = args.n_jobs,
                                disable_tqdm = False,
                                win_size_cluster = win_clust,
                                win_size_tree = win_tree,
                                is_multivariate = args.is_multivariate,
                                target_column = target_column
                            )
    te_tfi = TE_TFI(cluster_type="KMeans", n_clusters = args.num_cluster, cluster_cfg = { "max_iter" : 500, "verbose": True}, tree_confs=trees_cfg, n_jobs=args.n_jobs)
    train_size = int(0.7 * len(series))
    if not args.is_multivariate:
        train_series = series[: train_size]
        test_series = series[train_size : ]
        train_wins_cluster, train_wins_tree, train_target_tree = sliding_win_cluster_aware(series = train_series, window_size_cluster = win_clust, window_size_pred = win_tree, win_out_pred = 1)
        test_wins_cluster, test_wins_tree, test_target_tree = sliding_win_cluster_aware(series = test_series, window_size_cluster = win_clust, window_size_pred = win_tree, win_out_pred = 1)
    else:
        train_series = series.iloc[: train_size]
        test_series = series.iloc[train_size : ]
        train_wins_cluster, train_wins_tree, train_target_tree = sliding_win_cluster_aware_multivariate(df_series = train_series, target_column = target_column, window_size_cluster = win_clust, window_size_pred = win_tree, win_out_pred = 1)
        test_wins_cluster, test_wins_tree, test_target_tree = sliding_win_cluster_aware_multivariate(df_series = test_series, target_column = target_column, window_size_cluster = win_clust, window_size_pred = win_tree, win_out_pred = 1)
    te_tfi.fit_clust_ts(train_wins_cluster, train_wins_tree, train_target_tree, False)
    preds, sil_fin = te_tfi.predict_clust_ts(test_wins_cluster, test_wins_tree)
    te_tfi_mse  = mean_squared_error(y_true = test_target_tree, y_pred = preds)
    te_tfi_mape = mean_absolute_percentage_error(y_true = test_target_tree, y_pred = preds)
    te_tfi_mae  = mean_absolute_error(y_true = test_target_tree, y_pred = preds)

    print(trees_cfg)
    print("MSE")
    print(trees_best_mse)
    print("MAPE")
    print(trees_best_mape)
    print("MAE")
    print(trees_best_mae)
    print("SIL SCORE")
    print(sil_score)
    print(f"MAE: {te_tfi_mae} MAPE {te_tfi_mape} MSE {te_tfi_mse}")
    print(f"Sil Final: {sil_fin}")

    dict_out = {
        "Series": series_name,
        "NumClusters":args.num_cluster,
        "Sil Hyp": sil_score,
        "Sil Test": sil_fin,
        "Final MSE": te_tfi_mse,
        "Final MAPE": te_tfi_mape,
        "Final MAE": te_tfi_mae,
        "Seasonal" : has_seasonality,
        "Win_Clust" : win_clust,
        "Win_Tree" : win_tree,
        "Perc"      :args.lag_percentage,
        "Trees MSE Hyp" : trees_best_mse,
        "Trees MAPE Hyp": trees_best_mape,
        "Trees MAE Hyp" : trees_best_mae
    }
    csv_df = pd.DataFrame(dict_out, index = [0])
    csv_out_stats = os.path.join(args.out_path, "stats_hyp.csv")
    add_header = not os.path.exists(csv_out_stats)
    csv_df.to_csv(csv_out_stats, sep = ",", header=add_header, index=False, mode = "a")
    # Dump the model
    te_tfi.pool.close()
    te_tfi.pool = None
    out_dir_model = os.path.join(args.out_path, f"models")
    if not os.path.exists(out_dir_model):
        os.makedirs(out_dir_model) 
    te_tfi.pool
    out_models = os.path.join(out_dir_model, f"{series_name}.joblib")
    joblib.dump(te_tfi, out_models)
    # Dump Hyperparametrization
    out_hyp_dir =  os.path.join(args.out_path, f"hyp_trees")    
    if not os.path.exists(out_hyp_dir):
        os.makedirs(out_hyp_dir)
    out_hyp_model = os.path.join(out_hyp_dir, f"{series_name}.json5")
    with open(out_hyp_model, "w") as f:
        json5.dump(trees_cfg, f, indent = 2)
    