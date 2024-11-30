import argparse
import pandas as pd
import numpy as np
from src.te_tfi_fns import hyp_trees
import os
if __name__ == "__main__":
    # Creazione del parser
    parser = argparse.ArgumentParser(description="Script di esempio per accettare argomenti.")
    
    # Aggiunta degli argomenti
    parser.add_argument('--num_cluster', type=str, required=True )
    parser.add_argument('--win_clust', type=int, required=True)
    parser.add_argument('--path_stagionality', type=str, required=False)
    parser.add_argument('--series_path', type=str, required=True)
    parser.add_argument('--lag_percentage', type = float, required = False)
    args = parser.parse_args()
    series = np.loadtxt(args.series_path, dtype=float)

    series_name = args.series_path.split("/")[-1]
    stag_csv = pd.read_csv(args.path_stagionality)
    if args.lag_percentage is None:
        lag_percentage = 1.0
    else:
        lag_percentage = args.lag_percentage
    
    has_seasonality = stag_csv.loc[stag_csv['file'] == series_name, 'has_seasonality'].values[0]
    if has_seasonality:
        biggest_lag = stag_csv.loc[stag_csv['file'] == series_name, 'biggest_lag'].values[0]
        win_clust = int(biggest_lag * args.lag_percentage)
    else:
        win_clust = args.win_clust

    
    param_grid = {
        'max_depth': [5, 7, 10, 15, 20],      
        'min_samples_split': [2, 5, 10, 20, 50],            # Min samples to split a node
        'min_samples_leaf': [1, 5, 10, 20, 50],              # Min samples at a leaf node
        'max_features': [ 1.0, 'sqrt', 'log2']    # Max features considered for splitting
    }

    trees_cfg, trees_best_mse = hyp_trees(
                                cluster_type = "KMeans",
                                cluster_cfg = {},
                                num_clusters = 5,
                                tree_params = param_grid,
                                time_series = series,
                                cv_order = 10,
                                tree_cv_perc_start = 0.0,
                                clust_perc = -1,
                                n_jobs = os.cpu_count(),
                                disable_tqdm = False,
                                win_size_cluster = win_clust,
                                win_size_tree = win_clust
                            )
    