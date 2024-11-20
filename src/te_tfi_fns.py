from ..lib.ts_manip import sliding_win_clust_aw_target_timestamps, sliding_win
from TE_TFI import TE_TFI
import numpy as np
from multiprocessing import Pool
import os
def hyp_trees(  model : TE_TFI,         # Model in consideration. 
                tree_params,            # Cross validation parameters for trees
                time_series,            # Time series in consideration
                timestamps,             # Timestamps of the time series
                cv_order = 5,           # Cross validation order for each single tree
                clust_perc = 0.4,       # Percentage of data used for clustering
                win_size_cluster = 100, # Window size of the cluster TS
                win_size_tree = 10,     # Window size of trees TS.
                n_jobs = os.cpu_count() # Number of parallel workers
            ):
    assert len(time_series) == len(timestamps), "Please provide a time series and timestamps of equal size"
    # Split the cluster and trees set.
    cluster_ts_size = clust_perc * len(time_series)
    cluster_series  = time_series[0 : cluster_ts_size]
    trees_train_series = time_series[cluster_ts_size : ]    # Spit the time series and also the clusters
    trees_train_timestamps = timestamps[cluster_ts_size :]
    # Get the cluster training sliding windows
    cluster_train_wins = sliding_win(cluster_series, window_size = win_size_cluster)
    # Get the trees training windows
    win_cluster, _, win_pred, win_pred_ts, target, _ = sliding_win_clust_aw_target_timestamps(
                                                                                series = trees_train_series,
                                                                                time_series_timestamps = trees_train_timestamps, 
                                                                                window_size_cluster = win_size_cluster, 
                                                                                window_size_pred = win_size_tree, 
                                                                                win_out_pred = 1                    
                                                                            )
    # Train the cluster.
    model.cluster.fit(cluster_train_wins)
    # Get the labels for trees samples and their respective values.
    labels = model.cluster.predict(win_cluster)
    # Identify the labels for each sample.
    trees_args = [ (win_pred[labels == j], win_pred_ts[labels == j], target[labels == j]) for j in range(0, model.n_clusters)]
     
    # For each tree
    for tree_idx in range(0, model.n_clusters):
        # Order the elements for each tree to implement the rolling window.
        trees_X = trees_args[tree_idx][0]
        trees_y = trees_args[tree_idx][2]
        starting_timestamps = [ts[0] for ts in trees_args[tree_idx][1]] # Starting timestamps of each time window
        ordered_ts_idx      = np.argsort(starting_timestamps)
        trees_X = trees_X[ordered_ts_idx]
        trees_y = trees_X[ordered_ts_idx]
        # Analize with the rolling window the 5 models.
        pool_tree = Pool(n_jobs) # Todo: Add remaining trees idxes
        # Select the best model for each tree.

        # Train the best tree to get its accuracy (optional).
    
    # Compute the accuracy of the final tree.

    #

# Funzione che effettua la cross-validazione di una configurazione di cluster 
# cercando la miglior configurazione per albero.
def cross_validate_cluster_with_tree_hyp():
    return 0

# Funzione che effettua la cross-validazione di un cluster cercando la miglior configurazione 
# per ogni albero.
def cross_validate_cluster():
    return 0