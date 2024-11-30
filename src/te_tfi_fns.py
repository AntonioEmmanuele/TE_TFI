from lib.ts_manip import sliding_win_cluster_aware, sliding_win
from src.te_tfi_model import TE_TFI
import numpy as np
from multiprocessing import Pool
import os
import itertools
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import numpy as np
import time 
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score

from tqdm import tqdm

def list_partitioning(a_list, num_of_partitions):
    return [list(item) for item in np.array_split(a_list, num_of_partitions)]

"""
Generates the list of each possible configuration of hyperparameters.
hyperparameters ->  Dictionary containing the set of hyperparameters to explore.
"""
def generate_hyperparameter_grid(hyperparameters):
    # Get hyp names and values
    keys = list(hyperparameters.keys())
    values = list(hyperparameters.values())
    # Generate all possible combinations
    combinations = itertools.product(*values)
    # Convert in a list of dictionaries
    configs = [dict(zip(keys, combination)) for combination in combinations]
    return configs

def validate_series(configurations, x_labels, y_labels, cv_order, starting_percentage):
    starting_rolling = int(starting_percentage * len(x_labels))
    rolling_offset   = int((len(x_labels) - starting_rolling) / cv_order)
    mse = []
    mape = []
    mae = []
    # For each configuration
    for conf in configurations:
        mse_per_conf = []     # Vector containing the metrics for each cv.
        mape_per_conf= []
        mae_per_conf = []
        for i in range(cv_order):
            # Split samples based on the rolling window approach, remember that these samples
            # are the win_trees which are the outputs of the cluster.
            train_x = x_labels[ : starting_rolling + i*rolling_offset]
            train_y = y_labels[ : starting_rolling + i*rolling_offset]
            val_x   = x_labels[starting_rolling + i*rolling_offset : starting_rolling + (i + 1)*rolling_offset]
            val_y   = y_labels[starting_rolling + i*rolling_offset : starting_rolling + (i + 1)*rolling_offset]
            model = DecisionTreeRegressor(**conf, random_state = 42)
            model.fit(train_x, train_y)
            outcomes = model.predict(val_x)
            mse_per_conf.append(mean_squared_error(y_true = val_y, y_pred = outcomes))
            mae_per_conf.append(mean_absolute_percentage_error(y_true = val_y, y_pred = outcomes))
            mape_per_conf.append(mean_absolute_error(y_true = val_y, y_pred = outcomes))
        mse.append(np.mean(mse_per_conf))
        mape.append(np.mean(mape_per_conf))
        mae.append(np.mean(mae_per_conf))
    return np.array(mse), np.array(mape), np.array(mae)


def hyp_trees(  cluster_type,           # Type of the cluster
                cluster_cfg,            # Configuration of the cluster
                num_clusters,           # Number of clusters
                tree_params,            # Cross validation parameters for trees
                time_series,            # Time series in consideration
                cv_order = 5,           # Cross validation order for each single tree
                tree_cv_perc_start = 0.5,   # Starting percentage for tree rolling window
                clust_perc = 0.4,           # Percentage of data used for clustering
                win_size_cluster = 100,     # Window size of the cluster TS
                win_size_tree = 50,         # Window size of trees TS.
                n_jobs = os.cpu_count(),    # Number of parallel workers
                disable_tqdm = False        # Disable the TQDM
            ):
    # Split the cluster and trees set.
    cluster_ts_size = int(clust_perc * len(time_series))
    if clust_perc == -1:
        cluster_ts_size = 0
        cluster_series  = time_series
    else:
        cluster_ts_size = int(clust_perc * len(time_series))
        cluster_series  = time_series
    trees_train_series = time_series[cluster_ts_size : ]    # Spit the time series and also the clusters
    # if len(timestamps) > 0:                                     # Supports also the absence of timestamps.
    #     trees_train_series = time_series[cluster_ts_size : ]    # Spit the time series and also the clusters

    # Get the cluster training sliding windows
    cluster_train_wins = sliding_win(cluster_series, window_size = win_size_cluster)
    # Get the trees training windows
    win_cluster, win_pred, target =  sliding_win_cluster_aware(
                                                        series = trees_train_series,
                                                        window_size_cluster = win_size_cluster, 
                                                        window_size_pred = win_size_tree, 
                                                        win_out_pred = 1                    
                                                        )

    # Train the cluster.
    cluster = TE_TFI.supported_clusters[cluster_type](**cluster_cfg, n_clusters = num_clusters, random_state = 42)
    cluster.fit(cluster_train_wins)
    # Get the labels for trees samples and their respective values.
    labels = cluster.predict(win_cluster)
    # Evaluate the sil score
    sil_score = silhouette_score(win_cluster, labels=labels)
    # Identify the labels for each sample
    trees_X = [ win_pred[labels == j] for j in range(0, num_clusters)] 
    trees_y = [ target[labels == j] for j in range(0, num_clusters)]
    # Generate the list of hyperparameters to try
    list_params = generate_hyperparameter_grid(tree_params)
    params_per_cpu = list_partitioning(list_params, n_jobs)
    pool = Pool(n_jobs)
    cfg_per_tree = []
    min_mse_per_tree = []
    min_mape_per_tree = []
    min_mae_per_tree = []

    id_results = 0 # MSE MAPE or MAE for CV
    # For each tree
    for tree_idx in tqdm(range(0, num_clusters), desc = "Evaluating configurations", disable = disable_tqdm):
        # Get the elements for each time window
        t_X = trees_X[tree_idx]
        t_y = trees_y[tree_idx]
        # Compute execute the cross-validation with pool and select the best series.            
        args = [[cpu_param, t_X, t_y, cv_order, tree_cv_perc_start] for cpu_param in params_per_cpu]
        #mse, mape, mae = pool.starmap(validate_series, args)
        results = pool.starmap(validate_series, args)
        results[0] = np.concatenate(results[0])
        results[1] = np.concatenate(results[1])
        results[2] = np.concatenate(results[2])
        best_result_idx = np.argmin(results[id_results])
        # Append to the results.
        cfg_per_tree.append(list_params[best_result_idx])
        min_mse_per_tree.append(results[0][best_result_idx])
        min_mape_per_tree.append(results[1][best_result_idx])
        min_mae_per_tree.append(results[2][best_result_idx])

    pool.close()
    pool.join()
    return cfg_per_tree, min_mse_per_tree, min_mape_per_tree, min_mae_per_tree, sil_score


# Funzione che effettua la cross-validazione di una configurazione di cluster 
# cercando la miglior configurazione per albero.
def cross_validate_cluster_with_tree_hyp():
    return 0

# Funzione che effettua la cross-validazione di un cluster cercando la miglior configurazione 
# per ogni albero.
def cross_validate_cluster():
    return 0

if __name__ == "__main__":
    # ls = np.arange(10)
    # print(f"Data {ls}")
    # part = list_partitioning(ls,3)
    # print(part)
    # def process_data(arr, factor):
    #     return np.array([pow(a,factor) for a in arr] )
    # pool = Pool(3)
    # pool_ret = pool.starmap(process_data, [(p,2) for p in part])
    # print(pool_ret)
    # res = np.concatenate(pool_ret)
    # print(f"Res {res}")
    # pool.close()
    # pool.join()

    #     # Funzione di esempio chiamata in starmap
    # def example_function(x):
    #     # Calcola un array di metriche (dimensione variabile)
    #     metrics = [i * x for i in range(1, x + 1)]  # Cambia la logica come serve
    #     return np.array(metrics)  # Restituisce un vettore

    # # Numero di processi
    # n_processes = 4

    # # Lista di input per la starmap
    # inputs = [2, 3, 4, 5]  # Input di esempio

    # # Crea il pool e utilizza starmap
    # with Pool(n_processes) as pool:
    #     results = pool.starmap(example_function,[[i] for i in inputs])  # Ottiene una lista di array
    # print(results[0].shape)
    # # Concatenazione dei risultati in un unico grande vettore
    # concatenated_vector = np.concatenate(results)

    # print("Vettori originali:")
    # for r in results:
    #     print(r)

    # print("\nVettore concatenato:")
    # print(concatenated_vector)
    # exit(1)
    path = "./datasets/processed/151_UCR_Anomaly_MesoplodonDensirostris_10000_19280_19440.txt" # For relative paths
    train_perc = 0.7
    series = np.loadtxt(path)
    train_size = int(len(series) * train_perc)
    # Rimuovi componenti anomale.
    train_series = series[0:train_size]
    test_series = series[train_size:]

    param_grid = {
        'max_depth': [5, 7, 10, 15, 20, 22, 25],      
        'min_samples_split': [2, 5, 10, 20, 50],            # Min samples to split a node
        'min_samples_leaf': [1, 5, 10, 20, 50],              # Min samples at a leaf node
        'max_features': [ 1.0, 'sqrt', 'log2']    # Max features considered for splitting
        }
    tm = time.perf_counter()
    window_size_cluster = 500
    window_size_tree = 50
    trees_cfg, trees_best_mse = hyp_trees(
                                    cluster_type = "KMeans",
                                    cluster_cfg = {},
                                    num_clusters = 5,
                                    tree_params = param_grid,
                                    time_series = train_series,
                                    cv_order = 5,
                                    tree_cv_perc_start = 0.0,
                                    clust_perc = -1,
                                    n_jobs = os.cpu_count(),
                                    disable_tqdm = False,
                                    win_size_cluster = window_size_cluster,
                                    win_size_tree = window_size_tree
                                )
    tm = time.perf_counter() - tm
    print(f"Esecuzione terminata {tm}")
    print(f"Configurazioni ")
    print(trees_cfg)
    print("Miglior risultato ")
    print(trees_best_mse)
    print("Adesso alleniamo il modello definitivo")
    model = TE_TFI(n_clusters = 5, tree_confs = trees_cfg, n_jobs = os.cpu_count(), random_state=42)

    train_wins_cluster, train_wins_tree, train_target_tree = sliding_win_cluster_aware(series = train_series, window_size_cluster = window_size_cluster, window_size_pred = window_size_tree, win_out_pred = 1)
    test_wins_cluster, test_wins_tree, test_target_tree = sliding_win_cluster_aware(series = test_series, window_size_cluster = window_size_cluster, window_size_pred = window_size_tree, win_out_pred = 1)
    model.fit_clust_ts(hyst_buffers_cl = train_wins_cluster, train_wins_tree = train_wins_tree, train_target_tree = train_target_tree)
    class_inferences = model.predict_clust_ts(hyst_buff_cl = test_wins_cluster, wins_tree = test_wins_tree)
    mape_te = mean_absolute_percentage_error(y_true = test_target_tree, y_pred = class_inferences)
    mse_te = mean_squared_error(y_true = test_target_tree, y_pred = class_inferences)
    mae_te = mean_absolute_error(y_true = test_target_tree, y_pred = class_inferences)

    print(f"TE TFI CLASS:  MSE: {mse_te} MAPE {mape_te} MAE {mae_te}")
    print("Comparing with a default model")
    # Compare with the default cfg:
    model = TE_TFI(n_clusters = 5,  tree_confs = [ { "max_depth" : 20}] * 5, n_jobs = -1, random_state = 42)
    model.fit_clust_ts(hyst_buffers_cl = train_wins_cluster, train_wins_tree = train_wins_tree, train_target_tree = train_target_tree)
    class_inferences = model.predict_clust_ts(hyst_buff_cl = test_wins_cluster, wins_tree = test_wins_tree)
    
    mape_te = mean_absolute_percentage_error(y_true = test_target_tree, y_pred = class_inferences)
    mse_te = mean_squared_error(y_true = test_target_tree, y_pred = class_inferences)
    mae_te = mean_absolute_error(y_true = test_target_tree, y_pred = class_inferences)
    print(f"TE TFI CLASS:  MSE: {mse_te} MAPE {mape_te} MAE {mae_te}")