import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from lib.ts_manip import sliding_win_cluster_aware, sliding_win, sliding_win_target
#from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from src.te_tfi_model import TE_TFI
from scipy.stats import anderson
from sklearn.datasets import make_blobs
from tslearn.clustering import KShape

def gmeans(X, max_iter=10, alpha=0.05):
    clusters = [X]
    centroids = []
    for _ in range(max_iter):
        new_clusters = []
        for cluster in clusters:
            # Applica K-Means con 2 cluster su ogni cluster corrente
            kmeans = KMeans(n_clusters=2, random_state=42)
            labels = kmeans.fit_predict(cluster)
            cluster_centers = kmeans.cluster_centers_
            #print(f"I centri di clustering individuati {cluster_centers}")
            # Test di normalità su ciascun cluster
            for i in range(2):
                data = cluster[labels == i]
                stat, crit, sig_level = anderson(data[:, 0])  # Test di Anderson-Darling

                if stat < crit[2]:  # Normalità accettata
                    centroids.append(cluster_centers[i])
                else:  # Normalità rifiutata, suddividi ulteriormente
                    new_clusters.append(data)
        clusters = new_clusters
        if not clusters:  # Se non ci sono nuovi cluster da dividere
            break
    return np.array(centroids)

def dividi_lista_in_quattro(lista):
    lunghezza = len(lista)
    base = lunghezza // 4
    resto = lunghezza % 4
    sottoliste = []
    inizio = 0

    for i in range(4):
        # Distribuisci l'eventuale resto tra le prime sottoliste
        fine = inizio + base + (1 if i < resto else 0)
        sottoliste.append(lista[inizio:fine])
        inizio = fine

    return sottoliste, np.arange(0,1)

if __name__ == "__main__":
    # Path della serie temporale
    path = "./datasets/processed/019_UCR_Anomaly_DISTORTEDGP711MarkerLFM5z1_5000_6168_6212.txt" # For relative paths
    
    train_meta_perc = 0.4
    train_tree_perc = 0.4
    series = np.loadtxt(path)
    meta_train_size = int(len(series) * train_meta_perc)
    tree_train_size = int(len(series) * (train_tree_perc))
    # Rimuovi componenti anomale.
    train_series = series[0:meta_train_size]
    tree_series = series[meta_train_size : meta_train_size + tree_train_size]
    test_series = series[meta_train_size + tree_train_size: ]
        
    plot = False
    if plot: 
        # plotta la Serie temporale.
        plt.plot(series)
        plt.xlabel("Samples")
        plt.ylabel("Values")
        plt.show()
        #exit(1)
    window_size_cluster = 50
    window_size_tree = 50
    win_pred = 1   
    n_clusters = 6 # Dimezzati perchè abbiamo train e test    
    # Train 
    train_tree_split = dividi_lista_in_quattro(train_series)
    x_samples = []
    y_labels = []
    for split_class, split in enumerate(train_tree_split):
        
        x_samples.extend(sliding_win(split, window_size_cluster))
        y_labels.append(split_class)
    print(x_samples)
    metaclassifier = DecisionTreeRegressor(max_depth=20)
    metaclassifier.fit(x_samples, y_labels)

    # Labels 
    test_tree, targets =  sliding_win_target(tree_series, window_size_cluster, 1)
    labels = metaclassifier.predict(tree_series)
    print(labels[0])
    # train_wins_cluster, train_wins_tree, train_target_tree = sliding_win_cluster_aware(series = train_series, window_size_cluster = window_size_cluster, window_size_pred = window_size_tree, win_out_pred = win_pred)
    # print(f"La dimensione del cluster training è {len(train_wins_cluster)}")
    # print(f"La dimensione del tree training è {len(train_target_tree)}")
    
    # test_wins_cluster, test_wins_tree, test_target_tree = sliding_win_cluster_aware(series = test_series, window_size_cluster = window_size_cluster, window_size_pred = window_size_tree, win_out_pred = win_pred)

    # print(f"La dimensione del cluster testing è {len(train_wins_cluster)}")
    # print(f"La dimensione del tree testing è {len(test_wins_cluster)}")
    
    te_tfi = TE_TFI(
        cluster_type="KShape",
        n_clusters = n_clusters,
        cluster_cfg={ "verbose" : True},
        random_state = 42,
        tree_confs = [ { 
            "max_depth" : 20
        }
        ] * n_clusters
    )
    # print(train_target_tree)
    # exit(1)
    te_tfi.fit_clust_ts(hyst_buffers_cl = train_wins_cluster, train_wins_tree = train_wins_tree, train_target_tree = train_target_tree)
    class_inferences = te_tfi.predict_clust_ts(hyst_buff_cl = test_wins_cluster, wins_tree = test_wins_tree)
    
    mape_te = mean_absolute_percentage_error(y_true = test_target_tree, y_pred = class_inferences)
    mse_te = mean_squared_error(y_true = test_target_tree, y_pred = class_inferences)
    mae_te = mean_absolute_error(y_true = test_target_tree, y_pred = class_inferences)
    print(f"TE TFI CLASS:  MSE: {mse_te} MAPE {mape_te} MAE {mae_te}")
    
    t = DecisionTreeRegressor(random_state = 42, max_depth = 20)
    t.fit(X = train_wins_tree, y = train_target_tree)
    rt_preds = t.predict(test_wins_tree)

    rt_mape = mean_absolute_percentage_error(y_true = test_target_tree, y_pred = rt_preds)
    rt_mse = mean_squared_error(y_true = test_target_tree, y_pred = rt_preds)
    rt_mae = mean_absolute_error(y_true = test_target_tree, y_pred = rt_preds)
    print(f"RT MSE: {rt_mse} MAPE {rt_mape} MAE {rt_mae}")
    
    rf = RandomForestRegressor(random_state = 42, n_estimators = 4, max_depth = 20, n_jobs=6)
    rf.fit(X = train_wins_tree, y = train_target_tree.ravel())
    rf_preds = rf.predict(test_wins_tree)

    rf_mape = mean_absolute_percentage_error(y_true = test_target_tree, y_pred = rf_preds)
    rf_mse = mean_squared_error(y_true = test_target_tree, y_pred = rf_preds)
    rf_mae = mean_absolute_error(y_true = test_target_tree, y_pred = rf_preds)

    print(f"RF MSE: {rf_mse} MAPE {rf_mape} MAE {rf_mae}")
    
    # Time series plots 
    plt.plot(class_inferences[0:100], label = "TE-TFI")   
    plt.plot(rt_preds[0:100], label = "RT")
    plt.plot(rf_preds[0:100], label = "RF")
    plt.plot(test_target_tree[0:100], label = "Original")

    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.legend(loc = "best")
    plt.show()
