import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from ..lib.ts_manip import sliding_win_cluster_aware

if __name__ == "__main__":
    # Path della serie temporale
    path = "005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt"
    # Rimuovi componenti anomale.
    series = np.loadtxt(path)[0:4000]
    train_series = series[0:2000]
    test_series = series[2000:]

    # plotta la Serie temporale.
    plt.plot(series)
    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.show()
    window_size_cluster = 100
    window_size_tree = 50
    win_pred = 1   
    n_clusters = 5 # Dimezzati perchè abbiamo train e test    
    train_wins_cluster, train_wins_tree, train_target_tree = sliding_win_cluster_aware(series = train_series, window_size_cluster = window_size_cluster, window_size_pred = window_size_tree, win_out_pred = win_pred)
    test_wins_cluster, test_wins_tree, test_target_tree = sliding_win_cluster_aware(series = train_series, window_size_cluster = window_size_cluster, window_size_pred = window_size_tree, win_out_pred = win_pred)

    kmeans = KMeans(n_clusters = n_clusters, random_state = 0)
    labels = kmeans.fit_predict(train_wins_cluster)
    for i in range(n_clusters):    
        cluster = train_wins_cluster[labels == i]
        plt.figure(figsize=(10, 6))
        for seq in cluster:
            plt.plot(seq, alpha=0.4)
        plt.title(f'Cluster {i+1}')
        plt.xlabel('Sample')
        plt.ylabel('Values')
        plt.show()
