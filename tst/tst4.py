import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from lib.ts_manip import sliding_win_cluster_aware
from tqdm import tqdm
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    # Path della serie temporale
    #path = "./datasets/processed/087_UCR_Anomaly_DISTORTEDsel840mECG1_17000_51370_51740.txt" # For relative paths
    path = "./tst/005_UCR_Anomaly_DISTORTEDCIMIS44AirTemperature1_4000_5391_5392.txt" # For relative paths
    # Rimuovi componenti anomale.
    series = np.loadtxt(path)
    train_size =  int(0.7 * len(series))
    train_series = series[:train_size]
    test_series = series[train_size:]
    plot = False
    if plot: 
        # plotta la Serie temporale.
        plt.plot(series)
        plt.xlabel("Samples")
        plt.ylabel("Values")
        plt.show()
        #exit(1)
    window_size_cluster = 2000
    window_size_tree = 300
    win_pred = 1   
    n_clusters = 10 # Dimezzati perchè abbiamo train e test    
    train_wins_cluster, train_wins_tree, train_target_tree = sliding_win_cluster_aware(series = train_series, window_size_cluster = window_size_cluster, window_size_pred = window_size_tree, win_out_pred = win_pred)
    test_wins_cluster, test_wins_tree, test_target_tree = sliding_win_cluster_aware(series = test_series, window_size_cluster = window_size_cluster, window_size_pred = window_size_tree, win_out_pred = win_pred)

    print(f"La dimensione del cluster training è {len(train_wins_cluster)}")
    print(f"La dimensione del tree training è {len(train_target_tree)}")
    

    print(f"La dimensione del cluster testing è {len(train_wins_cluster)}")
    print(f"La dimensione del tree testing è {len(train_target_tree)}")
    # exit(1)
    kmeans = KMeans(n_clusters = n_clusters, random_state = 0)
    labels = kmeans.fit_predict(train_wins_cluster)
    trees = []
    # test_cluster_labels = kmeans.predict(test_wins_cluster)
    # print(test_cluster_labels)
    # exit(1)
    for i in range(n_clusters):    
        # Get the time series in different clusters.
        cluster_ts = train_wins_cluster[labels == i]
        tree_ts = train_wins_tree[labels == i]
        tree_ts_target = train_target_tree[labels == i]
        if plot :
            plt.figure(figsize=(10, 6))
            # Plot cluster wins
            for seq in cluster_ts:
                plt.plot(seq, alpha=0.4)
            plt.title(f'Cluster {i+1}')
            plt.xlabel('Sample')
            plt.ylabel('Values')
            plt.show()
            # Plot tree series wins
            for seq in tree_ts:
                plt.plot(seq, alpha=0.4)
            plt.title(f'Cluster Tree {i+1}')
            plt.xlabel('Sample')
            plt.ylabel('Values')
            plt.show()
            
        # Train a tree
        print(f"Fitting tree for cluster {i + 1}")
        tree = DecisionTreeRegressor(random_state = 42, max_depth = 20)
        tree.fit(tree_ts, tree_ts_target)
        # Append to cluster 
        trees.append(tree)
    print(f"Inference ")
    # Classify all the input samples
    test_cluster_labels = kmeans.predict(test_wins_cluster)
    # print(test_cluster_labels)
    # Inference
    inferences = []
    for idx, tree_label in tqdm(enumerate(test_cluster_labels)):
        # print("Sample inf:")
        # print(type(test_wins_tree[idx]))
        # print(test_wins_tree[idx])
        # exit(1)
        #samples = trees[tree_label].predict([test_wins_tree[idx]])
        # print("Inferred sample")
        # print(samples)
        # exit(1)
        # print(f"\n ")
        # print(tree_label)
        # print(test_wins_tree[idx])
        predicted_sample = trees[tree_label].predict([test_wins_tree[idx]]) 
        # print(f"Predicted sample {predicted_sample}")
        # exit(1)
        inferences.append(predicted_sample[0])
        
    mape_te = mean_absolute_percentage_error(y_true = test_target_tree, y_pred = inferences)
    mse_te = mean_squared_error(y_true = test_target_tree, y_pred = inferences)
    mae_te = mean_absolute_error(y_true = test_target_tree, y_pred = inferences)
    print(f"TE TFI MSE: {mse_te} MAPE {mape_te} MAE {mae_te}")
    
    t = DecisionTreeRegressor(random_state = 42, max_depth = 20)
    t.fit(X = train_wins_tree, y = train_target_tree)
    rt_preds = t.predict(test_wins_tree)

    rt_mape = mean_absolute_percentage_error(y_true = test_target_tree, y_pred = rt_preds)
    rt_mse = mean_squared_error(y_true = test_target_tree, y_pred = rt_preds)
    rt_mae = mean_absolute_error(y_true = test_target_tree, y_pred = rt_preds)
    print(f"RT MSE: {rt_mse} MAPE {rt_mape} MAE {rt_mae}")
    exit(1)
    rf = RandomForestRegressor(random_state = 42, n_estimators = 100, max_depth = 100)
    rf.fit(X = train_wins_tree, y = train_target_tree.ravel())
    rf_preds = rf.predict(test_wins_tree)

    rf_mape = mean_absolute_percentage_error(y_true = test_target_tree, y_pred = rf_preds)
    rf_mse = mean_squared_error(y_true = test_target_tree, y_pred = rf_preds)
    rf_mae = mean_absolute_error(y_true = test_target_tree, y_pred = rf_preds)

    print(f"RF MSE: {rf_mse} MAPE {rf_mape} MAE {rf_mae}")
    # if mape_te > mape:
    #     print("Migliore RT")
    # else : 
    #     print(f"TE TFI > RT")
    
    # Time series plots    
    plt.plot(inferences[0:100], label = "TE TFI")
    plt.plot(rt_preds[0:100], label = "RT")
    plt.plot(rf_preds[0:100], label = "RF")
    plt.plot(test_target_tree[0:100], label = "Original")

    plt.xlabel("Samples")
    plt.ylabel("Values")
    plt.legend(loc = "best")
    plt.show()
