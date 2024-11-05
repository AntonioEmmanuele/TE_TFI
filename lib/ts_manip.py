import numpy as np

def sliding_win(series, window_size):
    subsequences = []
    for i in range(len(series) - window_size + 1):
        subsequences.append(series[i:i + window_size])
    return np.array(subsequences)

def sliding_win_target(series, window_size, win_out):
    X, y = [], []
    for i in range(len(series) - window_size - win_out + 1):
        win = series[i:i + window_size]
        target = series[i + window_size : i + window_size + win_out]
        X.append(win)
        y.append(target)
    return np.array(X), np.array(y)

def sliding_win_cluster_aware(series, window_size_cluster, window_size_pred, win_out_pred):
    assert window_size_cluster > window_size_pred
    X_cluster, X_pred, y = [], [], []
    for i in range(len(series) - window_size_cluster - win_out_pred + 1):
        cluster = series[i:i + window_size_cluster]      
        pred_window = cluster[-window_size_pred:]
        target = series[i + window_size_cluster:i + window_size_cluster + win_out_pred]
        X_cluster.append(cluster)
        X_pred.append(pred_window)
        y.append(target)
    return np.array(X_cluster), np.array(X_pred), np.array(y)
if __name__ == "__main__":
    seq = np.array([i for i in range(1,16)])
    win_cluster, win_pred, target = sliding_win_cluster_aware(seq, 5, 3, 1)
    print(win_cluster)
    print(win_pred)
    print(target)