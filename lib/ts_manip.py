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
    assert window_size_cluster >= window_size_pred
    X_cluster, X_pred, y = [], [], []
    for i in range(len(series) - window_size_cluster - win_out_pred + 1):
        cluster = series[i:i + window_size_cluster]      
        pred_window = cluster[-window_size_pred:]
        target = series[i + window_size_cluster:i + window_size_cluster + win_out_pred]
        X_cluster.append(cluster)
        X_pred.append(pred_window)
        y.append(target)
    return np.array(X_cluster), np.array(X_pred), np.array(y)

""" Same of the other, just partition also the timestamps """
def sliding_win_timestamps(series, timestamps, window_size):
    assert len(series) == len(timestamps)
    subsequences = []
    subs_ts = []
    for i in range(len(series) - window_size + 1):
        subsequences.append(series[i:i + window_size])
        subs_ts.append(timestamps[i : i + window_size])
    return np.array(subsequences), np.array(subs_ts)

def sliding_win_target_timestamps(series, time_series_timestamps, window_size, win_out):
    assert len(series) == len(time_series_timestamps)
    X, y = [], []
    X_tim, y_tim = [], []
    for i in range(len(series) - window_size - win_out + 1):
        win = series[i:i + window_size]
        win_tim = time_series_timestamps[i:i + window_size]
        target = series[i + window_size : i + window_size + win_out]
        target_timestamps = time_series_timestamps[i + window_size : i + window_size + win_out]
        X.append(win)
        y.append(target)
        X_tim.append(win_tim)
        y_tim.append(target_timestamps)
    return np.array(X),np.array(X_tim), np.array(y), np.array(y_tim)

def sliding_win_clust_aw_target_timestamps(series, time_series_timestamps, window_size_cluster, window_size_pred, win_out_pred):
    assert window_size_cluster > window_size_pred
    X_cluster, X_pred, y = [], [], []
    X_cluster_ts, X_pred_ts, y_ts = [], [], []
    for i in range(len(series) - window_size_cluster - win_out_pred + 1):
        # Generate cluster data
        cluster = series[i:i + window_size_cluster]      
        pred_window = cluster[-window_size_pred:]
        target = series[i + window_size_cluster:i + window_size_cluster + win_out_pred]
        X_cluster.append(cluster)
        X_pred.append(pred_window)
        y.append(target)
        # Generate time series data
        cluster_ts = time_series_timestamps[i:i + window_size_cluster]      
        pred_window_ts = cluster_ts[-window_size_pred:]
        target_ts = time_series_timestamps[i + window_size_cluster:i + window_size_cluster + win_out_pred]
        X_cluster_ts.append(cluster_ts)
        X_pred_ts.append(pred_window_ts)
        y_ts.append(target_ts)
    return np.array(X_cluster), np.array(X_cluster_ts), np.array(X_pred), np.array(X_pred_ts), np.array(y), np.array(y_ts)


if __name__ == "__main__":
    seq = np.array([i for i in range(1,16)])
    seq_ts = np.array([i for i in range(0,15)])
    print("Test clustering")
    win_cluster, win_pred, target = sliding_win_cluster_aware(seq, 5, 3, 1)
    print(win_cluster)
    print(win_pred)
    print(target)
    print("Test clustering with timestamps")
    win_cluster, win_cluster_ts, win_pred, win_pred_ts, target, target_ts = sliding_win_clust_aw_target_timestamps(seq, seq_ts, 5, 3, 1)
    print("Showing win cluster ts")
    for x,y in zip(win_cluster, win_cluster_ts):
        print(x)
        print(y)
    print("Showing win pred ts")
    for x,y in zip(win_pred, win_pred_ts):
        print(x)
        print(y)
    print("Showing targets")
    for x,y in zip(target, target_ts):
        print(x)
        print(y)
    